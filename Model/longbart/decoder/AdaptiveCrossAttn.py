from fairseq.modules import (
LayerDropModuleList,
TransformerEncoderLayer,
MultiheadAttention,
PositionalEmbedding,
SinusoidalPositionalEmbedding
)
import logging
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from fairseq import utils
from typing import Any, Dict, List, Optional, Tuple


class DivideMultiheadAttention(MultiheadAttention):
    def __init__(
                self,
                embed_dim,
                num_heads,
                kdim=None,
                vdim=None,
                dropout=0.0,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                self_attention=False,
                encoder_decoder_attention=False,
                q_noise=0.0,
                qn_block_size=8,
                divide_type='stride',
                encoder_max_positions=1024,
                divide_ratio=4
        ):
        super(DivideMultiheadAttention, self).__init__(embed_dim,
                num_heads,
                kdim,
                vdim,
                dropout,
                bias,
                add_bias_kv,
                add_zero_attn,
                self_attention,
                encoder_decoder_attention,
                q_noise,
                qn_block_size)
        self.divide_type = divide_type
        assert divide_type in ['stride', 'normal', 'lsh', 'adaptive', 'cluster']
        self.block_num = divide_ratio
        logging.info('Divide Type: {}'.format(divide_type))
        logging.info('Block num: {}'.format(self.block_num))
        self.num_buckets = 64
        # if divide_type == 'lsh':
        #     self.head_query = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        #     nn.init.xavier_uniform_(self.head_query)
        #     self.chunk_length = 256
        if divide_type == 'adaptive':
            self.head_query = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
            #self.head_query = nn.Linear(self.num_heads*self.head_dim, self.num_heads)
            nn.init.uniform_(self.head_query)
            self.chunk_length = encoder_max_positions // self.block_num
            logging.info('Chunk length: {}'.format(self.chunk_length))

        if self.divide_type == 'cluster':
            self.commitment = 1e-4
            self.ema_decay = 0.999
            assert self.num_heads % self.block_num == 0
            num_cluster = self.num_heads // self.block_num
            self.chunk_length = encoder_max_positions // self.block_num
            # self.means = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
            self.register_buffer('means', torch.randn(self.num_heads, self.head_dim))
            self.register_buffer('initted', torch.tensor(False))
            self.num_new_means = 0
            self.new_means = None


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        aux_loss = None


        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None


        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.divide_type in ['normal', 'stride']:
                    input_len, bsz, key_dim = key.size()
                    if input_len // self.block_num != 0:
                        padding_size = (input_len // self.block_num + 1) * self.block_num - input_len
                        key = torch.cat([key, torch.zeros(padding_size, bsz, key_dim, dtype=key.dtype).to(key.device)], dim=0)
                        key_padding_mask = torch.cat([key_padding_mask, torch.zeros(bsz, padding_size, dtype=key_padding_mask.dtype).to(key_padding_mask.device)], dim=1)
                    input_len = key.size(0)
                    block_len = input_len // self.block_num
                    with torch.no_grad():
                        indexes = torch.linspace(1, input_len, steps=input_len)-1
                        if self.divide_type == 'stride':
                            indexes = indexes.reshape(block_len, self.block_num).repeat(1, self.num_heads // self.block_num)
                        elif self.divide_type == 'normal':
                            indexes = indexes.reshape(self.block_num, block_len).repeat(self.num_heads // self.block_num, 1).transpose(0, 1)
                        else:
                            raise NotImplementedError
                        indexes = indexes.long()
                k = self.k_proj(key)
                v = self.v_proj(key) # L*B*D

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )

        if k is not None:
            k = k.view(-1, bsz, self.num_heads, self.head_dim)
            if self.divide_type == 'lsh':
                indexes = self._has_vector(self.head_query, k, num_hashes=5) # B * num_attn_heads * chunk_length
                k = k.gather(dim=0, index=indexes.permute(2, 0, 1).unsqueeze(3).expand(-1, -1, -1, self.head_dim))
            elif self.divide_type == 'adaptive':
                indexes, key_probs = self.adaptive_select(k, self.chunk_length) # B * num_attn_heads * chunk_length
                k = k.gather(dim=0, index=indexes.permute(2, 0, 1).unsqueeze(3).expand(-1, -1, -1, self.head_dim))
            elif self.divide_type == 'cluster':
                indexes = self.kmeans_select(q.view(bsz, self.num_heads, tgt_len, self.head_dim).contiguous(), k.permute(1,2,0,3).contiguous())
                k = k.gather(dim=0, index=indexes.permute(2, 0, 1).unsqueeze(3).expand(-1, -1, -1, self.head_dim))
            else:
                k = [k[indexes[:, i], :, i, :].unsqueeze(2) for i in range(self.num_heads)]
                k = torch.cat(k, dim=2)
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = v.view(-1, bsz, self.num_heads, self.head_dim)
            if self.divide_type in ['lsh', 'adaptive', 'cluster']:
                v = v.gather(dim=0, index=indexes.permute(2, 0, 1).unsqueeze(3).expand(-1, -1, -1, self.head_dim))
            else:
                v = [v[indexes[:, i], :, i, :].unsqueeze(2) for i in range(self.num_heads)]
                v = torch.cat(v, dim=2)
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if key_padding_mask is not None and k is not None:
            if self.divide_type in ['lsh', 'adaptive', 'cluster']:
                with torch.no_grad():
                    key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1).gather(dim=2, index=indexes).transpose(1, 2).contiguous()
            else:
                key_padding_mask = [key_padding_mask[:, indexes[:, i]].unsqueeze(2) for i in range(self.num_heads)]
                key_padding_mask = torch.cat(key_padding_mask, dim=2)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            if prev_key_padding_mask is not None and static_kv:
                key_padding_mask = prev_key_padding_mask
            else:
                key_padding_mask = key_padding_mask
            # key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
            #     key_padding_mask=key_padding_mask,
            #     prev_key_padding_mask=prev_key_padding_mask,
            #     batch_size=bsz,
            #     src_len=k.size(1),
            #     static_kv=static_kv,
            # )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            if self.divide_type == 'adaptive':
                if 'adaptive_prob' in saved_state:
                    key_probs = saved_state['adaptive_prob']
                saved_state['adaptive_prob'] = key_probs
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        #attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                if len(key_padding_mask.size()) == 3:
                    key_padding_mask = key_padding_mask.transpose(1, 2).unsqueeze(2)
                elif len(key_padding_mask.size()) == 2:
                    key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # if self.divide_type == 'adaptive':
        #     key_probs = key_probs.view(-1, src_len).contiguous().unsqueeze(1)
        #     attn_weights = attn_weights * key_probs

        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        if self.divide_type == 'adaptive':
            key_probs = key_probs.view(-1, src_len).contiguous().unsqueeze(1)
            attn_weights_float = attn_weights_float * key_probs.type_as(attn_weights_float)
            attn_weights_float = attn_weights_float / attn_weights_float.sum(dim=-1, keepdim=True)



        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
            #print('attn prob:', attn_probs[0, 0, :])

        if self.divide_type == 'cluster':
            aux_loss = self.update_center(attn_probs.view(bsz, self.num_heads, tgt_len, src_len).contiguous(),
                               k.view(bsz, self.num_heads, src_len, self.head_dim).contiguous())


        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder stp (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, aux_loss

    def _has_vector(self, query, key, num_hashes):
        rotation_size = self.num_buckets

        query = query # H * D
        key = key.detach().permute(1, 2, 0, 3).contiguous()
        bsz = key.size(0)
        assert query.shape[-1] == query.shape[-1]
        rotations_shape = (self.num_heads, query.shape[-1], num_hashes, rotation_size // 2)
        random_rotations = torch.randn(rotations_shape, device=query.device, dtype=query.dtype)

        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        query_rotated_vectors = torch.einsum("md,mdhr->mhr", query, random_rotations) # n_head * Num_Hashes * Num_Buckets/2
        key_rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", key, random_rotations)
        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            query_rotated_vectors = torch.cat([query_rotated_vectors, -query_rotated_vectors], dim=-1) # n_head * Num_Hashes
            query_buckets = torch.argmax(query_rotated_vectors, dim=-1)
            key_rotated_vectors = torch.cat([key_rotated_vectors, -key_rotated_vectors], dim=-1)
            key_buckets = torch.argmax(key_rotated_vectors, dim=-1)
        #Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len
        else:
            raise NotImplementedError

        combined_buckets = query_buckets.unsqueeze(0).unsqueeze(3) - key_buckets #Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len
        combined_buckets = combined_buckets.abs()
        sorted_buckets, _ = combined_buckets.min(dim=2)#Batch_Size x Num_Attn_Heads x Seq_Len

        def _stable_argsort(vector, dim):
            # this function scales the vector so that torch.argsort is stable.
            # torch.argsort is not stable on its own
            scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
            scale_offset = scale_offset.expand(vector.shape)
            scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
            return torch.argsort(scaled_vector, dim=dim)

        idx = _stable_argsort(sorted_buckets, dim=-1) # B * num_attn_heads * input_len * target_len
        # mask = idx.new_zeros(idx.size())
        # ones = torch.ones(idx.size(), dtype=mask.dtype).to(mask.device)
        # mask.scatter_(3, idx[:, :, :, :self.chunk_length], ones)
        attention_index = idx[:, :, :self.chunk_length] # B * num_attn_heads * chunk_length
        return attention_index

    def adaptive_select(self, k, chunk_length):
        #seq_len, bsz, _, _ = k.size()
        probs = torch.einsum('lbhd,hd->lbh', k, self.head_query)
        probs = torch.sigmoid(probs) # seq_len * bsz * head_num
        probs = probs.permute(1, 2, 0).contiguous() # b*h*l
        def _stable_argsort(vector, dim):
            # this function scales the vector so that torch.argsort is stable.
            # torch.argsort is not stable on its own
            scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
            scale_offset = scale_offset.expand(vector.shape)
            scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
            return torch.argsort(scaled_vector, dim=dim, descending=True)
        with torch.no_grad():
            #indexes = _stable_argsort(probs.detach(), dim=2)
            indexes = torch.argsort(probs.detach(), dim=2, descending=True)
            indexes = indexes[:, :, :chunk_length].detach() # b*h*chunk_len
        probs = probs.gather(dim=2, index=indexes)

        return indexes, probs

    def kmeans_select(self, q, k):
        # q, k: bhld
        self.kmeans_init(k.detach())

        b, dtype = k.size(0), k.dtype
        means = self.means.type(dtype)
        k = F.normalize(k, 2, dim=-1)

        with torch.no_grad():
            dists = torch.einsum('bhld,hd->bhl', k, means).transpose(1, 2) #blh
            # _, buckets = torch.max(dists, dim=-1)
        if self.chunk_length < k.size(2):
            chunk_length = self.chunk_length
        else:
            chunk_length = k.size(2)
        _, topk_indices = dists.topk(k=chunk_length, dim=-2) #b*chunk*h

        #routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        return topk_indices.permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def kmeans_init(self, k):
        if self.initted:
            return
        means = k.transpose(0, 1).reshape(self.num_heads, -1, self.head_dim)
        num_samples = means.size(1)
        indices = torch.randint(0, num_samples, (self.num_heads, 1), device=k.device)
        indices = indices.unsqueeze(2).expand(-1, -1, self.head_dim)
        means = means.gather(dim=1, index=indices).squeeze(1)
        for _ in range(10):
            means = kmeans_iter(k, means, self.chunk_length)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))


    def update_center(self, attn_prob, k):
        # attn_prob: bsz*num_heads*tgt_len*src_len
        # k: b*n*src_len*d
        means = torch.einsum('bnts,bnsd->bntd', attn_prob, k) # bntd
        # norm_means = F.normalize(means, 2, dim=-1) #bntd
        # bsz = norm_means.size(0)
        # tgt_len = norm_means.size(2)
        # aux_loss = F.mse_loss(norm_means,
        #                       self.means.unsqueeze(0).unsqueeze(2).expand(bsz,-1, tgt_len, -1)
        #                       ) * self.commitment
        aux_loss = None

        norm_center = F.normalize(means.mean(dim=-2).sum(dim=0), 2, dim=-1) # nd
        self.means = self.means * self.ema_decay + norm_center * (1-self.ema_decay)
        return aux_loss







def expand_dim(t, dim, dim2):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = dim2
    return t.expand(*expand_shape)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))

def kmeans_iter(x, means, chunk_length):
    b, h, l, d, dtype = *x.shape, x.dtype
    means = means.unsqueeze(1)

    dists = torch.einsum('bhld,hcd->bhlc', x, means) # bhlc
    if chunk_length < dists.size(2):
        chunk_length = dists.size(2)
    _, buckets = dists.topk(k=chunk_length, dim=-2) # b*h*chunk*c
    means_ = x.gather(dim=-2, index=buckets.expand(-1, -1, -1, d)).mean(dim=-2)

    means = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = means.squeeze(0)
    return means
