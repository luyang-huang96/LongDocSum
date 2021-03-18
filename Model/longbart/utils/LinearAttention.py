import torch
import torch.nn as nn
from fairseq.modules import (
MultiheadAttention,
)
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from torch.autograd.function import Function
import torch.functional as F





class LinearAttention(MultiheadAttention):
    # For encoder-decoder attention
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
            qn_block_size=8
    ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv, add_zero_attn, self_attention, encoder_decoder_attention, q_noise, qn_block_size)
        self.elu = nn.ELU()

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False, # True in decoder cross attn
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

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
            src_len, bsz, enc_emb_dim = query.size()
            key = query.permute(1, 2, 0).contiguous() # B * D * T
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.eq(0).type(key.dtype)
                key = key * key_padding_mask.unsqueeze(1).type(key.dtype)
                key_padding_mask = None
            key = key.permute(2, 0, 1).contiguous() # seq_len * B * D
            k = self.k_proj(key)
            v = self.v_proj(key)

        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
                key_padding_mask = None
            else:
                src_len, bsz, enc_emb_dim = key.size()
                key = key.permute(1, 2, 0).contiguous()  # B * D * T
                key_padding_mask = key_padding_mask.eq(0).type(key.dtype)
                key = key * key_padding_mask.unsqueeze(1).type(key.dtype)
                key_padding_mask = None
                key = key.permute(2, 0, 1).contiguous()
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        #q *= self.scaling
        k = self.elu(k)
        q = self.elu(q)

        q = (
            q.contiguous()
            .view(tgt_len, bsz,  self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
            ) # B * L * H * D

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None


        if key_padding_mask is not None:
            #key_padding_mask = None
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        kv = torch.matmul(k.permute(0, 2, 3, 1).contiguous(), v.permute(0, 2, 1, 3).contiguous())
        #kv = torch.einsum("nshd,nshm->nhmd", k, v) # B * H * D * D
        z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + 1e-4)
        #z = 1 / (torch.matmul(q.transpose(3, 2), k.sum(dim=1, keepdim=True)) + 1e-4)
        attn = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z) # B * L * H * D
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)


        return attn, None