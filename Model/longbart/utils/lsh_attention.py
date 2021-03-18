import torch
import torch.nn as nn
from fairseq.modules import (
MultiheadAttention,
)
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from torch.autograd.function import Function
import logging

def _stable_argsort(vector, dim):
    # this function scales the vector so that torch.argsort is stable.
    # torch.argsort is not stable on its own
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)

class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """

    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where
            N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.permute(0, 2, 1, 3)
        return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors.shape) == 3:
            return torch.reshape(vectors, split_dim_shape)
        else:
            raise ValueError("Input vector rank should be one of [3, 4], but is: {}".format(len(vectors.shape)))


class LSHSelfAttention(MultiheadAttention, EfficientAttentionMixin):
    def __init__(self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=True,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        lsh_attn_chunk_length=256,
        num_hashes=4,
        num_buckets=None,
        max_position_embeddings=1026):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout,
                         bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)
        self.config = {}

        self.chunk_length = lsh_attn_chunk_length
        #logging.info('bucket size:', self.chunk_length)
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.max_position_embeddings = max_position_embeddings
        if self.num_buckets is None:
            num_buckets_pow_2 = (2 * ((self.max_position_embeddings-2) // self.chunk_length)).bit_length() - 1
            # make sure buckets are power of 2
            num_buckets = 2 ** num_buckets_pow_2
            self.num_buckets = num_buckets
        self.num_chunks_before = 1
        self.num_chunks_after = 0
        self.hash_seed = None
        self.is_decoder = False

        self.dropout = dropout

        self.num_attention_heads = num_heads
        self.attention_head_size = embed_dim // num_heads
        self.all_head_size = embed_dim
        self.hidden_size = embed_dim

        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        num_hashes = self.num_hashes
        seq_len, bsz, emb_dim = query.size()
        head_mask = None

        query_key_vectors = (self.q_proj(query) + self.k_proj(query)) / 2
        value_vectors = self.v_proj(query)

        # b * l * h * d
        query_key_vectors = query_key_vectors.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        value_vectors = value_vectors.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        # b * h * l * d
        query_key_vectors = query_key_vectors.transpose(2, 1)
        value_vectors = value_vectors.transpose(2, 1)

        do_standard_self_attention = seq_len <= self.chunk_length

        if key_padding_mask is not None:
            attention_mask = key_padding_mask == 0
        else:
            attention_mask = None
        if not do_standard_self_attention:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask) #(Batch_size x Num_Attn_Heads x (Num_Hashes x Seq_Len))

            assert (
                    int(buckets.shape[-1]) == num_hashes * seq_len
            ), "last dim of buckets is {}, but should be {}".format(buckets.shape[-1], num_hashes * seq_len)

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                seq_len, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % seq_len

            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = torch.arange(seq_len, device=query_key_vectors.device).repeat(
                bsz, self.num_attention_heads, 1
            )

        key_vectors = self._len_and_dim_norm(query_key_vectors)

        query_vectors = query_key_vectors

        # free memory
        del query_key_vectors

        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=head_mask,
            do_standard_self_attention=do_standard_self_attention,
        )

        # free memory
        del key_vectors, value_vectors

        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention:
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    seq_len,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    seq_len,
                    self.num_attention_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            bsz,
            self.num_attention_heads,
            seq_len,
            self.attention_head_size,
        ), "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`."



        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)
        assert list(out_vectors.size()) == [bsz, seq_len, emb_dim]
        out_vectors = out_vectors.transpose(0, 1).contiguous()
        out_vectors = self.out_proj(out_vectors)

        if need_weights:
            return out_vectors, attention_probs.sum(dim=1) / self.num_heads
        else:
            return out_vectors, None



    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                self.num_buckets % 2 == 0
            ), "There should be an even number of bucktes, but `self.num_bucktes`: {}".format(self.num_buckets)
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, "The number of buckets should be even, but `num_bucket`: {}".format(
                    bucket_factor
                )
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)

        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(
                buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device)
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # no gradients are needed
        with torch.no_grad():
            # hash-based sort
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)

    def _len_and_dim_norm(self, vectors):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        do_standard_self_attention
    ):
        # look at previous and following chunks if chunked attention
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        # (BS, NumAttn, NumHash x NumChunk, Chunk_L x Hidden),(BS, NumAttn, NumHash x NumChunk, Chunk_L * (Num_bef + Num_aft + 1) x Hidden) -> (BS, NumAttn, NumHash x NumChunk, Chunk_L, Chunk_L * (1 + Num_bef + Num_aft))
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads
            )
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32


        mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention,
            )

        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        # apply self_mask
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = self.dropout_module(attention_probs)
        #attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):
        # attention mask for LSH
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]
            if not do_standard_self_attention:
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # extract attention mask from LSH sorted key_indices
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask
        # if self.is_decoder is True:
        #     causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
        #
        #     # add attention mask if not None
        #     if attention_mask is not None:
        #         attention_mask = causal_mask * attention_mask
        #     else:
        #         attention_mask = causal_mask

        return attention_mask

class ReverseSort(Function):
    """
    After chunked attention is applied which sorted clusters,
    original ordering has to be restored.
    Since customized backward function is used for Reformer,
    the gradients of the output vectors have to be explicitely
    sorted here.
    """

    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        # save sorted_bucket_idx for backprop
        with torch.no_grad():
            ctx.sorted_bucket_idx = sorted_bucket_idx

            # undo sort to have correct order for next layer
            expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
            out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
            logits = torch.gather(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits):
        # get parameters saved in ctx
        sorted_bucket_idx = ctx.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = torch.gather(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = torch.gather(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None