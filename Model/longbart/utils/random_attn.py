import torch
import torch.nn as nn
from fairseq.modules import (
MultiheadAttention,
)
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from torch.autograd.function import Function
from functools import partial, wraps, reduce
from operator import mul
from inspect import isfunction
import torch.nn.functional as F
import logging

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def generate_mask(bucket_size, mask_window, perm_inds):
    bsz, buckets = perm_inds.size()
    all_mask = torch.ones(bucket_size, bucket_size)
    after_mask = torch.tril(all_mask, diagonal=0)
    before_mask = torch.tril(all_mask, diagonal=0).transpose(0, 1)
    non_mask = torch.zeros(bucket_size, bucket_size)
    bsz_masks = []
    for i in range(bsz):
        masks = []
        for j in range(buckets):
            ind = perm_inds[i, j]
            if ind < j + mask_window and ind > j - mask_window:
                masks.append(all_mask)
            elif ind == j + mask_window:
                masks.append(after_mask)
            elif ind == j - mask_window:
                masks.append(before_mask)
            else:
                masks.append(non_mask)
        masks = torch.stack(masks, dim=0)
        bsz_masks.append(masks)
    bsz_masks = torch.stack(bsz_masks, dim=0)
    return bsz_masks


def random_attention_qk(q, k, bucket_size, window_size, key_padding_mask):
    bh, seq_len, head_dim = q.size()
    bsz = key_padding_mask.size(0)
    num_heads = bh // bsz
    buckets = seq_len // bucket_size
    mask_window = window_size // bucket_size - 1
    b_q = bucket(buckets, q)
    b_k = bucket(buckets, k) # BH * bct * n_b * D

    perm = torch.eye(buckets)
    perm_inds = torch.randint(0, buckets, size=(bh, buckets))
    perm = [perm[perm_inds[i, :], :] for i in range(bh)]
    perm = torch.stack(perm, dim=0).to(q.device).type_as(q)

    bsz_masks = generate_mask(bucket_size, mask_window, perm_inds).bool().to(q.device)

    b_k_r = reorder_buckets(b_k, perm).reshape(bh, buckets, -1, head_dim)
    dots = torch.einsum('buie,buje->buij', b_q, b_k_r) # BH * bct * n_b * n_b

    dots = dots.masked_fill_(bsz_masks, value=-10000.)

    if key_padding_mask is not None:
        q_mask = key_padding_mask.eq(0)
        kv_mask = q_mask
        mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)  # B * bkt * n_b
        expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, num_heads))
        mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))  # BH * bkt * n_b
        mk = batched_index_select(mk, perm.argmax(dim=-1))
        mk = mk.reshape(bh, buckets, -1)
        mask = mq[:, :, :, None] * mk[:, :, None, :]
        dots.masked_fill_(~mask, -10000.)
        del mask

    return perm, dots.reshape(bh, seq_len, bucket_size).contiguous()

def random_attention_pv(p, v, bucket_size, perm):
    bh, seq_len, head_dim = v.size()
    buckets = seq_len // bucket_size

    b_v = bucket(buckets, v)
    b_p = bucket(buckets, p)
    b_v_r = reorder_buckets(b_v, perm).reshape(bh, buckets, -1, head_dim)

    attn = torch.einsum('buij,buje->buie', b_p, b_v_r)

    attn = unbucket(attn)

    return attn.view(bh, seq_len, head_dim).contiguous()