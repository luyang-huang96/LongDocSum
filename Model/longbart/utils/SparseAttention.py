import torch


def strided_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, remove_from_windowed_attention_mask: torch.Tensor):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    q = q.view(bsz * num_heads, chunks_count, w, head_dim).transpose(1, 2)
    k = k.view(bsz * num_heads, chunks_count, w, head_dim).transpose(1, 2)

    # matrix multipication
    # bcxd: bsz*num_heads x w x chunk x head_dim
    # bcyd: bsz*num_heads x w x chunk  x head_dim
    # bcxy: bsz*num_heads x w x chunk x chunk
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (q, k))  # multiply
    if remove_from_windowed_attention_mask is not None:
        mask_size = chunks_count // w
        remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.squeeze().view(bsz,
                                              chunks_count, w).transpose(1, 2).unsqueeze(3)
        float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.)
        ones = float_mask.new_ones(size=float_mask.size())
        mask_attn = torch.einsum('bcxd,bcyd->bcxy', (ones, float_mask))
        # regular_mask = torch.tensor(
        #     [[-10000. if i == j or i + 1 == j or i - 1 == j else 0 for j in range(chunks_count)] for i in range(chunks_count)])
        regular_mask = torch.tensor(
            [[-10000. if i < j + mask_size and i >  j - mask_size else 0. for j in range(chunks_count)] for i in range(chunks_count)])
        mask_attn = mask_attn + regular_mask.type_as(mask_attn).unsqueeze(0).unsqueeze(1)
        mask_attn = mask_attn.repeat(num_heads, 1, 1, 1)
        chunk_attn = chunk_attn + mask_attn
    chunk_attn = chunk_attn.transpose(1, 2)
    chunk_attn = chunk_attn.reshape(bsz, num_heads, seqlen, chunks_count).transpose(2, 1)

    return chunk_attn

def strided_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    chunks_count = seqlen // w
    assert seqlen % w == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == chunks_count

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen, chunks_count)
    prob = prob.view(bsz * num_heads, chunks_count, w, chunks_count).transpose(1, 2)

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    v = v.view(bsz * num_heads, chunks_count, w, head_dim).transpose(1, 2)


    # chunk padded_v into chunks of size 3w and an overlap of size w

    context = torch.einsum('bcwd,bcdh->bcwh', (prob, v)) # (b*num_heads)*w*c*head_dim
    context = context.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)