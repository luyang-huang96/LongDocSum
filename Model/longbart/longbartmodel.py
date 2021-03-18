from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart.model import BARTModel, bart_large_architecture
from fairseq.models.transformer import TransformerEncoder, EncoderOut
from fairseq.modules import (
LayerDropModuleList,
TransformerEncoderLayer,
MultiheadAttention,
PositionalEmbedding,
SinusoidalPositionalEmbedding
)
import logging
import math


from torch import Tensor, nn
from typing import Dict, Optional, Tuple
import torch
from .utils.sliding_trunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
import torch.nn.functional as F
from .utils.sparse_decoder import SparseTransformerDecoder, LinearMultiheadAttention
from fairseq.models.bart.hub_interface import BARTHubInterface
import math
from .hub_interface import LongBARTHubInterface
from .utils.SparseAttention import strided_chunks_matmul_qk, strided_chunks_matmul_pv
from .utils.lsh_attention import LSHSelfAttention
from .utils.LinearAttention import LinearAttention
from .utils.sinkhorn import SinkhornSelfAttention
from fairseq.checkpoint_utils import prune_state_dict
from .utils.graph_encoder import TransformerEncoderWizGraph
from .utils.random_attn import random_attention_pv, random_attention_qk

@register_model('longbart')
class LongBartModel(BARTModel):

    @staticmethod
    def add_args(parser):
        super(LongBartModel, LongBartModel).add_args(parser)
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--lsh_attention', action='store_true', help='use lsh attention')
        group.add_argument('--strided_attention', action='store_true', help='strided pattern')
        group.add_argument('--random_attention', action='store_true', help='random pattern')
        group.add_argument('--encoder_linear', action='store_true')
        group.add_argument('--encoder_kernel_linear', action='store_true')
        group.add_argument('--sinkhorn', action='store_true')
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--decoder_divide_attention', action='store_true')
        group.add_argument('--decoder_linear_attention', action='store_true', help='decoder use linear attention')
        group.add_argument('--decoder_importance_sample', action='store_true', help='decoder use linear attention')

        parser.add_argument('--sw', action='store_true', help='sliding window')
        parser.add_argument('--window', default=['128'], nargs='+', help='sliding window attention window size')
        parser.add_argument('--compress_dim', default=1024, type=int, help='linear projection size')
        parser.add_argument('--divide_ratio', default=4, type=int, help='linear projection size')
        parser.add_argument('--divide_type', default='stride', type=str)
        parser.add_argument('--global_attention', action='store_true', help='add global attention')
        parser.add_argument('--adaptive_span', action='store_true', help='adaptive span')
        parser.add_argument('--adaptive_type', action='store', default='head', type=str, help='adaptive span type')
        parser.add_argument('--encoder_compress_dim', action='store', default=256, type=int)
        parser.add_argument('--encoder_not_hybrid', action='store_true', help='encoder uses hybrid layers')

        parser.add_argument('--bucket_size', action='store', default=64, type=int, help='lsh/sinkhorn bucket')

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        strict = False
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict)
        if 'encoder.embed_positions_new.weight' in missing_keys:
            self.encoder.align_embed_position()
        if len(unexpected_keys) > 0:
            raise Exception
        return []


    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        args.graph_encoding = getattr(args, 'graph_encoding', False)
        if args.graph_encoding:
            return TransformerEncoderWizGraph(args, src_dict, embed_tokens)
        else:
            return SparseTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SparseTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        length=1024,
        limit_length=False,
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        #print(x['models'][0].encoder._window)
        print(x['models'][0].encoder.layers[:2])
        print(x['models'][0].decoder.layers[:2])
        #raise Exception
        x['args'].limit_length = limit_length
        x['args'].length = length
        return LongBARTHubInterface(x['args'], x['task'], x['models'][0])

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True



        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra



@register_model_architecture('longbart', 'longbart')
def longbart_architecture(args):
    # args.max_target_positions = 1024
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    bart_large_architecture(args)



class SparseTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        assert self.num_layers % len(args.window) == 0
        duplicate = self.num_layers // len(args.window)
        self._window = [int(args.window[i // duplicate]) for i in range(self.num_layers)]
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        args.sw = getattr(args, 'sw', False)
        args.lsh_attention = getattr(args, 'lsh_attention', False)
        args.strided_attention = getattr(args, 'strided_attention', False)
        args.random_attention = getattr(args, 'random_attention', False)
        args.adaptive_span = getattr(args, 'adaptive_span', False)
        args.encoder_linear = getattr(args, 'encoder_linear', False)
        args.encoder_kernel_linear = getattr(args, 'encoder_kernel_linear', False)
        args.sinkhorn = getattr(args, 'sinkhorn', False)
        args.encoder_not_hybrid = getattr(args, 'encoder_not_hybrid', False)
        assert not (args.encoder_linear and args.sw)
        assert not (args.lsh_attention and args.encoder_linear)
        assert not ((args.encoder_kernel_linear and args.encoder_linear) or (args.encoder_kernel_linear and args.sw) or args.encoder_kernel_linear and args.lsh_attention)
        if args.adaptive_span:
            logging.info('Use adaptive span')

        if args.lsh_attention:
            if args.encoder_not_hybrid:
                self.layers.extend(
                    [self.build_lsh_encoder_layer(
                        args, self.padding_idx) for i in
                     range(args.encoder_layers)]
                )
            else:
                self.layers.extend(
                [self.build_sparse_encoder_layer(args, self._window[i], self.padding_idx) if i % 2 == 0 else self.build_lsh_encoder_layer(args, self.padding_idx) for i in
                 range(args.encoder_layers)]
            )
        elif args.sinkhorn:
            if args.encoder_not_hybrid:
                self.layers.extend(
                    [self.build_sinkhorn_layer(
                        args, self.padding_idx)
                     for i in
                     range(args.encoder_layers)]
                )
            else:
                self.layers.extend(
                [self.build_sparse_encoder_layer(args, self._window[i], self.padding_idx) if i % 2 == 0 else self.build_sinkhorn_layer(args, self.padding_idx)
                 for i in
                 range(args.encoder_layers)]
            )
        elif args.sw or args.encoder_linear or args.encoder_kernel_linear:
            self.layers.extend(
            [self.build_sparse_encoder_layer(args, self._window[i], self.padding_idx) for i in range(args.encoder_layers)]
        )
        else:
            self.layers.extend(
                [self.build_encoder_layer(args) for i in
                 range(args.encoder_layers)]
            )
        self.num_layers = len(self.layers)

        embed_dim = embed_tokens.embedding_dim
        self.embed_positions = (
            PositionalEmbedding(
                1024,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        ) # initialize for bart large base not implemented

        self.embed_positions_new = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        nn.init.xavier_uniform_(self.embed_positions_new.weight)
        self.left_pad_source = args.left_pad_source
        self.section = args.sec_emb
        if self.section:
            self.embed_section = (
            PositionalEmbedding(
                args.max_source_positions + 2,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
            nn.init.normal_(self.embed_section.weight, mean=0, std=embed_dim ** -0.5)
            nn.init.constant_(self.embed_section.weight[self.padding_idx], 0)
            nn.init.xavier_uniform_(self.embed_section.weight)

    def build_lsh_encoder_layer(self, args, padding_idx):
        return LSHTransformerEncoderLayer(args, padding_idx)

    def build_sparse_encoder_layer(self, args, window_size, padding_idx):
        return SparseTransformerEncoderLayer(args, window_size, padding_idx)

    def build_sinkhorn_layer(self, args, padding_idx):
        return SinkhornTransformerEncoderLayer(args, padding_idx)

    def align_embed_position(self):
        self.embed_positions_new.weight.data[:1026, :] = self.embed_positions.weight.data
        self.embed_positions_new.weight.data[1026:, :] = self.embed_positions.weight.data[-1][None, :].repeat(self.max_source_positions-1024, 1)
        if self.section:
            self.embed_section.weight.data[4:1028, :] = self.embed_positions.weight.data[2:, :]
            self.embed_section.weight.data[0:2, :] = self.embed_positions.weight.data[0:2, :]
            self.embed_section.weight.data[1028:, :] = self.embed_positions.weight.data[-1][None, :].repeat(self.max_source_positions-1026, 1)
        # self.embed_positions = self.embed_positions_new

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # padding
        bsz, src_len = src_tokens.size()
        if self.section:
            sec_tokens = kwargs['sec']
        if src_len % (max(self._window)*2) != 0:
            chunks = src_len // (max(self._window)*2) + 1
            if self.left_pad_source:
                src_tokens = torch.cat([torch.zeros(bsz, chunks*max(self._window)*2-src_len, device=src_tokens.device).fill_(self.padding_idx).long(), src_tokens], dim=1)
                if self.section:
                    sec_tokens = torch.cat([torch.zeros(bsz, chunks*max(self._window)*2-src_len, device=sec_tokens.device).fill_(self.padding_idx).long(), sec_tokens], dim=1)
            else:
                src_tokens = torch.cat([src_tokens, torch.zeros(bsz, chunks * max(self._window) * 2 - src_len, device=src_tokens.device).fill_(self.padding_idx).long()], dim=1)
                if self.section:
                    sec_tokens = torch.cat([sec_tokens, torch.zeros(bsz, chunks * max(self._window) * 2 - src_len, device=sec_tokens.device).fill_(self.padding_idx).long()], dim=1)


        if self.section:
            x, encoder_embedding = self.forward_embedding(src_tokens, sec_tokens)
        else:
            x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )
    def forward_embedding(self, src_tokens, secs=None):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions_new is not None:
            x = embed + self.embed_positions_new(src_tokens)
        if self.section:
            x = x + self.embed_section(secs)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions_new.max_positions)


class SinkhornTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, padding_idx):
        super().__init__(args)
        self.padding_idx = padding_idx
        if args.sinkhorn:
            self.self_attn  = self.build_sinkhorn_self_attention(self.embed_dim, args)
        else:
            raise Exception

    def build_sinkhorn_self_attention(self, embed_dim, args):
        return SinkhornSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            bucket_size=args.bucket_size
        )

class LSHTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, padding_idx):
        super().__init__(args)
        self.padding_idx = padding_idx
        if args.lsh_attention:
            self.self_attn  = self.build_lsh_self_attention(self.embed_dim, args)
        else:
            raise Exception

    def build_lsh_self_attention(self, embed_dim, args):
        return LSHSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            lsh_attn_chunk_length=args.bucket_size,
            num_hashes=4,
            num_buckets=None,
            max_position_embeddings=args.max_source_positions+2
        )


class SparseTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, window_size, padding_idx):
        super().__init__(args)
        self.window_size = window_size
        self.padding_idx = padding_idx
        if args.sw:
            self.self_attn  = self.build_sparse_self_attention(self.embed_dim, window_size, padding_idx, args)
        elif args.encoder_linear:
            self.self_attn = self.build_linear_self_attention(self.embed_dim, args)
        elif args.encoder_kernel_linear:
            self.self_attn = self.build_kernel_linear_self_attention(self.embed_dim, args)
        else:
            raise NotImplementedError

    def build_kernel_linear_self_attention(self, embed_dim, args):
        return LinearAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size
        )


    def build_linear_self_attention(self, embed_dim, args):
        return LinearMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            compress_dim=args.encoder_compress_dim,
            max_source_position=args.max_source_positions

        )

    def build_sparse_self_attention(self, embed_dim, window_size, padding_idx, args):
        return SparseSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            window_size=window_size,
            padding_idx=padding_idx,
            global_attention=getattr(args, "global_attention", False),
            adaptive_span=args.adaptive_span,
            adaptive_type=args.adaptive_type,
            strided_attention=args.strided_attention,
            max_source_positions=args.max_source_positions,
            random_attention=args.random_attention
        )



class SparseSelfAttention(MultiheadAttention):
    def __init__(self,
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
        window_size=256,
        padding_idx=1,
        global_attention=False,
        adaptive_span=False,
        adaptive_type='head',
        strided_attention=False,
        random_attention=False,
        max_source_positions=1024):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout,
                         bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)
        self.attention_window = window_size
        self.padding_idx = padding_idx


        self.global_attention = global_attention
        self.global_tokens = 128
        self.attention_dilation = 1

        self.stride_chunk = max_source_positions // self.global_tokens
        if window_size % self.stride_chunk != 0:
            self.stride_chunk = 2 ** math.floor(math.log2(self.stride_chunk))

        if global_attention:
            logging.info('use global attention')

        self.adaptive_span = adaptive_span
        self.strided_attention = strided_attention
        self.random_attention = random_attention
        if random_attention:
            logging.info('random chunk size: {}'.format(self.global_tokens))
        if strided_attention:
            logging.info('stride chunk size: {}'.format(self.stride_chunk))
        if adaptive_span:
            self.adaptive_mask = AdaptiveMask(max_size=window_size, _type=adaptive_type, head_dim=self.head_dim, emb_dim=embed_dim, head_num=num_heads)

    def forward(self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,):
        # takes from longformer implementation

        if need_head_weights:
            need_weights = True

        # attn_mark None
        # key padding mask 0,1 bool tensor 1 means masked position (bsz x seqlen)


        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attention_mask = attn_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask == 0

        if self.global_attention:
            extra_attention_mask = key_padding_mask == 0
            remove_from_windowed_attention_mask = key_padding_mask == 1
            extra_attention_mask[:, self.global_tokens:] = 0 # do not use left source padding, use first x tokens for global attention
            remove_from_windowed_attention_mask = (remove_from_windowed_attention_mask + extra_attention_mask) > 0

            extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
            zero_to_max_range = torch.arange(0, self.global_tokens, device=extra_attention_mask.device)
            # mask indicating which values are actually going to be padding
            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
            # 2) location of the non-padding values in the selected global attention
            selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
            # 3) location of the padding values in the selected global attention
            selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = key_padding_mask == 1
            extra_attention_mask = None



        seq_len, bsz, embed_dim = query.size()

        assert self.self_attention
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)

        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size()) == [bsz, seq_len, self.num_heads, self.attention_window * 2 + 1]

        if self.global_attention:
            # this part can be viewed as the column of attention matrix
            selected_k = k.new_zeros(bsz, self.global_tokens, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            global_attn_num = selected_attn_weights.size(-1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        if self.strided_attention:
            # q: seq_len, bsz, num_heads, head_dim
            strided_attn_weights = strided_chunks_matmul_qk(q, k, self.stride_chunk, remove_from_windowed_attention_mask)
            attn_weights = torch.cat((strided_attn_weights, attn_weights), dim=-1)
            chunks_count = strided_attn_weights.size(-1)
            strided_num = strided_attn_weights.size(-1)

        if self.random_attention:
            perm, random_attn_weights = random_attention_qk(q.transpose(0, 1).view(seq_len, bsz*self.num_heads, self.head_dim).contiguous().transpose(0, 1),
                                                            k.transpose(0, 1).view(seq_len, bsz*self.num_heads, self.head_dim).contiguous().transpose(0, 1),
                                                            self.global_tokens,
                                                            self.attention_window,
                                                            key_padding_mask)
            random_attn_weights = random_attn_weights.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
            attn_weights = torch.cat((random_attn_weights, attn_weights), dim=-1)
            random_num = random_attn_weights.size(-1)



        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if self.adaptive_span:
            extra_attn_num = 0
            if self.global_attention:
                extra_attn_num += global_attn_num
            if self.strided_attention:
                extra_attn_num += strided_num
            if self.random_attention:
                extra_attn_num += random_num


            attn_weights_float = self.adaptive_mask(attn_weights_float, words=query.transpose(0, 1), extra_attn_num=extra_attn_num)

        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1),
                                                   0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        #attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        attn_probs = self.dropout_module(attn_weights)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0

        if self.random_attention:
            selected_random_probs = attn_probs.narrow(-1, 0, self.global_tokens)
            attn = random_attention_pv(selected_random_probs.transpose(1, 2).reshape(bsz*self.num_heads, seq_len, -1),
                                       v.transpose(0, 1).view(bsz*self.num_heads, seq_len, self.head_dim).contiguous(),
                                       self.global_tokens,
                                       perm).view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, self.global_tokens, attn_probs.size(-1) - self.global_tokens)

        if self.strided_attention:
            selected_strided_probs = attn_probs.narrow(-1, 0, chunks_count)
            attn += strided_chunks_matmul_pv(selected_strided_probs, v, self.stride_chunk)
            attn_probs = attn_probs.narrow(-1, chunks_count, attn_probs.size(-1) - chunks_count)


        if self.global_attention:
            # this part can be viewed as the column of attention matrix
            selected_attn_probs = attn_probs.narrow(-1, 0, self.global_tokens)
            selected_v = v.new_zeros(bsz, self.global_tokens, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn += torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, self.global_tokens, attn_probs.size(-1) - self.global_tokens).contiguous()


        attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)

        attn = attn.type_as(query)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()


        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor.
        if self.global_attention:
            # this part can be viewed as the row of attention matrix
            selected_hidden_states = query.new_zeros(self.global_tokens, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = query[extra_attention_mask_nonzeros[::-1]]
            q = self.q_proj(selected_hidden_states)
            k = self.k_proj(query)
            v = self.v_proj(query)
            q /= math.sqrt(self.head_dim)
            q = q.contiguous().view(self.global_tokens, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, self.global_tokens, seq_len]
            attn_weights = attn_weights.view(bsz, self.num_heads, self.global_tokens, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, self.global_tokens, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            #attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, self.global_tokens, self.head_dim]
            selected_attn_4d = selected_attn.view(bsz, self.num_heads, self.global_tokens, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(query)

        # context_layer = attn # seqlen x bsz x embed_dim

        if need_weights:
            if self.global_attention:
                attn_weights = attn_weights.view(bsz, self.num_heads, self.global_tokens, seq_len)
            else:
                attn_weights = attn_weights.permute(0, 2, 1, 3) #bsz x head x seqlen x head_dim
            return attn, attn_weights.sum(dim=1) / self.num_heads
        else:
            return attn, None

class AdaptiveMask(nn.Module):
    def __init__(self, max_size, ramp_size=16, shape=(16, ), init_val=1., _type='head', emb_dim=1024, head_dim=64, head_num=16):
        super(AdaptiveMask, self).__init__()
        logging.info('max size: {}'.format(max_size))
        left_template = torch.linspace(1 - max_size, 0, steps=max_size)
        right_template = torch.linspace(0, 1 - max_size, steps=max_size)
        center_template = torch.zeros(1)
        mask_template = torch.cat([left_template, center_template, right_template])
        self.register_buffer('mask_template', mask_template)
        self.max_size = max_size
        self.ramp_size = ramp_size
        self.type = _type
        if _type == 'head':
            self.current_val = nn.Parameter(torch.zeros(*shape) + init_val, requires_grad=True)
        elif _type == 'word':
            self.linear1 = nn.Linear(emb_dim, head_dim)
            self.linear2 = nn.Linear(head_dim, head_num)
            logging.info('Adaptive TYPE: {}'.format(_type))
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise Exception



    def forward(self, attn_prob, words=None, extra_attn_num=0):
        # attn prob: B*L*H*(2W+1+extra_dim)
        if self.type == 'head':
            mask = self.mask_template.unsqueeze(0) + torch.sigmoid(self.current_val.unsqueeze(1)) * self.max_size
            mask = mask / self.ramp_size + 1.
            mask = mask.unsqueeze(0).unsqueeze(1)
        elif self.type == 'word':
            # print('word size:', words.size())
            current_val = self.linear2(F.elu(self.linear1(words))) # B*L*H
            mask = self.mask_template.unsqueeze(0).unsqueeze(1).unsqueeze(2) + torch.sigmoid(current_val).unsqueeze(3) * self.max_size
            # print('mask size:', mask.size())
            # print('prob size:', attn_prob.size())
            mask = mask / self.ramp_size + 1.

        else:
            raise Exception
        mask = mask.type_as(attn_prob)
        mask = mask.clamp(0, 1)
        bsz, seq_len, head_num, _ = mask.size()


        if extra_attn_num > 0:
            mask = torch.cat([torch.ones(bsz, seq_len, head_num, extra_attn_num, dtype=mask.dtype).to(mask.device), mask], dim=-1)
        attn_prob = attn_prob * mask
        attn_prob = attn_prob / (attn_prob.sum(dim=-1, keepdim=True) + 1e-8)
        return attn_prob

    def get_span_size(self):
        if self.type == 'head':
            current_sizes = torch.sigmoid(self.current_val)
            current_sizes = current_sizes.detach() * self.max_size
            current_sizes = current_sizes.tolist()
            current_sizes = [math.ceil(current_size) for current_size in current_sizes]
            return current_sizes
        elif self.type == 'word':
            raise NotImplementedError







