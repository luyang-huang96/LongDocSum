from fairseq.data.language_pair_dataset import LanguagePairDataset
import logging
import torch
from fairseq.data import data_utils
import numpy as np

logger = logging.getLogger(__name__)

class FastPairDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        num_buckets=0,
        max_window_size=256,
        max_tokens=8192,
        max_source_positions=1024,
        max_target_positions=1024,
        section=None,
        sec_dict=None,
        subsection=None,
        subsection_dict=None,
        memory_test=False
    ):
        super().__init__(src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        left_pad_source, left_pad_target,
        shuffle, input_feeding,
        remove_eos_from_source, append_eos_to_target,
        align_dataset,
        append_bos, eos,
        num_buckets)
        self.window_size = max_window_size
        self.max_tokens = max_tokens
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.section = section
        self.sec_dict = sec_dict
        self.subsection = subsection
        # subsection part not implemented
        self.subsec_dict = subsection_dict
        self.memory_test = memory_test
        if section is not None:
            self.sec_eos = sec_dict.eos()
        if subsection is not None:
            self.subsec_eos = subsection_dict.eos()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        if self.sec_dict is not None:
            sec_item = self.section[index]
        else:
            sec_item = None
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

            if self.sec_dict is not None:
                bos = self.sec_dict.bos()
                if self.section[index][-1] != bos:
                    sec_item = torch.cat([torch.LongTensor([bos]), self.section[index]])
            else:
                src_item = None


        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
            if self.sec_dict is not None:
                eos = self.sec_dict.eos()
                if self.section[index][-1] == eos:
                    sec_item = self.section[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'section': sec_item
        }
        if sec_item is not None:
            assert len(sec_item) == len(src_item)
        #logging.debug('src length: {}'.format(len(src_item)))
        #logging.debug('tgt length: {}'.format(len(tgt_item)))
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        src_len = self.src_sizes[index]
        if src_len % (self.window_size * 2) == 0:
            chunks = src_len // (self.window_size * 2)
        else:
            chunks = src_len // (self.window_size * 2) + 1
        tgt_len = self.tgt_sizes[index]
        if tgt_len == 1024:
            decoder_chunks = tgt_len // self.window_size
        else:
            decoder_chunks = tgt_len // self.window_size + 1
        encoder_max = self.window_size * 2 * chunks * 2
        decoder_max = self.window_size * decoder_chunks * decoder_chunks // 2  if self.tgt_sizes is not None else 0
        # if decoder_max < self.max_tokens and encoder_max < self.max_tokens:
        #     max_tokens = self.max_tokens
        # else:
        #     max_tokens = max(encoder_max, decoder_max)
        if self.max_source_positions > 1024:
            max_tokens = self.max_source_positions
        else:
            max_tokens = self.max_source_positions
            #max_tokens = max(src_len, tgt_len)
        return max_tokens

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            memory_test=self.memory_test,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions
        )


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    memory_test=False,
    max_source_positions=1024,
    max_target_positions=1024
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_padmax(key, left_pad, move_eos_to_beginning=False, max_positions=1024):
        def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
            """Convert a list of 1d tensors into a padded 2d tensor."""
            max_pos_size = max(v.size(0) for v in values)
            size = max_positions
            assert max_pos_size < size+1
            res = values[0].new(len(values), size).fill_(pad_idx)

            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                if move_eos_to_beginning:
                    if eos_idx is None:
                        # if no eos_idx is specified, then use the last token in src
                        dst[0] = src[-1]
                    else:
                        dst[0] = eos_idx
                    dst[1:] = src[:-1]
                else:
                    dst.copy_(src)

            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
            return res
        return collate_tokens(
            [s[key] for s in samples],
            4, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    if memory_test:
        src_tokens = merge_padmax('source', left_pad=left_pad_source, max_positions=max_source_positions)
        src_lengths = torch.LongTensor([
            s['source'].ne(pad_idx).long().sum() for s in samples
        ])
    else:
        src_tokens = merge('source', left_pad=left_pad_source)
        src_lengths = torch.LongTensor([
            s['source'].ne(pad_idx).long().sum() for s in samples
        ])
    # sort by descending source length
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        if memory_test:
            target = merge_padmax('target', left_pad=left_pad_target, max_positions=max_target_positions)
        else:
            target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        if memory_test:
            tgt_lengths = torch.LongTensor([
                s['target'].ne(pad_idx).long().sum() for s in samples
            ]).index_select(0, sort_order)
        else:
            tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            if memory_test:
                prev_output_tokens = merge_padmax(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                    max_positions=max_target_positions
                )
            else:
                prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()

    if samples[0].get('section', None) is not None:
        sec_tokens = merge('section', left_pad=left_pad_source)
        sec_tokens = sec_tokens.index_select(0, sort_order)
    else:
        sec_tokens = None


    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'sec': sec_tokens
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # this part not implemented for section/subsection
    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch