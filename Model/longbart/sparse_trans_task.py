from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
import itertools
import logging
import os

import numpy as np

from fairseq import utils, options
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)
from .fast_pair_dataset import FastPairDataset
from .dataset.FastPairDatasetwizGraph import FastPariDatasetwizGraph


EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@register_task('translation_longbart')
class TranslationSparseTransformerTask(TranslationTask):

    def __init__(self, args, src_dict, tgt_dict, sec_dict=None):
        super().__init__(args, src_dict, tgt_dict)
        #self.src_dict = src_dict
        #self.tgt_dict = tgt_dict
        self.sec_dict = sec_dict

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=2048, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=True,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--sec-emb', action='store_true', help='enable section embedding')
        parser.add_argument('--memory_test', action='store_true', help='test memory boundary')
        parser.add_argument('--graph_encoding', action='store_true', help='include graph')
        parser.add_argument('--graph_encode_type', action='store', default='edge', type=str)
        # fmt: on

    def max_positions(self):
        return (self.args.max_source_positions, self.args.max_target_positions)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        try:
            args.sec_emb
        except:
            args.sec_emb = False
        #print('sec emb:', args.sec_emb)
        args.graph_encoding = getattr(args, 'graph_encoding', False)
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        args.memory_test = getattr(args, 'memory_test', False)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        if args.sec_emb:
            sec_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format('sec')))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        if args.sec_emb:
            assert sec_dict.pad() == src_dict.pad() and src_dict.eos() == sec_dict.eos() and sec_dict.unk() == src_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        if args.sec_emb:
            logger.info('[{}] dictionary: {} types'.format('section', len(sec_dict)))
        else:
            sec_dict = None

        return cls(args, src_dict, tgt_dict, sec_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if self.sec_dict is not None:
            section = 'sec'
        else:
            section = None

        if not self.args.left_pad_source:
            logging.info('left pad source disabled')


        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            window_size=self.args.window,
            max_tokens=self.args.max_tokens,
            section=section,
            sec_dict=self.sec_dict,
            memory_test=self.args.memory_test,
            load_graph=self.args.graph_encoding,

        )
    def build_dataset_for_inference(self, src_tokens, src_lengths, sections=None, data_path=None, raw_graph=None):
        assert (sections is not None and self.sec_dict is not None) or (sections is None and self.sec_dict is None)
        if self.args.graph_encoding:
            assert data_path is not None
            graph_dict_path = os.path.join(data_path, 'graph_dict.txt')
            return FastPariDatasetwizGraph(src_tokens, src_lengths, self.source_dictionary, raw_graph=raw_graph,
                                           graph_dict_path=graph_dict_path
                                       )
        else:
            return FastPairDataset(src_tokens, src_lengths, self.source_dictionary, section=sections, sec_dict=self.sec_dict)

def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    window_size=['256'],
    max_tokens=8096,
    section=None,
    sec_dict=None,
    subsection=None,
    subsec_dict=None,
    memory_test=False,
    load_graph=False
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    sec_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        if section is not None:
            assert split_exists(split_k + '-' + section, src, tgt, src, data_path)
            sec_prefix = os.path.join(data_path, '{}-{}.{}-{}.'.format(split_k, section, src, tgt))
            sec_dataset = data_utils.load_indexed_dataset(sec_prefix + src, sec_dict, dataset_impl)
            if truncate_source:
                sec_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(sec_dataset, sec_dict.eos()),
                    max_source_positions - 1,
                ),
                sec_dict.eos(),
            )
            sec_datasets.append(sec_dataset)


        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )


        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if truncate_source:
            tgt_dataset = TruncateDataset(
                tgt_dataset,
                max_target_positions - 1
            )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    if section is not None:
        assert len(src_datasets) == len(sec_datasets)

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        if section is not None:
            sec_dataset = sec_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        if section is not None:
            sec_dataset = ConcatDataset(sec_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        if section is not None:
            sec_dataset = PrependTokenDataset(sec_dataset, sec_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))
        if section is not None:
            sec_dataset = AppendTokenDataset(sec_dataset, sec_dict.index('[{}]'.format(section)))

    if section is None:
        sec_dataset = None

    # not implemented for section/subsection
    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    max_window_size = max([int(w) for w in window_size])
    if load_graph:
        if split == 'valid':
            split = 'val'
        graph_path = os.path.join(data_path, '{}.graph.source'.format(split))
        graph_dict_path = os.path.join(data_path, 'graph_dict.txt')
        return FastPariDatasetwizGraph(src_dataset, src_dataset.sizes, src_dict,
                                       tgt_dataset, tgt_dataset_sizes, tgt_dict,
                                       left_pad_source=left_pad_source,
                                       left_pad_target=left_pad_target,
                                       align_dataset=align_dataset, eos=eos,
                                       num_buckets=num_buckets,
                                       max_tokens=max_tokens,
                                       max_source_positions=max_source_positions,
                                       max_target_positions=max_target_positions,
                                       memory_test=memory_test,
                                       graph_path=graph_path,
                                       graph_dict_path=graph_dict_path
                                       )
    else:
        return FastPairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
            max_window_size=max_window_size,
            max_tokens=max_tokens,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            section=sec_dataset,
            sec_dict=sec_dict,
            memory_test=memory_test
        )
