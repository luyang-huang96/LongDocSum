
# taken from fairseq
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart import BARTHubInterface

logger = logging.getLogger(__name__)


class LongBARTHubInterface(BARTHubInterface):
    def __init__(self, args, task, model):
        super().__init__(args, task, model)
        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))
        args.limit_length = getattr(args, 'limit_length', False)
        self.max_source_position = min((self.task.max_positions()[0], self.model.max_positions()[0]))
        if args.limit_length:
            self.max_source_position = args.length
        print('max position:', self.max_source_position)

    def encode(self, sentence: str, *addl_sentences, no_separator=True, is_bpe=False) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        if not is_bpe:
            tokens = self.bpe.encode(sentence)
        else:
            tokens = sentence
        if len(tokens.split(' ')) > self.max_source_position - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_source_position - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False, sections=None, is_bpe=False, **kwargs) -> str:
        # if sections is not None:
        #     is_bpe = True
        # else:
        #     is_bpe = False
        input = [self.encode(sentence, is_bpe=is_bpe) for sentence in sentences]
        if sections is not None and self.model.encoder.section:
            sections = [self.encode_sections(section) for section in sections]
        else:
            sections = None
        hypos = self.generate(input, sections, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def encode_sections(self, section):
        if len(section.split(' ')) > self.max_source_position - 2:
            section = ' '.join(section.split(' ')[:self.max_source_position - 2])
        section = '<s> ' + section + ' </s>'
        section = self.task.sec_dict.encode_line(section, append_eos=False)
        section = section.long()
        return section

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        sentences = [s.replace('\n', ' ') for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences


    def _build_sample(self, src_tokens: List[torch.LongTensor], sections, data_path=None, raw_graph=None):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            sections,
            raw_graph=raw_graph,
            data_path=data_path
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def generate(self, tokens: List[torch.LongTensor], sections,  beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        raw_graph = kwargs.get('raw_graph', None)
        data_path = kwargs.get('data_path', None)
        if 'raw_graph' in kwargs:
            del kwargs['raw_graph']
        if 'data_path' in kwargs:
            del kwargs['data_path']
        #print(kwargs.keys())
        sample = self._build_sample(tokens, sections, raw_graph=raw_graph, data_path=data_path)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

