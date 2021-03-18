""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists

from evaluate import eval_meteor, eval_rouge
import stanza
import glob

# try:
#     _DATA_DIR = os.environ['DATA']
# except KeyError:
#     print('please use environment variable to specify data directories')

def process_doc(doc):
    processed = []
    for sent in doc.sentences:
        _sent = []
        for word in sent.tokens:
            _sent.append(word.text)
        processed.append(' '.join(_sent))
    return processed

def main(args):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt', use_gpu=args.use_gpu)

    if not os.path.exists(join(args.ref_dir, 'ref')):
        os.makedirs(join(args.ref_dir, 'ref'))
        with open(join(args.ref_dir, 'test.target')) as f:
            _i = 0
            for line in f:
                doc = nlp(line.strip())
                with open(join(args.ref_dir, 'ref', str(_i) + '.ref'), 'w') as g:
                    processed = process_doc(doc)
                    g.write('\n'.join(processed))
                _i += 1
    if not os.path.exists(join(args.decode_dir, 'output')):
        os.makedirs(join(args.decode_dir, 'output'))
    all_docs = []
    with open(join(args.decode_dir, 'test.hypo')) as f:
        _i = 0
        for line in f:
            all_docs.append(line.strip())
    already_tokenized = len(glob.glob(join(args.decode_dir, 'output', '*')))

    if len(all_docs) > already_tokenized:
        _i = 0
        for doc in all_docs:
            if _i < already_tokenized:
                _i += 1
                continue
            doc = nlp(doc)
            with open(join(args.decode_dir, 'output', str(_i) + '.dec'), 'w') as g:
                processed = process_doc(doc)
                g.write('\n'.join(processed))
            _i += 1



    print('Start evaluating Rouge')
    try:
        dec_dir = join(args.decode_dir, 'output')
        with open(join(args.decode_dir, 'log.json')) as f:
            split = json.loads(f.read())['split']
    except:
        split = 'test'
    ref_dir = join(args.ref_dir, 'ref')
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--ref_dir', action='store', required=True)
    parser.add_argument('--use_gpu', action='store_true', required=False)

    args = parser.parse_args()
    main(args)
