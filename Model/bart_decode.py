import torch
from longbart.longbartmodel import LongBartModel
from fairseq.models.bart import BARTModel
import argparse
import os
import logging

handler = logging.Handler()
handler.setLevel(30)

def initialize_bart(model_path, data_path, **kwargs):
    bart = BARTModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=data_path,
        **kwargs
    )
    torch.cuda.set_device(0)
    bart.cuda()
    bart.eval()
    bart.half()
    return bart

def decode_bart(bart, args):
    count = 1
    bsz = args.batch_size
    data_path = args.data_dir
    save_path = args.path
    os.makedirs(save_path)
    for split in ['test']:
        with open(os.path.join(data_path, split + '.source')) as source, open(os.path.join(save_path, split + '.hypo'), 'w') as fout:
            sline = source.readline().strip()
            #slines = [' ' + sline]
            slines = [sline]
            for sline in source:
                if count % bsz == 0:
                    with torch.no_grad():
                        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2, max_len_b=800, min_len=100, no_repeat_ngram_size=3)

                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                slines.append(sline.strip())
                count += 1
            if slines != []:
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2, max_len_b=800, min_len=100, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()


def main(args):
    #kwargs = {'left_pad_source': False}
    kwargs = {}
    bart = initialize_bart(args.model_dir, args.data_dir, **kwargs)
    decode_bart(bart, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('bart decoding')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', required=True, help='path to model')
    parser.add_argument('--data_dir', required=True, help='path todata')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=32, help='gpu id')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    main(args)
