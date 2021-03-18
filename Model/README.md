# README

This code repo is built on a frozen version of [fairseq](https://github.com/pytorch/fairseq). We run our experiment on 2 RTX 6000 GPU (24GB). If your GPUs have smaller memory, you may need to adjust batch size or the maximum length.

For model training, you can refer to the script in [train script](https://github.com/luyang-huang96/LongDocSum/tree/main/Model/train_script).  

```
./finetune_sinkhorn_hepos.sh
```

For model decoding, you can use the command  
```
python longbart_decode.py --path [/path/to/save/summary] --model_dir [/path/to/model] --data_dir [/path/to/data]
```

For ROUGE evaluation, you can run
```
python eval_mode.py --decode_dir [/path/to/save/summary] --ref_dir [/path/to/ref] --rouge
```

