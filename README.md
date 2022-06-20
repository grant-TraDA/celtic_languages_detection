# BERT on Celtic dataset

## Usage: Train BERT on Celtic
Here is an example of using this package.
1. Install requirements
```
pip install -r /path/to/requirements.txt
```
2. Train RoBERTa-large model
```
python train_bert.py --model-name roberta-large --dataset-path ./data/1preproc.tsv
```

3. The plots of the training:
   1. roberta-large, uncased (data/1preproc.tsv) dataset:
   
      https://wandb.ai/wsosnowski/huggingface/reports/Celtic-Bert-base--VmlldzoyMDg1OTkz?accessToken=9se65uycy7vrssfnxn9ag2uxvf0wvcg1d47o5xszuswa2zlutxt34yvq15a57ior

   2. roberta-large, cased (data/final.tsv) dataset:

      https://wandb.ai/wsosnowski/huggingface/reports/BERT-on-cased-celtic-dataset---VmlldzoyMDg2MDcz?accessToken=6p31gbb5i1djqgolzjzodbthplsblwfrrolz19kzan2pnp2uickh30uwh81u8zuf
    



