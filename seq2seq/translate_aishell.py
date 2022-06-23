from transformers import MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BertTokenizer, BartForConditionalGeneration

import torch
from torch.utils.data import random_split
import json
from datasets import load_metric
import numpy as np
# metric = load_metric("bleu")
import random
from opencc import OpenCC
cc = OpenCC('s2t')
from collections import defaultdict

model_name = '/work/u9296553/aics/seq2seq/checkpoints/TGB_tailo_to_zh/checkpoint-12033'
tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)

device = "cuda"
# model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)


script_txt = '/work/u9296553/aics/data/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
id2script = defaultdict(str)


fp = open(script_txt, "r")
for line in fp.readlines():
    line = line.strip().split()
    wav_id = line[0]
    script = ''.join(line[1:])
    script = cc.convert(script)
    # print(wav_id)
    # print(script)
    id2script[wav_id] = script
    # input()  
fp.close()

wav_ids = list(id2script.keys())
scripts = list(id2script.values())
    
print(wav_ids[:5])
print(scripts[:5])

batch_size = 64
max_length = 48
script_tailo = []
print(len(wav_ids))
for i in range(0, len(wav_ids), batch_size):

    input_texts = [scripts[k]for k in range(i, i + batch_size) if k < len(wav_ids)]

    

    encoded_input = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if 'token_type_ids' in encoded_input : del encoded_input['token_type_ids']
    encoded_input = encoded_input.to(device)
    generated_tokens = model.generate(**encoded_input, early_stopping=True, max_length=max_length)

    # generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], early_stopping=True, max_length=48)
    # generated_tokens = model.generate(**encoded_input, forced_bos_token_id=21210, early_stopping=True, max_length=48)

    generated_tokens = [i[i != tokenizer.cls_token_id ] for i in generated_tokens]
    generated_tokens = [i[i != tokenizer.sep_token_id ] for i in generated_tokens]
    generated_tokens = [i[i != tokenizer.pad_token_id ] for i in generated_tokens]
    generated_tokens = [i[i != tokenizer.unk_token_id ] for i in generated_tokens]

    output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    script_tailo.extend(output_texts)

    print(f'\r {i}', end='')


fp = open('./aishell_tailo.txt', "w")

for wav_id , tai_lo in zip(wav_ids, script_tailo):
    print(f'{wav_id} {tai_lo}', file=fp)

fp.close()