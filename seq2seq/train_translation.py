from transformers import MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from torch.utils.data import random_split
import json
from datasets import load_metric
import numpy as np
import editdistance as ed
from datasets import load_metric
from transformers import BertTokenizer, BartForConditionalGeneration
# cer_metric = load_metric("cer")
# read text 


# fp = open('/work/u9296553/aics/seq2seq/translation_data.json')
# data = json.load(fp)
# print(len(data))
# split = 0.85
# train_size = int(split*len(data))
# eval_size = int(int((1-split)*len(data))+1)

# train_dataset = data[:train_size]
# eval_dataset = data[train_size:]

# with open('train_data.json', 'w', encoding='utf8') as f:
#     json.dump(data[:train_size], f, ensure_ascii=False)
# with open('eval_data.json', 'w', encoding='utf8') as f:
#     json.dump(data[train_size:], f, ensure_ascii=False)

# train_dataset, eval_dataset = random_split(data, lengths=[int(split*len(data)), int((1-split)*len(data))+1])
# print(type(train_dataset))

data_pre_fix = '/work/u9296553/aics/seq2seq/data'
train_path = f'{data_pre_fix}/train_data.json'
eval_path = f'{data_pre_fix}/eval_data.json'
test_path = f'{data_pre_fix}/test_data.json'

with open(train_path) as f:
    train_dataset = json.load(f)
with open(eval_path) as f:
    eval_dataset = json.load(f)
with open(test_path) as f:
    test_dataset = json.load(f)



print(len(train_dataset))
print(len(eval_dataset))
print(len(test_dataset))

# exit(1)
src_lang="en_XX"
tgt_lang="zh_CN"
max_input_length=48

# defining collator functioon for preparing batches on the fly ..
def data_collator(features:list):

    labels = [f["tai_wen"] for f in features]
    inputs = [f["tai_lo"] for f in features]

    # batch = tokenizer.prepare_seq2seq_batch(src_texts=inputs, src_lang="tw_TL", tgt_lang="tw_TW", tgt_texts=labels, max_length=32, max_target_length=32)
    batch = tokenizer.prepare_seq2seq_batch(src_texts=inputs, src_lang=src_lang, tgt_lang=tgt_lang, tgt_texts=labels, max_length=32, max_target_length=32)

    for k in batch:
        batch[k] = torch.tensor(batch[k])

    return batch


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def cer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot

def compute_metrics(pred):
    pred_ids = pred.predictions.cpu()
    pred_ids = [i[i != -100] for i in pred_ids]
    # for cal wer need to adding spaces_between_special_tokens=True, if cer spaces_between_special_tokens=False
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)

    label_ids = pred.label_ids.cpu()
    label_ids = [i[i != -100] for i in label_ids]

    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

    cer = cer_cal(label_str, pred_str)
    return {"cer": cer}


def preprocess_function(examples):
    inputs = examples["tai_lo"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tai_wen"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# init model
# tokenizer = AutoTokenizer.from_pretrained("/work/u9296553/aics/seq2seq/tokenizer/tailo_taiwen_tokenizer", src_lang="tw_TL", tgt_lang="tw_TW")
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", 
#                                                     vocab_size=len(tokenizer),
#                                                     ignore_mismatched_sizes=True)

# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=src_lang, tgt_lang=tgt_lang)
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

# tokenizer = AutoTokenizer.from_pretrained("/work/u9296553/aics/seq2seq/tokenizer/mbart-large-cc25_add_unk", src_lang=src_lang, tgt_lang=tgt_lang)
# # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", 
# #                                                     vocab_size=len(tokenizer),
# #                                                     ignore_mismatched_sizes=True)
# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")                                                 
# model.resize_token_embeddings(len(tokenizer))

tokenizer = BertTokenizer.from_pretrained("/work/u9296553/aics/seq2seq/tokenizer/fnlp-bart-base-chinese_add_unk")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
model.resize_token_embeddings(len(tokenizer))
# defining trainer using 🤗


args = Seq2SeqTrainingArguments(output_dir="./tailo_to_taiwen_bart_base_chinese_with_unk/",
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy="epoch",
                        # eval_steps=1000,
                        per_device_train_batch_size=64,
                        per_device_eval_batch_size=32,
                        learning_rate=5e-5,
                        num_train_epochs=25,
                        logging_dir="./logs",
                        save_strategy='epoch',
                        save_total_limit=10,
                        # eval_accumulation_steps=1,
                        prediction_loss_only=True,
                        fp16=True,
                        warmup_steps=500,
                        predict_with_generate=True,)


trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset+test_dataset,
                # compute_metrics=compute_metrics,
                tokenizer=tokenizer,)

trainer.train()
