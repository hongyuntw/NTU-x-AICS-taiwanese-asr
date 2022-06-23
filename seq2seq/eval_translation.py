from transformers import MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from torch.utils.data import random_split
import json
from datasets import load_metric
import numpy as np
import editdistance as ed
from datasets import load_metric

cer_metric = load_metric("cer")
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

with open('/work/u9296553/aics/seq2seq/train_data.json') as f:
    train_dataset = json.load(f)
with open('/work/u9296553/aics/seq2seq/eval_data.json') as f:
    eval_dataset = json.load(f)
with open('/work/u9296553/aics/seq2seq/test_data.json') as f:
    test_dataset = json.load(f)



print(len(train_dataset))
print(len(eval_dataset))
print(len(test_dataset))

# exit(1)


# defining collator functioon for preparing batches on the fly ..
def data_collator(features:list):

    labels = [f["tai_wen"] for f in features]
    inputs = [f["tai_lo"] for f in features]

    batch = tokenizer.prepare_seq2seq_batch(src_texts=inputs, src_lang="tw_TL", tgt_lang="tw_TW", tgt_texts=labels, max_length=32, max_target_length=32)

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




# init model
model_name = './tailo_to_taiwen_eval_test/checkpoint-2000'
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="tw_TL", tgt_lang="tw_TW")

# defining trainer using 🤗


args = Seq2SeqTrainingArguments(output_dir="./eval/",
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy="steps",
                        eval_steps=2000,
                        per_device_train_batch_size=128,
                        per_device_eval_batch_size=8,
                        learning_rate=5e-5,
                        num_train_epochs=75,
                        logging_dir="./logs",
                        save_total_limit=15,
                        # eval_accumulation_steps=1,
                        # prediction_loss_only=True,
                        fp16=True,
                        predict_with_generate=True,)


trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset+test_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,)

eval_res = trainer.evaluate(eval_dataset=eval_dataset)
print(eval_res)

test_res = trainer.evaluate(eval_dataset=test_dataset)
print(test_res)