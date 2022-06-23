import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead 
import IPython.display as ipd
from math import log
import editdistance as ed
import csv

model_name = './facebook/wav2vec2-xls-r-300m-vol1_vol2_clean_cleanest_data/checkpoint-1400'
processor_name = "./facebook/wav2vec2-xls-r-300m-vol1_vol2_clean_cleanest_data/"
device = "cuda"

# eval
eval_csv_path = '../data/vol1_vol2_lavalier_test.csv'
print(eval_csv_path)


tokenizer = AutoTokenizer.from_pretrained("ckiplab/gpt2-base-chinese")  
lm_model = AutoModelWithLMHead.from_pretrained("ckiplab/gpt2-base-chinese").to(device)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(processor_name)


def load_file_to_data(file,sampling_rate=16_000):
    batch = {}
    speech, _ = torchaudio.load(file)
    if sampling_rate != '16_000' or sampling_rate != '16000':
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
        batch["sampling_rate"] = resampler.new_freq
    else:
        batch["speech"] = speech.squeeze(0).numpy()
        batch["sampling_rate"] = '16000'
    return batch

def predict_beam(data,beamsize=3):
    features = processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    decoded_results = []
    for logit in logits:
        sequences = [[[], 1.0]]
        pred_ids = torch.argmax(logit, dim=-1)
        mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
        vocab_size = logit.size()[-1]
        voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1,vocab_size)),dim=-1)
        while True:
            all_candidates = list()
            exceed = False
            for seq in sequences:
                tokens, score = seq
                gpt_input = torch.tensor([tokenizer.cls_token_id]+tokens).to(device)
                gpt_prob = torch.nn.functional.softmax(lm_model(gpt_input).logits, dim=-1)[:len(gpt_input),:]
                if len(gpt_input) >= len(voice_prob):
                    exceed = True
                comb_pred_ids = gpt_prob*voice_prob[:len(gpt_input)]
                v,i = torch.topk(comb_pred_ids,50,dim=-1)
                for tok_id,tok_prob in zip(i.tolist()[-1],v.tolist()[-1]):
                    candidate = [tokens + [tok_id], score + -log(tok_prob)]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beamsize]
            if exceed:
                break

        for i in sequences:
            decoded_results.append(processor.decode(i[0]))

    return decoded_results

def predict(data, GPT_FIX=False):
    features = processor(data["speech"], sampling_rate=data["sampling_rate"], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    
    decoded_results = []
    for logit in logits:
        pred_ids = torch.argmax(logit, dim=-1)
        mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
        vocab_size = logit.size()[-1]
        voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1,vocab_size)),dim=-1)
        gpt_input = torch.cat((torch.tensor([tokenizer.cls_token_id]).to(device),pred_ids[pred_ids>0]), 0)
        gpt_prob = torch.nn.functional.softmax(lm_model(gpt_input).logits, dim=-1)[:voice_prob.size()[0],:]
        if GPT_FIX: comb_pred_ids = torch.argmax(gpt_prob*voice_prob, dim=-1)
        else: comb_pred_ids = torch.argmax(voice_prob, dim=-1)
        decoded_results.append(processor.decode(comb_pred_ids))

    return decoded_results



def cer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot



def eval_on_csv(csv_path):
    file = open(csv_path)
    csvreader = csv.reader(file)
    groundtruth = []
    hypothesis = []
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        wav_path = row[0]
        label = row[1]
        groundtruth.append(str(label))
        
        vdata = load_file_to_data(wav_path)
        pred = predict(vdata, GPT_FIX=True)
        pred = ''.join(pred)
        pred = pred.replace('[UNK]', '@')
        hypothesis.append(pred)

        # cer = cer_cal(groundtruth, hypothesis)
        # print(hypothesis)
        # print(groundtruth)
        # print(cer)

        print(f'\r {i}', end='')

    cer = cer_cal(groundtruth, hypothesis)
    print(eval_csv_path)
    print(cer)

eval_on_csv(eval_csv_path)





