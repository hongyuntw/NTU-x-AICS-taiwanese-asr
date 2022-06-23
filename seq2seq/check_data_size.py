import json

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