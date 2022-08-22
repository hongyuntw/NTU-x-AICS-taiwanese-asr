from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer
from transformers import AutoTokenizer
from transformers import MBartTokenizerFast, BertTokenizer



# fp = open('/work/u9296553/aics/seq2seq/tokenizer/tailo_taiwen.txt', 'r')
fp = open('/work/u9296553/aics/data/tailo_number_taiwen.txt', 'r')
texts = fp.readlines()
texts = [text.replace('\n', '') for text in texts]
texts = list(dict.fromkeys(texts))
fp.close()
print(len(texts))

unk_tokens = set()
# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
# tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
# tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
tokenizer = BertTokenizer.from_pretrained('/work/u9296553/aics/seq2seq/data_rnn/fnlp-bart-base-chinese-add-rnn-data')
for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    unk_indices = np.array(ids) == tokenizer.unk_token_id
    try:
        unk = np.array(list(text))[unk_indices]
    except:
        unk = []
        pass
    unk_tokens.update(unk)

print(len(tokenizer))
tokenizer.add_tokens(list(unk_tokens))
print(len(tokenizer))


fp = open('/work/u9296553/aics/seq2seq/tokenizer/tailo.txt', 'r')
texts = fp.readlines()
texts = [text.replace('\n', '') for text in texts]
texts = list(dict.fromkeys(texts))
fp.close()
print(len(texts))


unk_tokens = set()
for text in texts:
    tokens = text.split()
    unk_tokens.update(tokens)

print(len(tokenizer))
tokenizer.add_tokens(list(unk_tokens))
print(len(tokenizer))

tokenizer.save_pretrained("bart-base-chinese-with-data-rnn-TAT-unk-tailo-number-tokens")