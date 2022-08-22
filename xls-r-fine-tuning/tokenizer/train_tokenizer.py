from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from transformers import Wav2Vec2CTCTokenizer


fp = open('/work/u9296553/aics/data/tailo_number.txt', 'r')
texts = fp.readlines()
texts = [text.replace('\n', '') for text in texts]
fp.close()

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('voidful/wav2vec2-large-xlsr-53-tw-gpt')

unk_tokens = set()
for text in texts:
    tokens = text.split()
    unk_tokens.update(tokens)


print(len(tokenizer))
tokenizer.add_tokens(list(unk_tokens))
print(len(tokenizer))

tokenizer.save_pretrained("wav2vec2-large-xlsr-53-tw-gpt-tailo-number")

