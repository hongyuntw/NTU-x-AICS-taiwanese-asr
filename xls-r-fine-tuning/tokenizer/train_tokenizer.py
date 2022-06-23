from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer
# from Wav2Vec2CTCTokenizer import Wav2Vec2CTCTokenizer



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
# for unk_token in unk_tokens:
#     tokenizer.add_tokens([unk_token])
tokenizer.add_tokens(unk_tokens)
print(len(tokenizer))

tokenizer.save_pretrained("wav2vec2-large-xlsr-53-tw-gpt-tailo-number")
exit(1)








# import json
# with open('./tai_lo_tokenizer.json') as fp:
#     data = json.load(fp)

# with open('./tai_lo/vocab.json', 'w') as fp:
#     json.dump(data['model']['vocab'], fp)

# fp = open('./tai_lo/tokenizer.json', 'r')
# texts = fp.readlines()
# texts = [text.replace('\n', '') for text in texts]
# fp.close()



text  = 'guá ê hōo tsiò hō bé sī kiú tshit pat jī pat jī khòng kiú khòng'
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('voidful/wav2vec2-large-xlsr-53-tw-gpt')
tokenizer = Wav2Vec2CTCTokenizer('./tai_lo/vocab.json')
tokenizer.pre_tokenizer = Whitespace

print(type(tokenizer))
ids = tokenizer(text).input_ids
print(ids)
print(type(ids))
decode = tokenizer.decode(ids, spaces_between_special_tokens=True)
print(ids)
print(decode)
exit(1)

tokenizer.save_pretrained("./org/")

print(tokenizer.is_fast)
exit(1)
# new_tokenizer = tokenizer.train_new_from_iterator(texts)
# new_tokenizer.save_pretrained("tai_lo_tokenizer")

# tokenizer = Tokenizer(BPE(unk_token="[UNK]", pad_token="[PAD]"))
# trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])
# tokenizer.pre_tokenizer = Whitespace()
# tokenizer.train_from_iterator(texts, trainer)
# # tokenizer._save_pretrained("./tai_lo_tokenizer/")


# tokenizer.save_vocabulary()


# from tokenizers import BertWordPieceTokenizer

# # Initialize an empty BERT tokenizer
# tokenizer = BertWordPieceTokenizer(
#   lowercase=True,
# )

# # prepare text files to train vocab on them
# files = ['text_raw.txt']

# # train BERT tokenizer
# tokenizer.train(
#   files,
#   show_progress=True,
#   special_tokens=['[PAD]', '[UNK]'],
#   wordpieces_prefix="##"
# )

# # save the vocab
# # tokenizer.save_model("tai_lo")
# tokenizer.save_pretrained("tai_lo")

