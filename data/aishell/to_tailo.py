import os
import glob
from collections import defaultdict
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('voidful/wav2vec2-large-xlsr-53-tw-gpt')

unk_tokens = set()

modes = ['train', 'dev', 'test']

script_txt = '/work/u9296553/aics/data/aishell/aishell_tailo.txt'
id2script = defaultdict(str)


fp = open(script_txt, "r")
for line in fp.readlines():
    line = line.strip().split()
    wav_id = line[0]
    script = ' '.join(line[1:])
    id2script[wav_id] = script


print(len(tokenizer))

for mode in modes:
    fp = open(f'/work/u9296553/aics/data/aishell_tailo_{mode}.csv', 'w')
    print('path,text', file=fp)
    base_path = '/work/u9296553/aics/data/aishell/'
    path = base_path  + mode
    # print(path)
    # print(f'{path}/*.wav')
    for file in glob.glob(f'{path}/**/*.wav', recursive=True):
        wav_id = (file.split('/')[-1]).replace('.wav','')
        script = id2script[wav_id]
        if len(script.strip()) == 0:
            print(wav_id)
            continue
        tokens = script.split()
        unk_tokens.update(tokens)
        print(f'{file},{script}', file=fp)
    fp.close()

tokenizer.add_tokens(list(unk_tokens))
print(len(tokenizer))
tokenizer.save_pretrained("wav2vec2-large-xlsr-53-tw-gpt-aishell-tailo-number")
exit(1)