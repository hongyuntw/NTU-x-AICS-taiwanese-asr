import json
import re
from transformers import MBartTokenizerFast, BertTokenizer


tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
vocab = tokenizer.vocab
print(type(vocab))
print(len(tokenizer))


all_tokens = set()
chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"


# mode = 'train'


for mode in ['train', 'eval', 'test']:
    fp = open(f'./{mode}.txt', 'r')
    texts = fp.readlines()
    texts = [text.replace('\n', '') for text in texts]
    fp.close()
    print(len(texts))

    data = []

    for text in texts:
        chinese, tailo = text.split('\t')
        
        chinese = re.sub(chars_to_ignore_regex, ' ', chinese).lower().strip()
        tailo = re.sub(chars_to_ignore_regex, ' ', tailo).lower().strip()

        chinese_list = chinese.split()
        chinese = ''.join([token for token in chinese_list if token != ''])

        tailo_list = tailo.split()
        tailo = ' '.join([token for token in tailo_list if token != ''])

        all_tokens.update(chinese_list)
        all_tokens.update(tailo_list)


        data.append({
            'tai_lo' : tailo.strip(),
            'tai_wen' : chinese.strip()
        })


    with open(f'./{mode}_data.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)

print('-----update tokenizer-----')
all_tokens = list(all_tokens)
# print(len(all_tokens))
# unk_tokens = set()
# for token in all_tokens:
#     if token not in vocab



# print(all_tokens[:10])
# input()



print(len(tokenizer))
print(len(all_tokens))
tokenizer.add_tokens(all_tokens)
print(len(tokenizer))

tokenizer.save_pretrained("fnlp-bart-base-chinese-add-rnn-data")