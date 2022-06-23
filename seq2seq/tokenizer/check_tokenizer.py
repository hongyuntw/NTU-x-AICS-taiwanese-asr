from transformers import AutoTokenizer


# path = '/work/u9296553/aics/xls-r-fine-tuning/tokenizer/wav2vec2-large-xlsr-53-tw-gpt-tailo-number'
path = 'voidful/wav2vec2-large-xlsr-53-tw-gpt'
# path = '/work/u9296553/aics/seq2seq/tokenizer/fnlp-bart-base-chinese-add-unk-tailo-number'
# path = '/work/u9296553/aics/seq2seq/tokenizer/huggingface_tbert_base'

# 臺羅 char base
# path = '/work/u9296553/aics/seq2seq/tokenizer/fnlp-bart-base-chinese_add_unk'

# 臺羅 word base
# path = '/work/u9296553/aics/seq2seq/tokenizer/fnlp-bart-base-chinese-add-unk-tailo-token-based'


# text = "gua2 e5 hoo7 tsio3 ho7 be2 si7 kiu2 tshit4 pat4 ji7 pat4 ji7 khong3 kiu2 khong3"
# text = '平埔族正名的路走來辛苦'
# text  = '佇廣場的儀式中有一段按呢的齣頭'

texts = [
    # 'kàu tuā hàn tio̍h tsiông kiánn līng phuè ha̍p tsi tshî guán ê kuat tīng',
    "gua2 e5 hoo7 tsio3 ho7 be2 si7 kiu2 tshit4 pat4 ji7 pat4 ji7 khong3 kiu2 khong3",
    '平埔族正名的路走來辛苦',
    '佇廣場的儀式中有一段按呢的齣頭',
]


tokenizer = AutoTokenizer.from_pretrained(path)

for text in texts:
    print(text)
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer(text).input_ids
    print(ids)
    decode = tokenizer.decode(ids, skip_special_tokens=False, spaces_between_special_tokens=True)
    print(decode)
    input()