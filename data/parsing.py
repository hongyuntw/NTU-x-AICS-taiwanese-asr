import os
import glob
import json
import re


mic = 'condenser'

paths = [
    f'/work/u9296553/aics/data/TAT-Vol1-test-master/{mic}/',
    # f'/work/u9296553/aics/data/TAT-Vol2-test-master/{mic}/',
]


replace_names = [
    '/condenser/wav',
    '/lavalier/wav',
    '/XYH-6-X/wav',
    '/XYH-6-Y/wav',
    '/ios/wav',
    '/android/wav'
]

# paths = [
    # '/work/u9296553/aics/data/TAT-Vol1-train-master/condenser/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/condenser/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/lavalier/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/lavalier/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/XYH-6-X/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/XYH-6-X/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/XYH-6-Y/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/XYH-6-Y/',
# ]
mode = 'test'
# save_csv_name = f'./vol1_vol2_{mic}_{mode}_tai_lo_number.csv'
save_csv_name = f'./vol1_{mic}_{mode}_tai_lo_number.csv'


# text_raw_name = './tailo_taiwen.txt'
# text_raw_fp = open(text_raw_name, 'w')

chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"
fp = open(save_csv_name, 'w')
print('path,text', file=fp)
count = 0
for rootdir in paths:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            full_path = os.path.join(subdir, file)
            speaker = full_path.split('/')[-2]
            wav_id = file.replace('.wav', '')
            json_id = '-'.join(wav_id.split('-')[:-1])

            print(full_path)
            json_path = subdir
            for replace_name in replace_names:
                json_path = json_path.replace(replace_name, '/json')

            if mode == 'train':
                json_path +=  f'/{json_id}.json'
            elif mode == 'test':
                json_path = json_path.replace('-test-master', '-test-key-master')
                json_path +=  f'/{json_id}.json'


            with open(json_path, 'r') as f:
                json_data = json.load(f)
                
                tai_lo = json_data['台羅'].replace(',', '')
                tai_lo = tai_lo.replace('--',' ').replace('-', ' ')
                tai_lo = re.sub(chars_to_ignore_regex, '', tai_lo).lower().replace("’", "'")

                tai_lo_number = json_data['台羅數字調'].replace(',', ' ')
                tai_lo_number = tai_lo_number.replace('--',' ').replace('-', ' ').replace("”", " ")
                tai_lo_number = re.sub(chars_to_ignore_regex, ' ', tai_lo_number).lower()
                tai_lo_number_list = tai_lo_number.split()
                tai_lo_number = ' '.join([token for token in tai_lo_number_list if token != ''])
                

            print(f'{full_path},{tai_lo_number}', file=fp)
            # print(tai_lo_number, file=text_raw_fp)
            count += 1

fp.close()
# text_raw_fp.close()
print(count)


