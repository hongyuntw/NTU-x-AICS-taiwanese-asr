import os
import glob
import json
import re


mic = 'condenser'


replace_names = [
    '/condenser/wav',
    '/lavalier/wav',
    '/XYH-6-X/wav',
    '/XYH-6-Y/wav',
    '/ios/wav',
    '/android/wav'
]

mode = 'test'

paths = [

    f'/work/u9296553/aics/data/TAT-Vol1-{mode}-master/condenser/',
    f'/work/u9296553/aics/data/TAT-Vol2-{mode}-master/condenser/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/condenser/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/condenser/',

    # '/work/u9296553/aics/data/TAT-Vol1-eval-master/condenser/',
    # '/work/u9296553/aics/data/TAT-Vol2-eval-master/condenser/',

    # '/work/u9296553/aics/data/TAT-Vol1-test-master/condenser/',
    # '/work/u9296553/aics/data/TAT-Vol2-test-master/condenser/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/lavalier/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/lavalier/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/XYH-6-X/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/XYH-6-X/',

    # '/work/u9296553/aics/data/TAT-Vol1-train-master/XYH-6-Y/',
    # '/work/u9296553/aics/data/TAT-Vol2-train-master/XYH-6-Y/',
]


# tailo_raw_name = './tailo.txt'
# tailo_raw_fp = open(tailo_raw_name, 'w')

# taiwen_raw_name = './taiwen.txt'
# taiwen_raw_fp = open(taiwen_raw_name, 'w')

# text_raw_name = '/work/u9296553/aics/seq2seq/tokenizer/tailo_number_taiwen.txt'
# text_raw_name = '/work/u9296553/aics/data/tailo_number_taiwen.txt'
# text_raw_fp = open(text_raw_name, 'w')

# tailo_number_name = '/work/u9296553/aics/data/tailo_number.txt'
# tailo_number_raw_fp = open(tailo_number_name, 'w')

data = []

chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"

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

            if 'test' in json_path:
                json_path = json_path.replace('-test-master', '-test-key-master')
            elif 'eval' in json_path:
                json_path = json_path.replace('-eval-master', '-eval-key-master')

            json_path +=  f'/{json_id}.json'
            


            with open(json_path, 'r') as f:
                json_data = json.load(f)
                
                # tai_lo = json_data['台羅'].replace(',', '')
                # tai_lo = tai_lo.replace('--',' ').replace('-', ' ')
                # tai_lo = re.sub(chars_to_ignore_regex, '', tai_lo).lower().replace("’", "'")

                tai_lo_number = json_data['台羅數字調'].replace(',', ' ')
                tai_lo_number = tai_lo_number.replace('--',' ').replace('-', ' ').replace("”", " ")
                tai_lo_number = re.sub(chars_to_ignore_regex, ' ', tai_lo_number).lower()
                tai_lo_number_list = tai_lo_number.split()
                tai_lo_number = ' '.join([token for token in tai_lo_number_list if token != ''])
                tai_lo = tai_lo_number

                tai_wen = json_data['漢羅台文'].replace(',', '')
                tai_wen = re.sub(chars_to_ignore_regex, '', tai_wen).lower().replace("’", "'")

            # print(tai_lo, file=text_raw_fp)
            # print(tai_wen, file=text_raw_fp)
            # print(tai_lo, file = tailo_number_raw_fp)

            data.append({
                'tai_lo' : tai_lo.strip(),
                'tai_wen' : tai_wen.strip()
            })
            count += 1

# tailo_raw_fp.close()
# taiwen_raw_fp.close()
# text_raw_fp.close()

with open(f'/work/u9296553/aics/seq2seq/data_tailo_number/{mode}_data.json', 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii=False)

print(count)


