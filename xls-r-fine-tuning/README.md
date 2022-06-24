
### Training wav -> 台羅 / 台文 / 台羅數字調

#### train.py / train.sh
```
python  train.py \
--custom_set aishell_tailo_number \   
--custom_set_train /work/u9296553/aics/data/aishell_tailo_train.csv \ 
--custom_set_test /work/u9296553/aics/data/aishell_tailo_test.csv \ 
--tokenize_config /work/u9296553/aics/data/aishell/wav2vec2-large-xlsr-53-tw-gpt-aishell-tailo-number \ 
--xlsr_config facebook/wav2vec2-xls-r-300m \ 
--batch 48 --grad_accum 15 --max_input_length_in_sec 15 --eval_step 200
````
* custom_set
    * 儲存實驗名稱
* custom_set_train / custom_set_test
    * train/test csv path (wav_path, script)
* tokenize_config
    * tokenizer path 可以替換自己的tokenizer 
    * 想要custom tokenizer (新增 vocab) 可以參考 ```tokenizer/train_tokenizer.py```
* xlsr_config
    * wav2vec2 model name
    
train_multi.sh 是 for 多 GPU 訓練，參數用法一樣


### Evaluation
大部分的evaluation都可以在```eval.ipynb```完成
包括
* wav -> 台羅/台文/台羅數字調 CER/WER evaluation
* 2 stage evaluation (wav -> 台羅/台羅數字調 -> 台文) evaluation
