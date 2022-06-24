
### train_bart.py 
* 訓練seq2seq translation model 
    * 任務有 台羅/台羅拼音 -> 中文 or 中文台羅/台羅拼音，根據任務在  ```data_collator```的function內選擇對應的input & label
    * 根據任務選擇對應的 model / tokenizer
        * custom tokenizer 的話可以參考 ```tokenizer/train_tokenizer.py``` 加入vocab
        * 如果tokenizer size跟原始model不符合要加入 
        ```
        model.resize_token_embeddings(len(tokenizer))
        # defining trainer using 🤗
        ````

    * 訓練框架是參考huggingface的trainer framework來訓練
    
    
### eval_translation.ipynb
evaluate translation任務 performance (CER/WER)

### translate_aishell.py
負責將aishell中文 scirpt 利用 seq2seq model 轉成台羅數字調的 code
