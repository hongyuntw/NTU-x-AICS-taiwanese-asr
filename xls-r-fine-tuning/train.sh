python  train.py \
--custom_set aishell_tailo_number \
--custom_set_train /work/u9296553/aics/data/aishell_tailo_train.csv \
--custom_set_test /work/u9296553/aics/data/aishell_tailo_test.csv \
--tokenize_config /work/u9296553/aics/data/aishell/wav2vec2-large-xlsr-53-tw-gpt-aishell-tailo-number \
--xlsr_config facebook/wav2vec2-xls-r-300m \
--batch 48 --grad_accum 15 --max_input_length_in_sec 15 --eval_step 200



#--tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt \
