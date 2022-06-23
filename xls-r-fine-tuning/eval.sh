python  eval.py \
--custom_set_test /work/u9296553/aics/data/vol1_vol2_lavalier_test.csv \
--model_name ./facebook/wav2vec2-xls-r-300m-vol1_vol2_clean_cleanest_data/checkpoint-2000 \
--tokenize_config voidful/wav2vec2-large-xlsr-53-tw-gpt \
--xlsr_config facebook/wav2vec2-xls-r-300m \
--batch 32 --grad_accum 15 --max_input_length_in_sec 15 --eval_step 10000