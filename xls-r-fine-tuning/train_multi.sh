python -m torch.distributed.launch --nproc_per_node=2 train.py \
--custom_set fintune_aishell_condenser_tailo_number \
--custom_set_train /work/u9296553/aics/data/vol1_vol2_condenser_train_tai_lo_number_for_aishell.csv \
--custom_set_test /work/u9296553/aics/data/vol1_vol2_condenser_test_tai_lo_number_for_aishell.csv \
--tokenize_config /work/u9296553/aics/data/aishell/wav2vec2-large-xlsr-53-tw-gpt-aishell-tailo-number \
--xlsr_config facebook/wav2vec2-xls-r-300m \
 --batch 32 --grad_accum 15 --max_input_length_in_sec 15 --eval_step 80



# --custom_set vol1_vol2_condenser_tai_lo_number \
# --custom_set_train /work/u9296553/aics/data/vol1_vol2_condenser_train_tai_lo_number.csv \
# --custom_set_test /work/u9296553/aics/data/vol1_vol2_condenser_test_tai_lo_number.csv \
# --tokenize_config /work/u9296553/aics/xls-r-fine-tuning/tokenizer/wav2vec2-large-xlsr-53-tw-gpt-tailo-number \