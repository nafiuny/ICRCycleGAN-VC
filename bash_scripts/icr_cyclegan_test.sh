python -m icr_cyclegan_vc.test \
    --name icr_cyclegan_vc_VCC2SF2_VCC2SF1 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF2 \
    --speaker_B_id VCC2SF1 \
    --ckpt_dir results/icr_cyclegan_vc_VCC2SF2_VCC2SF1/ckpts \
    --load_epoch 500 \
    --model_name generator_A2B \
