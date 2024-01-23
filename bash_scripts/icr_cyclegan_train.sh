# Sample training script to convert between VCC2SF2 and VCC2SF1
# Continues training from epoch 500

python -W ignore::UserWarning -m icr_cyclegan_vc.train \
    --name icr_cyclegan_vc_VCC2SF2_VCC2SF1 \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training \
    --speaker_A_id VCC2SF2 \
    --speaker_B_id VCC2SF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 100000 \
    --batch_size 1 \
    --generator_lr 2e-4 \
    --discriminator_lr 1e-4 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
 


