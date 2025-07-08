torchrun --nproc-per-node=2 --master-port 12346 \
    scripts/train_magicdrive.py \
    configs/magicdrive/train/stage2_1-33x224x400-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr8e-5.py \
    --cfg-options num_workers=2 prefetch_factor=2