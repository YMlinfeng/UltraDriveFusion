torchrun --nproc-per-node=1 \--master-port 12346 \
    scripts/train_magicdrive.py \
    configs/magicdrive/train/stage1_1x224x400_stdit3_CogVAE_noTemp_xCE_wSST_bs4_lr8e-5.py \
    --cfg-options num_workers=8 prefetch_factor=2