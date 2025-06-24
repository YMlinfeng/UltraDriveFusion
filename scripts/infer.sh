PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node 1 \
    scripts/inference_magicdrive.py \
    configs/magicdrive/inference/fullx848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options model.from_pretrained=./ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt \
    num_frames=9 \
    scheduler.type=rflow-slice \


# cpu_offload=true \

# ln -s /mnt/bn/occupancy3d/workspace/mzj/data/ data
# 20:42开始推理，预计得5min