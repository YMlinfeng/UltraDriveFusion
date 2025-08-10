#!/bin/bash
# 自动化下载脚本

# 设置目标目录
TARGET_DIR="/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz"
mkdir -p "$TARGET_DIR"

# Hugging Face 仓库基础路径
BASE_URL="https://hf-mirror.com/datasets/flymin/MagicDriveDiT-nuScenes-metadata/resolve/main"

# 要下载的文件列表
FILES=(
  "nuscenes_interp_12Hz_infos_train_with_bid.pkl"
  "nuscenes_interp_12Hz_infos_val_with_bid.pkl"
)

# 使用 wget 下载每个文件
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    wget -c "${BASE_URL}/${FILE}" -O "${TARGET_DIR}/${FILE}"
done

echo "✅ All files downloaded successfully to $TARGET_DIR"
