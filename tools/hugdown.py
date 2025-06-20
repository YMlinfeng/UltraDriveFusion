# from huggingface_hub import snapshot_download

# # 下载模型
# snapshot_download(
#     repo_id="google/t5-v1_1-xxl",
#     local_dir="./pretrained/t5-v1_1-xxl",
#     local_dir_use_symlinks=False
# )



# from huggingface_hub import hf_hub_download
# import os

# # 设置目标目录
# output_dir = "./ckpts/MagicDriveDiT-stage3-40k-ft"
# os.makedirs(output_dir, exist_ok=True)

# # 要下载的文件列表
# files_to_download = [
#     "ema.pt",
#     "running_states.json"
# ]

# # 下载每个文件
# for filename in files_to_download:
#     hf_hub_download(
#         repo_id="flymin/MagicDriveDiT-stage3-40k-ft",
#         filename=filename,
#         local_dir=output_dir,
#         local_dir_use_symlinks=False
#     )



from huggingface_hub import hf_hub_download
import os
import shutil

# 设置本地数据路径
target_dir = "/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes_mmdet3d-12Hz"
os.makedirs(target_dir, exist_ok=True)

# 仓库信息
repo_id = "flymin/MagicDriveDiT-nuScenes-metadata"
filenames = [
    "nuscenes_interp_12Hz_infos_train_with_bid.pkl",
    "nuscenes_interp_12Hz_infos_val_with_bid.pkl"
]

# 下载文件
for filename in filenames:
    print(f"Downloading {filename}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"  # 重要：指定是 dataset 类型
    )
    # 移动到目标目录
    shutil.copy(downloaded_path, os.path.join(target_dir, filename))
    print(f"Saved to {os.path.join(target_dir, filename)}")

print("✅ All files downloaded successfully.")