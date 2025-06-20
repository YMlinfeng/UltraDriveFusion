import os

# 要链接的文件夹列表
required_dirs = ["can_bus", "maps", "sweeps", "v1.0-trainval", "v1.0-mini"]

# 源路径和目标路径
src_root = "/mnt/bn/perception3d/open_source_data/nuscenes"
dst_root = "/mnt/bn/occupancy3d/workspace/mzj/data/nuscenes"

# 确保目标根目录存在
os.makedirs(dst_root, exist_ok=True)

for dir_name in required_dirs:
    src_path = os.path.join(src_root, dir_name)
    dst_path = os.path.join(dst_root, dir_name)

    # 如果目标链接已存在，跳过或覆盖
    if os.path.islink(dst_path) or os.path.exists(dst_path):
        print(f"Skipping existing: {dst_path}")
        continue

    try:
        os.symlink(src_path, dst_path)
        print(f"Linked {src_path} -> {dst_path}")
    except OSError as e:
        print(f"Failed to link {src_path} -> {dst_path}: {e}")


os.symlink("/mnt/bn/occupancy3d/workspace/mzj/data/", "/mnt/bn/occupancy3d/workspace/mzj/MagicDriveDiT/data")