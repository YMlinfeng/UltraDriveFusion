#!/bin/bash
# /mnt/bn/occupancy3d/workspace/mzj/data
# /mnt/bn/occupancy3d/workspace/mzj/env/magic

# set -e  # 遇到错误立即退出
cd /mnt/bn/occupancy3d/workspace/mzj/MagicDriveDiT/ || exit 1
source /mnt/bn/occupancy3d/workspace/mzj/env/magic/bin/activate
sudo chown -R tiger:tiger /mnt/bn/occupancy3d/workspace/mzj/MagicDriveDiT/

# pip install colossalai==0.4.3 --no-deps  #! 不能再虚拟环境中用sudo
# # # 安装 apex（编译）
# cd /mnt/bn/occupancy3d/workspace/mzj/lib/apex/ || exit 1
# pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
#   --config-settings="--build-option=--cpp_ext" \
#   --config-settings="--build-option=--cuda_ext" ./

# 可选：修复权限和安装额外库
sudo apt update
sudo apt install libgl1-mesa-glx -y

which nvcc
nvcc --version
echo $CUDA_HOME

# python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}')"
# pip freeze > requirements.txt
# BUILD_EXT=1 pip install colossalai
# python3 -c "from colossalai.kernel.kernel_loader import CPUAdamLoader; CPUAdamLoader().load()"
source /mnt/bn/occupancy3d/workspace/mzj/env/magic/bin/activate