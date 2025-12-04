# 进入你的源码目录
cd /pathtoColossalAI

# 卸载之前可能错误安装的版本
pip uninstall colossalai -y

# 设置 CUDA 环境（根据你的实际情况）
export CUDA_HOME=/mnt/bn/occupancy3d/workspace/mzj/tools/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 构建并开发者模式安装
BUILD_EXT=1 pip install -v -e .

# 验证是否使用的是源码版本
python -c "import colossalai; print(colossalai.__file__)"
