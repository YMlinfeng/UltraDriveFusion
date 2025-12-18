# 设置 NVIDIA 软件源
apt update && apt install -y wget gnupg software-properties-common

# 下载并安装 CUDA 11.8 本地 repo
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb

dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb

cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install -y cuda-toolkit-11-8
