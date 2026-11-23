set -x
cd ~/multinex_repro
if [ ! -x ~/miniforge3/bin/conda ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh && bash /tmp/mf.sh -b -p ~/miniforge3
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n multinex python=3.9 -y
conda activate multinex
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm pytorch_msssim
cd Multinex && pip install -e .
echo INSTALL_EXIT=$?
pip freeze > ../logs/03_pip_freeze.txt
