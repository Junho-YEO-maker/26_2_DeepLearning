#!/bin/bash
# ============================================================
# denoising_SEM: One-click server setup
# Run this on the GPU server after cloning the repo:
#   git clone https://github.com/Junho-YEO-maker/26_2_DeepLearning.git
#   bash ~/26_2_DeepLearning/denoising_SEM/setup_server.sh
# ============================================================
set -e

echo "=== denoising_SEM Server Setup ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. Install Miniconda (if not exists)
if [ ! -d ~/miniconda3 ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo 'export CONDA_PKGS_DIRS=~/miniconda3/pkgs' >> ~/.bashrc
else
    echo "Miniconda already installed."
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
fi

# 2. Create conda environment
if conda env list | grep -q "sem"; then
    echo "Conda env 'sem' exists. Activating..."
    conda activate sem
else
    echo "Creating conda env 'sem'..."
    conda create -n sem python=3.10 -y
    conda activate sem
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy scipy scikit-image matplotlib tqdm Pillow tensorboard
fi

# 3. Set cache dirs to home (avoid filling container root)
export HF_HOME=~/hf_cache
export TORCH_HOME=~/torch_cache
echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc
echo 'export TORCH_HOME=~/torch_cache' >> ~/.bashrc

# 4. Navigate to project
cd ~/26_2_DeepLearning/denoising_SEM

# 5. Generate synthetic dataset
echo ""
echo "=== Generating synthetic SEM dataset ==="
python -m data.synthetic_sem --output data/synthetic_sem_dataset --num-images 200 --seed 2021147017

# 6. Test GPU + models
echo ""
echo "=== Testing ==="
python -c "
import torch
from models.dncnn import DnCNN
from models.gan import Generator
from models.cyclegan import CycleGANGenerator
from models.diffusion import DiffusionUNet

print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
for name, m in [('DnCNN', DnCNN()), ('GAN', Generator()), ('CycleGAN', CycleGANGenerator()), ('Diffusion', DiffusionUNet())]:
    n = sum(p.numel() for p in m.parameters())
    print(f'  {name}: {n:,} params')
print('All models OK!')
"

echo ""
echo "=== Setup complete! ==="
echo "To train: conda activate sem && cd ~/26_2_DeepLearning/denoising_SEM"
echo "  python train.py --model dncnn --noise medium --epochs 100"
echo "  python train.py --model gan --noise medium --epochs 100"
echo "  python train.py --model cyclegan --noise medium --epochs 100"
echo "  python train.py --model diffusion --noise medium --epochs 50"
echo "To evaluate: python evaluate.py --noise medium"
