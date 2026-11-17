#!/usr/bin/env bash
# Anomalydiffusion conda 환경 (공식 environment.yaml과 같은 버전).
# 공식 yaml은 Anaconda defaults 채널을 쓰므로, conda-forge python + PyTorch 공식 cu113 pip 휠로 대신 만든다.
set -euo pipefail
CONDA=${CONDA:-conda}
if ! $CONDA env list | grep -q '^Anomalydiffusion '; then
  $CONDA create -y -n Anomalydiffusion -c conda-forge --override-channels python=3.8 pip
fi
PY=$($CONDA run -n Anomalydiffusion python -c 'import sys; print(sys.executable)')
$PY -m pip install --no-input torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
$PY -m pip install --no-input numpy==1.23.1 albumentations==1.3.0 imageio==2.9.0 \
  imageio-ffmpeg==0.4.2 pytorch-lightning==1.5.0 omegaconf==2.1.1 "test-tube>=0.7.5" einops==0.3.0 transformers==4.19.2 \
  webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 "invisible-watermark>=0.1.5" torchmetrics==0.6.0 timm==0.6.12 \
  addict==2.4.0 yapf==0.32.0 prettytable==3.6.0 safetensors==0.2.7 scipy natsort imgaug pudb
# 공식 yaml에 있지만 학습·생성 코드가 import하지 않는 데모용(gradio, streamlit, streamlit-drawable-canvas)과 basicsr은 뺐다.

# 공식 yaml에 빠져 있지만 코드가 import하는 것 (latent-diffusion 원래 환경에 있던 것)
#  - taming: upstream/taming은 lpips 가중치만 든 빈 폴더라 VectorQuantizer2를 못 찾는다. CompVis 저장소는 pip로 깔면
#    패키지가 비어서 들어가므로(-e 전용) 같은 코드를 패키지로 묶은 rom1504 판을 쓴다.
#  - clip: ldm/modules/encoders/modules.py가 맨 위에서 import한다.
$PY -m pip install --no-input taming-transformers-rom1504==0.0.6 \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

# OpenCV: 위 패키지들이 끌고 오는 opencv-python(GUI판)은 libSM 등 시스템 라이브러리가 필요하다(apt 금지 서버).
# 코드는 cv2의 기본 함수(resize, cvtColor, imread 등)만 쓰므로 headless판 하나만 남긴다.
$PY -m pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless || true
$PY -m pip install --no-input opencv-python-headless==4.5.5.64
# Pillow 10에서 없어진 PIL.Image.LINEAR를 공식 데이터 로더가 쓴다.
$PY -m pip install --no-input "pillow==9.5.0"

# CUDA 컴파일 도구: 공식 코드가 import할 때 stylegan2 CUDA 연산(fused_act, upfirdn2d)을 torch cpp_extension으로 직접 빌드한다.
# 서버에 gcc·nvcc가 없으므로(apt 금지) 이 환경 안에 넣는다. torch 1.12.1+cu113에 맞춰 CUDA 11.3, gcc 10.
#  - 버전을 다 고정한다. 안 그러면 conda-forge의 CUDA 13 헤더가 섞여 들어온다.
#  - sysroot 2.17: 기본으로 오는 최신 커널 헤더(__int128)는 nvcc 11.3이 못 읽는다.
#  - lib64 링크: torch 1.12는 $CUDA_HOME/lib64에서 cudart를 찾는다. run.py가 CUDA_HOME을 이 환경으로 잡는다.
$CONDA install -y -n Anomalydiffusion --override-channels -c nvidia/label/cuda-11.3.1 -c conda-forge \
  cuda-nvcc=11.3.122 cuda-cudart=11.3.109 cuda-thrust=11.3.109 libcublas=11.5.1.109 libcurand=10.2.4.109 \
  libcusolver=11.1.2.109 libcusparse=11.6.0.109 ninja gcc=10 gxx=10 sysroot_linux-64=2.17
PREFIX=$($PY -c 'import sys; print(sys.prefix)')
[ -e "$PREFIX/lib64" ] || ln -s lib "$PREFIX/lib64"

$PY -c "import torch, pytorch_lightning, cv2, taming.modules.vqvae.quantize, clip; print('env ok', torch.__version__, torch.cuda.is_available())"
echo "### ENV DONE"
