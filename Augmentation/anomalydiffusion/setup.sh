#!/usr/bin/env bash
# AnomalyDiffusion 공식 코드와 환경 준비 (A100 리눅스 서버에서 한 번만).
#   cd Augmentation/anomalydiffusion && bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d upstream ]; then
  git clone https://github.com/sjtuplayer/anomalydiffusion.git upstream
fi
git -C upstream rev-parse HEAD > upstream_commit.txt
echo "공식 코드 커밋: $(cat upstream_commit.txt)"

# 공식 환경과 같은 버전: python 3.8, pytorch 1.12.1 + cu113 (A100 지원).
# 공식 environment.yaml은 Anaconda defaults 채널을 쓰므로 env_setup.sh가 conda-forge + PyTorch pip 휠로 만든다.
bash env_setup.sh

# 사전학습 LDM (공식 README 주소, 약 6.2GB). -c: 끊기면 이어 받기
mkdir -p upstream/models/ldm/text2img-large/
if [ "$(stat -c %s upstream/models/ldm/text2img-large/model.ckpt 2>/dev/null || echo 0)" != 6152314307 ]; then
  wget -c -O upstream/models/ldm/text2img-large/model.ckpt \
    https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
fi

# 우리 어댑터(run.py)는 아무 python에서나 돈다: numpy, pillow, pyyaml 필요
python -c "import numpy, PIL, yaml" || pip install numpy pillow pyyaml
echo "준비 완료. README의 '처음 실행 점검'부터 하세요."
