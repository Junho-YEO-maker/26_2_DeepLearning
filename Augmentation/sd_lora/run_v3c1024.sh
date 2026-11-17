#!/usr/bin/env bash
# v3 설정 그대로 크롭만 1024(→512로 줄여 넣음): 재학습 → 생성 → photonorm FNR 평가 (v3 512와 비교)
#   cd Augmentation && bash sd_lora/run_v3c1024.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/junho/miniforge3/envs/26_2_dl/bin:$PATH PYTHONUNBUFFERED=1
export DATA_ROOT=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
echo "### sd_lora v3c1024 start $(date '+%F %T')"
CONTEXT=1024 TAG=v3c1024 GEN_CHECK=0 bash sd_lora/run_all.sh || { echo "### sd_lora FAILED $(date '+%F %T')"; exit 1; }
echo "### photonorm eval start $(date '+%F %T')"
AUGS=sd_lora_v3c1024 bash ../AnomalyDetection/results/logs/run_photonorm.sh
echo "### ALL DONE $(date '+%F %T')"
