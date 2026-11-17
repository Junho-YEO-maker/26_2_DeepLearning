#!/usr/bin/env bash
# 생성 방법 4가지를 원래 시험셋 FNR@FPR1%로 비교 (none / 전부 / 무작위 40 / edit 상위 40, DevNet·MLP, 헤드 시드 5개)
#   bash AnomalyDetection/results/logs/run_gen_select_all.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/junho/miniforge3/envs/26_2_dl/bin/python}
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
for PREP in photonorm canon; do
  echo "### $PREP start $(date '+%F %T')"
  $PY -m AnomalyDetection.ad.gen_select --augs sd_lora sd_lora_v3 sd_lora_v3c1024 anomalydiffusion --k 1 --seed 0 \
    --prep "$PREP" --data-root "$DR" 2>&1 | grep -v Warning | tee "AnomalyDetection/results/logs/gen_select_all_K1_seed0_${PREP}.log"
done
echo "### DONE $(date '+%F %T')"
