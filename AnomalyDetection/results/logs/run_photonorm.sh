#!/usr/bin/env bash
# 13.5절 다음 할 일 1: 밝기·색 지름길을 없애면(photonorm) none vs 생성 증강 FNR이 어떻게 바뀌는지 GPU 서버에서 확인.
#   bash AnomalyDetection/results/logs/run_photonorm.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/junho/miniforge3/envs/26_2_dl/bin/python}
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
for AUG in ${AUGS:-sd_lora sd_lora_v3}; do
  echo "### $AUG start $(date '+%F %T')"
  $PY -m AnomalyDetection.ad.shortcut_check --photonorm --aug "$AUG" --k 1 --seed 0 --data-root "$DR" \
    2>&1 | grep -v Warning | tee "AnomalyDetection/results/logs/shortcut_photonorm_${AUG}_K1_seed0.log"
done
echo "### DONE $(date '+%F %T')"
