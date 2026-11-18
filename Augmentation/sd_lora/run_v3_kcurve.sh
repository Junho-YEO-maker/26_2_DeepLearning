#!/usr/bin/env bash
# v3를 K {2,4,8,10} × seed 0으로 만든다 (11/18, 공정 시험셋 K 곡선에서 "실제 K장 + 생성" 비교용).
# run_v3_seeds.sh(K1 seed 1–4)가 끝나기를 기다렸다가 K마다 병렬. 레시피는 run_all.sh 기본값(K=1과 같음).
#   cd Augmentation && setsid nohup bash sd_lora/run_v3_kcurve.sh > sd_lora/run_v3_kcurve.console.log 2>&1 &
# 로그: sd_lora/run_v3_K{k}_seed0.log, 결과: Dataset/generated/sd_lora_v3/K{k}_seed0/
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/junho/miniforge3/envs/26_2_dl/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export DATA_ROOT=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
until grep -q "ALL DONE" sd_lora/run_v3_seeds.console.log 2>/dev/null; do sleep 60; done
echo "### start $(date '+%F %T')"
for k in ${KS:-2 4 8 10}; do
  (KS=$k SEED=${SEED:-0} GEN_CHECK=0 bash sd_lora/run_all.sh > /dev/null 2>&1; echo "### K$k exit $? $(date '+%F %T')") &
done
wait
echo "### ALL DONE $(date '+%F %T')"
