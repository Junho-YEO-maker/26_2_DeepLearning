#!/usr/bin/env bash
# v3를 K1 seed 1–4에도 만든다 (11/18). 촬영 조건을 뺀 시험셋(fair_ad.py)에서 seed 5개로 비교하기 위해서.
# 레시피는 seed 0과 같다(run_all.sh 기본값: 1000 step, rank 16, p_normal 0.1, --diverse --overgen 2). seed마다 병렬.
#   cd Augmentation && setsid nohup bash sd_lora/run_v3_seeds.sh > sd_lora/run_v3_seeds.console.log 2>&1 &
# 로그: sd_lora/run_v3_K1_seed{s}.log, 결과: Dataset/generated/sd_lora_v3/K1_seed{s}/
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/junho/miniforge3/envs/26_2_dl/bin:$PATH PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export DATA_ROOT=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
echo "### start $(date '+%F %T')"
for s in ${SEEDS:-1 2 3 4}; do
  (SEED=$s GEN_CHECK=0 bash sd_lora/run_all.sh > /dev/null 2>&1; echo "### seed $s exit $? $(date '+%F %T')") &
done
wait
echo "### ALL DONE $(date '+%F %T')"
