#!/usr/bin/env bash
# 강화 표준화(prep=std)로 지름길이 사라지는지 확인한다. SSH가 끊겨도 계속된다(setsid nohup).
#   1) defect_removal_check --prep std --remove-dilate 40 : 실제만 학습한 DevNet의 DR(지름길)/ND(결함 자체), 진단 캐시 생성
#   2) label_weight_check --prep std : 원래 시험 FNR + 진단, 생성 비율·약하게 조건별
#   setsid nohup bash AnomalyDetection/results/logs/run_std_check.sh > AnomalyDetection/results/logs/run_std_check.console.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/junho/miniforge3/envs/26_2_dl/bin/python}
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOGS=AnomalyDetection/results/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

echo "### std defect_removal start $(date '+%F %T')"
if ! $PY -m AnomalyDetection.ad.defect_removal_check --prep std --remove-dilate 40 --aug sd_lora_v3 --data-root "$DR" \
     > "$LOGS/defect_removal_K1_seed0_std_dil40.log" 2>&1; then
  echo "### std defect_removal FAILED $(date '+%F %T')"; tail -30 "$LOGS/defect_removal_K1_seed0_std_dil40.log"; exit 1
fi
grep -v Warning "$LOGS/defect_removal_K1_seed0_std_dil40.log" | tail -12

echo "### std label_weight start $(date '+%F %T')"
if ! $PY -m AnomalyDetection.ad.label_weight_check --prep std --diag-tag _std_dil40 --aug sd_lora_v3 --data-root "$DR" \
     > "$LOGS/label_weight_K1_seed0_std.log" 2>&1; then
  echo "### std label_weight FAILED $(date '+%F %T')"; tail -30 "$LOGS/label_weight_K1_seed0_std.log"; exit 1
fi
grep -v Warning "$LOGS/label_weight_K1_seed0_std.log" | tail -26
echo "### std DONE $(date '+%F %T')"
