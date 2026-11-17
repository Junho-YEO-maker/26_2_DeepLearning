#!/usr/bin/env bash
# B(패치 단위 AD)를 서버에서 혼자 끝까지 돌린다: A(counterfactual_check)가 끝나길 기다림 → 짧은 시험(pcb3, 시드 1, 300 step)
# → 성공하면 본 실행(PCB 3개, 시드 5개, sd_lora v2·v3·v3c1024·anomalydiffusion). SSH가 끊겨도 계속된다(setsid nohup).
#   setsid nohup bash AnomalyDetection/results/logs/run_patch_ad.sh > AnomalyDetection/results/logs/run_patch_ad.console.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/home/junho/miniforge3/envs/26_2_dl/bin/python}
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOGS=AnomalyDetection/results/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

echo "### wait for A $(date '+%F %T')"
while pgrep -f "AnomalyDetection.ad.counterfactual_check" > /dev/null; do sleep 30; done

echo "### B smoke test start $(date '+%F %T')"
if ! $PY -m AnomalyDetection.ad.patch_ad --augs sd_lora_v3 --pcbs pcb3 --head-seeds 1 --steps 300 --data-root "$DR" \
     > "$LOGS/patch_ad_smoke.log" 2>&1; then
  echo "### B smoke test FAILED $(date '+%F %T')"; tail -30 "$LOGS/patch_ad_smoke.log"; exit 1
fi
grep -v Warning "$LOGS/patch_ad_smoke.log" | tail -12
mv AnomalyDetection/results/patch_ad_K1_seed0.csv AnomalyDetection/results/patch_ad_K1_seed0_smoke.csv

echo "### B full run start $(date '+%F %T')"
if ! $PY -m AnomalyDetection.ad.patch_ad --augs sd_lora sd_lora_v3 sd_lora_v3c1024 anomalydiffusion --k 1 --seed 0 \
     --data-root "$DR" > "$LOGS/patch_ad_K1_seed0.log" 2>&1; then
  echo "### B full run FAILED $(date '+%F %T')"; tail -30 "$LOGS/patch_ad_K1_seed0.log"; exit 1
fi
grep -v Warning "$LOGS/patch_ad_K1_seed0.log" | tail -25
echo "### B DONE $(date '+%F %T')"
