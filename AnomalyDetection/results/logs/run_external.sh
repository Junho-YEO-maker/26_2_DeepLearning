#!/usr/bin/env bash
# 3단계 (11/20, 사용자 승인 "3단계까지 다"): 공개 지도 학습 방법(DRA, DevNet 이미지판) 공식 코드를 SegAD 규칙으로 학습 →
# 공식 시험 이미지 AUROC(SegAD 논문 수치와 대조) + 우리 공정한 시험. PCB1–3 × seed 0–4, 두 방법 동시에.
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_external.sh \
#       > AnomalyDetection/results/logs/run_external.console.log 2>&1 &
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "### start $(date '+%F %T')"
for m in devnet dra; do
  $PY -m AnomalyDetection.ad.external_eval --method $m > AnomalyDetection/results/logs/external_$m.log 2>&1 \
    && echo "### $m done $(date '+%F %T')" || echo "### $m FAILED $(date '+%F %T')" &
done
wait
echo "### ALL DONE $(date '+%F %T')"
