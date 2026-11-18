#!/usr/bin/env bash
# Grad-CAM과 결함 지도 (11/18, ad/gradcam.py). K=1 seed 0, PCB마다 병렬, 끝나면 합치기.
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_gradcam.sh \
#       > AnomalyDetection/results/logs/run_gradcam.console.log 2>&1 &
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "### start $(date '+%F %T')"
pids=()
for p in pcb1 pcb2 pcb3; do
  $PY -m AnomalyDetection.ad.gradcam --k 1 --seed 0 --pcbs $p --data-root $DR > AnomalyDetection/results/logs/gradcam_K1_seed0_$p.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
$PY -m AnomalyDetection.ad.gradcam --k 1 --seed 0 --summarize
echo "### ALL DONE $(date '+%F %T')"
