#!/usr/bin/env bash
# AnomalyDiffusion 한 조건(prepare → train → generate → export)을 GPU를 같이 쓰는 다른 작업 옆에서 안전하게 돌린다.
#   cd Augmentation && K=1 SEED=0 STEPS=6000 bash anomalydiffusion/run_safe.sh
# CUDA OOM으로 단계가 실패하면: 같은 서버의 다른 GPU 작업(sd_lora 학습·생성, AD 평가)이 끝날 때까지 기다렸다가 그 단계만 다시 한다(최대 2번).
#   - generate는 이미 만든 샘플을 건너뛰므로 이어서 진행된다. train은 처음부터 다시 한다.
#   - OOM이 아닌 에러는 바로 멈춘다.
set -uo pipefail
cd "$(dirname "$0")/.."
K=${K:-1}; SEED=${SEED:-0}; STEPS=${STEPS:-6000}
PY=${PY:-/home/junho/miniforge3/envs/26_2_dl/bin/python}
ADPY=${ADPY:-/home/junho/miniforge3/envs/Anomalydiffusion/bin/python}
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOG=anomalydiffusion/run_K${K}_seed${SEED}.log
OTHERS='sd_lora\.train|sd_lora\.generate|AnomalyDetection\.ad\.'
export PYTHONUNBUFFERED=1

run_stage() {   # $1 = stage
  local tries=0
  while :; do
    echo "### $1 start $(date '+%F %T') (try $((tries + 1)))" | tee -a "$LOG"
    local tmp; tmp=$(mktemp -p anomalydiffusion --suffix=.log stage.XXXX)
    $PY -m anomalydiffusion.run "$1" --k "$K" --seed "$SEED" --steps "$STEPS" --data-root "$DR" --ad-python "$ADPY" \
      > "$tmp" 2>&1
    local rc=$?
    cat "$tmp" >> "$LOG"
    if [ $rc -eq 0 ]; then rm -f "$tmp"; echo "### $1 done $(date '+%F %T')" | tee -a "$LOG"; return 0; fi
    if grep -q "CUDA out of memory" "$tmp" && [ $tries -lt 2 ]; then
      rm -f "$tmp"; tries=$((tries + 1))
      echo "### $1 OOM -> 다른 GPU 작업이 끝날 때까지 대기 $(date '+%F %T')" | tee -a "$LOG"
      while pgrep -f "$OTHERS" > /dev/null; do sleep 60; done
      continue
    fi
    rm -f "$tmp"; echo "### $1 FAILED rc=$rc $(date '+%F %T')" | tee -a "$LOG"; return 1
  done
}

for s in prepare train generate export; do
  run_stage "$s" || exit 1
done
echo "### ALL DONE $(date '+%F %T')" | tee -a "$LOG"
