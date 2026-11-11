#!/usr/bin/env bash
# 1단계 범위: PCB1–3 × K {1, 2, 4} × seed 0.  STEPS는 K=1 시험 후 정한다.
#   cd Augmentation && bash anomalydiffusion/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-6000}
for K in 1 2 4; do
  python -m anomalydiffusion.run all --k "$K" --seed 0 --steps "$STEPS" "$@" 2>&1 | tee -a "anomalydiffusion/run_K${K}_seed0.log"
done
