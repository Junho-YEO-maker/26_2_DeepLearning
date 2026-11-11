#!/usr/bin/env bash
# 1단계 범위: PCB1–3 × K {1, 2, 4} × seed 0.  학습(LoRA 3개) → 생성(480장) 순서.
#   cd Augmentation && bash sd_lora/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-1000}
for K in 1 2 4; do
  python -m sd_lora.train --k "$K" --seed 0 --steps "$STEPS" "$@" 2>&1 | tee -a "sd_lora/run_K${K}_seed0.log"
  python -m sd_lora.generate --k "$K" --seed 0 "$@" 2>&1 | tee -a "sd_lora/run_K${K}_seed0.log"
done
