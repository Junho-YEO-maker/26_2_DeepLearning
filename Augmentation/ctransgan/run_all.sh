#!/usr/bin/env bash
# 범위: PCB1–3 × K {1, 2, 4} × seed 0. 조건마다 PCB별로 학습 → 480장 생성 → 누수 검사 → 확인용 시트.
#   cd Augmentation && bash ctransgan/run_all.sh                      # 원본이 Dataset/visa/raw에 있을 때
#   cd Augmentation && bash ctransgan/run_all.sh --data-root /경로/VisA_PCB
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ARGS=()
if [ "${1:-}" = "--data-root" ]; then DATA_ARGS=(--data-root "$2"); fi
for K in 1 2 4; do
  python -m ctransgan.generate --k "$K" --seed 0 "$@" 2>&1 | tee -a "ctransgan/run_K${K}_seed0.log"
  python -m ctransgan.review --k "$K" --seed 0 "${DATA_ARGS[@]}" 2>&1 | tee -a "ctransgan/run_K${K}_seed0.log"
done
