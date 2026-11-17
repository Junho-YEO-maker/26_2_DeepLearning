#!/usr/bin/env bash
# GPU 서버용 sd_lora v3: 조건마다 학습(LoRA 3개) → 생성(480장) → 빠른 점검(gen_check). README "v3 방향" 참고.
#
#   cd Augmentation
#   DATA_ROOT=/abs/path/VisA_PCB bash sd_lora/run_all.sh                 # 기본: K=1 seed 0, 단계 B(재학습 + 생성)
#   STAGE=A DATA_ROOT=... bash sd_lora/run_all.sh                         # 단계 A: 기존 v2 LoRA로 생성만 바꿈
#   KS="1 2 4" DATA_ROOT=... bash sd_lora/run_all.sh                      # 1단계 전체
#
# DATA_ROOT는 절대경로(점검 단계에서 저장소 루트로 옮겨 실행하므로). 없으면 Dataset/visa/raw를 쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."

KS=${KS:-1}
SEED=${SEED:-0}
STAGE=${STAGE:-B}               # A: v2 LoRA(sd_lora/runs) 재사용, B: v3 학습 설정으로 다시 학습(sd_lora/runs_v3)
TAG=${TAG:-v3}
STEPS=${STEPS:-1000}
P_NORMAL=${P_NORMAL:-0.1}
RANK=${RANK:-16}
OVERGEN=${OVERGEN:-2}
CONTEXT=${CONTEXT:-512}         # 원본에서 잘라 낼 크기. 1024면 넓게 잘라 512로 줄여 학습·생성 (예: CONTEXT=1024 TAG=v3c1024)
GEN_CHECK=${GEN_CHECK:-1}       # 0이면 gen_check를 건너뛴다 (CPU에서 PCB당 1시간 이상 걸림)
DR=()
if [ -n "${DATA_ROOT:-}" ]; then DR=(--data-root "$DATA_ROOT"); fi

if [ "$STAGE" = A ]; then RUNS=sd_lora/runs; TAG="${TAG}a"; else RUNS="sd_lora/runs_${TAG}"; fi

for K in $KS; do
  LOG="sd_lora/run_${TAG}_K${K}_seed${SEED}.log"
  if [ "$STAGE" != A ]; then
    python -m sd_lora.train --k "$K" --seed "$SEED" --steps "$STEPS" --p-normal "$P_NORMAL" --rank "$RANK" \
      --context "$CONTEXT" --runs-dir "$RUNS" ${DR[@]+"${DR[@]}"} 2>&1 | tee -a "$LOG"
  fi
  python -m sd_lora.generate --k "$K" --seed "$SEED" --runs-dir "$RUNS" --tag "$TAG" --diverse --overgen "$OVERGEN" \
    --context "$CONTEXT" ${DR[@]+"${DR[@]}"} 2>&1 | tee -a "$LOG"
  if [ "$GEN_CHECK" = 1 ]; then
    (cd .. && python -m AnomalyDetection.ad.gen_check --aug "sd_lora_${TAG}" --k "$K" --seed "$SEED" \
      ${DR[@]+"${DR[@]}"}) 2>&1 | tee -a "$LOG"
  fi
done
