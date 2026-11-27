#!/usr/bin/env bash
# SMT 추가 실험 (11/26): 학습 배치의 불량 자리 중 합성 비율을 1/2 → 1/4로 줄이면 어떻게 되나.
#   배치 8장 = 정상 4 + 불량 4. 기본(0.5)은 불량 4장 중 2장이 합성, 0.25는 1장이 합성.
#   (0.1은 반올림하면 0장이라 뜻이 없다)
#   두 공정 × 실제 K 1 · 4 · 16 × 나누기 0–2 × 생성 모델 2개(SD+LoRA = sd_lora_fix, cTransGAN), 합성 40장.
#   비교 기준: 같은 칸의 비율 0.5 결과와 실제만(none)은 이미 seg_ad_{pre,solder}.csv에 있다.
# 결과: AnomalyDetection/results/smt/seg_ad_{pre,solder}.csv 에 행이 더해진다 (이미 있는 칸은 건너뜀).
#   cd <저장소 루트> && setsid nohup bash AnomalyDetection/results/logs_jh/run_smt_ratio.sh \
#       > AnomalyDetection/results/logs_jh/run_smt_ratio.console.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # 저장소 루트
cd $ROOT
PY=${PY:-python}   # 다른 파이썬이면 PY=/경로/python bash ... 로 넘긴다
L=AnomalyDetection/results/logs_jh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
stage() { echo "### $* $(date '+%F %T')"; }

seg() {   # category k
  local c=$1 k=$2
  stage "$c K$k 시작"
  $PY -m AnomalyDetection.smt.seg_ad --category $c --k $k --seeds 0 1 2 \
      --conds sd_lora_fix:40:0.25 ctransgan:40:0.25 >> $L/smt_ratio_${c}_K${k}.log 2>&1 \
    || stage "$c K$k FAILED (exit $?)"
  stage "$c K$k 끝"
}

stage "start"
for c in pre solder; do
  for k in 1 4 16; do seg $c $k & done
done
wait
stage "ALL DONE"
