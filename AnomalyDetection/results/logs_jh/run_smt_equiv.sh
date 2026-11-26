#!/usr/bin/env bash
# SMT 추가 실험 (11/26): "실제 K장 + 합성 N장 = 실제 몇 장" 을 SMT에서도 잰다. VisA 실행(run_visa_equiv.sh)이 끝난 뒤 시작.
#   A) 실제만 곡선을 촘촘하게: K = 2 · 8 · 32 × 나누기 0–4 (사전·납땜). 기존은 K 1 · 4 · 16 × 나누기 0–2뿐
#      + K 1 · 4 · 16 × 나누기 3–4 (실제만)
#   B) 사전 공정 SD+LoRA 합성 나누기 3–4 추가: K 1 · 4 · 16 × N 1 · 5 · 10 · 20 · 40
#      (사전 공정은 sd_lora와 sd_lora_fix가 같은 행이라 이미 있는 sd_lora/K*_seed{3,4}를 쓴다. 생성 없음)
#      cTransGAN·납땜 SD는 나누기 3–4 생성물이 없어 이번에는 뺀다.
# 모델·지표는 팀과 같다: AnomalyDetection.smt.seg_ad (위치 모델, 2000 step, 시험 FNR@과검 1%, AUROC)
# 결과: AnomalyDetection/results/smt/seg_ad_{pre,solder}.csv 에 행이 더해진다. 이미 있는 (K, 나누기, 조건)은 건너뛴다.
#   cd <저장소 루트> && setsid nohup bash AnomalyDetection/results/logs_jh/run_smt_equiv.sh \
#       > AnomalyDetection/results/logs_jh/run_smt_equiv.console.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # 저장소 루트
cd $ROOT
PY=${PY:-python}   # 다른 파이썬이면 PY=/경로/python bash ... 로 넘긴다
L=AnomalyDetection/results/logs_jh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
stage() { echo "### $* $(date '+%F %T')"; }
SD="sd_lora:1:0.5 sd_lora:5:0.5 sd_lora:10:0.5 sd_lora:20:0.5 sd_lora:40:0.5"

seg() {   # category k "seeds" conds...
  local c=$1 k=$2 seeds=$3; shift 3
  stage "$c K$k 나누기 $seeds 시작 ($*)"
  $PY -m AnomalyDetection.smt.seg_ad --category $c --k $k --seeds $seeds --conds "$@" >> $L/smt_${c}_K${k}.log 2>&1 \
    || stage "$c K$k FAILED (exit $?)"
  stage "$c K$k 나누기 $seeds 끝"
}

stage "VisA 실행이 끝나기를 기다림"
while pgrep -f "run_visa_equiv.sh" > /dev/null; do sleep 120; done
stage "start"
# 실제만 곡선을 먼저 (나누기 0–2 → 3–4 순), 그다음 사전 공정 SD 나누기 3–4
( seg pre 2 "0 1 2 3 4" none;  seg pre 1 "3 4" none $SD ) &
( seg pre 8 "0 1 2 3 4" none;  seg pre 4 "3 4" none $SD ) &
( seg pre 32 "0 1 2 3 4" none; seg pre 16 "3 4" none $SD ) &
( seg solder 2 "0 1 2 3 4" none; seg solder 32 "0 1 2 3 4" none ) &
( seg solder 8 "0 1 2 3 4" none; seg solder 1 "3 4" none; seg solder 4 "3 4" none; seg solder 16 "3 4" none ) &
wait
stage "ALL DONE"
