#!/usr/bin/env bash
# VisA 추가 실험 (11/25): "실제 K장 + 합성 N장 = 실제 몇 장" 추정을 탄탄하게 한다.
#   A) 실제만 곡선을 촘촘하게: K = 3 · 5 · 6 × 나누기 0–4 (위치 모델 + 촬영 조건 섞기, 실제만). K 1·2·4·8은 이미 있음
#   B) 합성 세트 추가: K = 2 · 4 × 나누기 1–4 (나누기 0은 이미 있음). 팀과 같은 설정
#        cTransGAN: generate_diverse --diverse-scratch → 휜 핀 잔상 제거 → N 1·5·10·20 부분 세트
#        SD+LoRA  : train(1000 step, rank 16, p_normal 0.1, context 512) → generate --diverse --overgen 2 → 잔상 제거 → 부분 세트
#   C) B의 칸마다 위치 모델 + 촬영 조건 섞기 × 합성 N {1, 5, 10, 20, 40} (배경 섞은 시험셋)
# 결과: AnomalyDetection/results/fair_ad_K{k}_seed{s}.csv (A), fair_ad_K{k}_seed{s}_grid_{ctg,v3}.csv (C)
# 이미 있는 결과·세트는 건너뛴다. 언제 다시 띄워도 된다.
#   cd <저장소 루트> && setsid nohup bash AnomalyDetection/results/logs_jh/run_visa_equiv.sh \
#       > AnomalyDetection/results/logs_jh/run_visa_equiv.console.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # 저장소 루트
cd $ROOT
PY=${PY:-python}   # 다른 파이썬이면 PY=/경로/python bash ... 로 넘긴다
DR=$ROOT/Dataset/visa/VisA_official/VisA_PCB
L=AnomalyDetection/results/logs_jh
GEN=Dataset/generated
NS=(1 5 10 20)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
stage() { echo "### $* $(date '+%F %T')"; }

merge() {   # k seed tag
  $PY - "$1" "$2" "$3" <<'EOF'
import sys, pandas as pd
from pathlib import Path
k, s, tag = sys.argv[1], sys.argv[2], sys.argv[3]
r = Path('AnomalyDetection/results')
parts = [r / f'fair_ad_K{k}_seed{s}{tag}_{p}.csv' for p in ('pcb1', 'pcb2', 'pcb3')]
miss = [p.name for p in parts if not p.exists()]
if miss:
    print('### 빠진 PCB:', miss)
else:
    out = r / f'fair_ad_K{k}_seed{s}{tag}.csv'
    pd.concat([pd.read_csv(p) for p in parts]).to_csv(out, index=False)
    for p in parts:
        p.unlink()
    d = pd.read_csv(out)
    g = d.groupby(['model', 'train', 'pcb'])[['fair_fnr5', 'fnr']].mean().groupby(['model', 'train']).mean()
    print(f'### 합침 {out.name}\n' + g.round(3).to_string())
EOF
}

fair() {    # k seed tag, fair_ad 인자...
  local k=$1 s=$2 tag=$3; shift 3
  [ -f AnomalyDetection/results/fair_ad_K${k}_seed${s}${tag}.csv ] && { stage "K$k seed $s$tag 이미 있음"; return 0; }
  stage "K$k seed $s$tag start"
  local pids=()
  for p in pcb1 pcb2 pcb3; do
    $PY -m AnomalyDetection.ad.fair_ad --k "$k" --seed "$s" --pcbs $p --tag "$tag" --data-root "$DR" \
      --seg-seeds 2 --img-seeds 2 --img-kinds devnet "$@" > $L/fair_ad_K${k}_seed${s}${tag}_$p.log 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  merge "$k" "$s" "$tag"
  stage "K$k seed $s$tag done"
}

gen_ctg() {  # k seed
  local k=$1 s=$2 raw=../$GEN/_before_pinfix/ctransgan_diverse
  while pgrep -f -- "ctransgan.generate_diverse --k $k --seed $s " > /dev/null; do sleep 30; done   # 따로 띄운 생성이 돌면 기다림
  if [ ! -f $GEN/ctransgan_diverse/K${k}_seed${s}/manifest.csv ]; then
    stage "cTransGAN K$k seed $s 생성"
    (cd Augmentation \
      && { [ -f $raw/K${k}_seed${s}/manifest.csv ] \
           || $PY -m ctransgan.generate_diverse --k $k --seed $s --diverse-scratch --device cuda --threads 8 --data-root "$DR" --out $raw; } \
      && $PY -m pin_cleanup.clean --manifest $raw/K${k}_seed${s}/manifest.csv --out-name ctransgan_diverse --k $k --seed $s --data-root "$DR") \
      > $L/ctg_K${k}_seed${s}.log 2>&1 || { stage "cTransGAN K$k seed $s FAILED"; return 1; }
  fi
  [ -f $GEN/ctransgan_diverse_n20/K${k}_seed${s}/manifest.csv ] \
    || (cd Augmentation && $PY -m gen_subset --aug ctransgan_diverse --k $k --seed $s --n "${NS[@]}") >> $L/ctg_K${k}_seed${s}.log 2>&1
}

gen_sd() {   # k seed
  local k=$1 s=$2
  while pgrep -f -- "sd_lora.train --k $k --seed $s " > /dev/null; do sleep 30; done
  if [ ! -f $GEN/sd_lora_v3/K${k}_seed${s}/manifest.csv ]; then
    stage "SD+LoRA K$k seed $s 학습 · 생성"
    (cd Augmentation \
      && $PY -m sd_lora.train --k $k --seed $s --steps 1000 --p-normal 0.1 --rank 16 --context 512 --runs-dir sd_lora/runs_v3 --data-root "$DR" \
      && { [ -f ../$GEN/sd_lora_v3raw/K${k}_seed${s}/manifest.csv ] \
           || $PY -m sd_lora.generate --k $k --seed $s --runs-dir sd_lora/runs_v3 --tag v3raw --diverse --overgen 2 --context 512 --data-root "$DR"; } \
      && $PY -m pin_cleanup.clean --manifest ../$GEN/sd_lora_v3raw/K${k}_seed${s}/manifest.csv --out-name sd_lora_v3 --k $k --seed $s --data-root "$DR") \
      > $L/sd_K${k}_seed${s}.log 2>&1 || { stage "SD+LoRA K$k seed $s FAILED"; return 1; }
  fi
  [ -f $GEN/sd_lora_v3_n20/K${k}_seed${s}/manifest.csv ] \
    || (cd Augmentation && $PY -m gen_subset --aug sd_lora_v3 --k $k --seed $s --n "${NS[@]}") >> $L/sd_K${k}_seed${s}.log 2>&1
}

grid_sets() {  # aug → 위치 모델 + 촬영 조건 섞기 × N 1·5·10·20·40
  local A=$1
  echo --augs ${A}_n1 ${A}_n5 ${A}_n10 ${A}_n20 $A --sets seg_aug:${A}_n1 seg_aug:${A}_n5 seg_aug:${A}_n10 seg_aug:${A}_n20 seg_aug:$A
}

stage "start"
# A) 실제만 곡선
(
  for s in 0 1 2 3 4; do
    for k in 3 5 6; do fair $k $s "" --sets seg_aug:none; done
  done
) &
# B·C) 생성 모델마다 일꾼 2개. 생성이 PCB당 20분 넘게 걸려 칸 두 개를 나란히 돌린다.
#      나누기 1·2번을 먼저 끝내도록 순서를 섞었다 (중간에 멈춰도 K2·K4 모두 나누기가 고르게 늘어남)
W1=("2 1" "4 2" "2 3" "4 4")
W2=("4 1" "2 2" "4 3" "2 4")
for W in W1 W2; do
  declare -n cells=$W
  (
    for c in "${cells[@]}"; do
      set -- $c
      gen_ctg $1 $2 && fair $1 $2 _grid_ctg $(grid_sets ctransgan_diverse)
    done
  ) &
  (
    for c in "${cells[@]}"; do
      set -- $c
      gen_sd $1 $2 && fair $1 $2 _grid_v3 $(grid_sets sd_lora_v3)
    done
  ) &
  unset -n cells
done
wait
stage "ALL DONE"
