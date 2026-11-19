#!/usr/bin/env bash
# 남은 실험 예약 (11/18 16시 50분, 사용자 요청 "나머지 실험도 예약 걸어줘"). 지금 도는 두 실행과 GPU를 겹치지 않게 각각 앞 작업이 끝나면 시작한다.
#
# A. SMT 납땜 ms 따로 실험 (run_smt_fix_grid.sh가 끝난 뒤)
#    ms만 실제 빈 패드 기반으로 바꾼 세트(Augmentation/smt_sd_lora/generate_ms_real.py → sd_lora_fix_msreal)를 K 1·4·16 × seed 0–2로 만들고,
#    납땜 seg를 N 5·40으로 돌린다. 비교 대상 = 같은 칸의 sd_lora_fix(ms를 SD가 그림)와 실제만. 다른 유형은 같은 파일이라 ms만 다르다.
# B. VisA cTransGAN K1 seed 1–4 (run_pinfix_rerun.sh의 AD 격자가 끝난 뒤)
#    생성 → 검수 그림 → bent 잔상 제거 → N 부분 세트 → AD 칸(합성 N 1·5·10·20·40). SD+LoRA K1 seed 1–4와 같은 설정.
#    누가 먼저 만들어 두었으면(보관함 또는 고친 세트에 manifest가 있으면) 생성은 건너뛴다.
# C. 집계 (A·B와 run_pinfix_rerun.sh가 모두 끝난 뒤): grid_summary(생성기 × K × N 그림·표), ppt_summary, ppt_figures/collect.sh
#
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_followups.sh \
#       > AnomalyDetection/results/logs/run_followups.console.log 2>&1 &
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB
LOGS=AnomalyDetection/results/logs
SLOG=AnomalyDetection/results/smt/logs
GEN=Dataset/generated
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
COMMON=(--seg-seeds 2 --img-seeds 2 --img-kinds devnet)
NS=(1 5 10 20)
stage() { echo "### $* $(date '+%F %T')"; }

# ---------- A. SMT ms 따로 실험 ----------
part_a() {
  stage "A 기다림 (run_smt_fix_grid.sh ALL DONE)"
  until grep -q "### ALL DONE" $LOGS/run_smt_fix_grid.console.log 2>/dev/null; do sleep 120; done
  stage "A start"
  local pids=()
  for ks in "1 0" "1 1" "1 2" "4 0" "4 1" "4 2" "16 0" "16 1" "16 2"; do
    set -- $ks
    local k=$1 s=$2
    (cd Augmentation && $PY -m smt_sd_lora.generate_ms_real --k $k --seed $s) \
      > Augmentation/smt_sd_lora/run_msreal_K${k}_seed${s}.log 2>&1 || { stage "A gen K$k seed $s FAILED"; continue; }
    stage "A gen K$k seed $s done"
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done
    $PY -m AnomalyDetection.smt.seg_ad --category solder --k $k --seeds $s \
      --conds sd_lora_fix_msreal:5:0.5 sd_lora_fix_msreal:40:0.5 > $SLOG/seg_ad_msreal_K${k}_seed${s}.log 2>&1 &
  done
  wait
  stage "A done"
}

# ---------- B. VisA cTransGAN K1 seed 1–4 ----------
merge() {   # k seed tag (run_pinfix_rerun.sh와 같음)
  $PY - "$1" "$2" "$3" <<'EOF'
import sys, pandas as pd
from pathlib import Path
k, s, tag = sys.argv[1], sys.argv[2], sys.argv[3]
r = Path('AnomalyDetection/results')
parts = [r / f'fair_ad_K{k}_seed{s}{tag}_{p}.csv' for p in ('pcb1', 'pcb2', 'pcb3')]
miss = [p.name for p in parts if not p.exists()]
if miss:
    print('### 빠진 PCB:', miss); sys.exit()
out = r / f'fair_ad_K{k}_seed{s}{tag}.csv'
d = pd.concat([pd.read_csv(p) for p in parts])
d.to_csv(out, index=False)
for p in parts:
    p.unlink()
g = d.groupby(['model', 'train', 'pcb'])[['fair_fnr5', 'fair_fnr_bent']].mean().groupby(['model', 'train']).mean()
print(f'### 합침 {out.name}\n' + g.round(3).to_string())
EOF
}

cell_ctg() {   # seed: K1 cTransGAN 칸 (N 1·5·10·20 seg_aug + N 40 seg_aug·seg·DevNet)
  local s=$1 k=1 tag=_grid_ctg aug=ctransgan_diverse
  [ -f AnomalyDetection/results/fair_ad_K${k}_seed${s}${tag}.csv ] && { echo "### K$k seed $s$tag 이미 있음"; return; }
  local augs=() sets=()
  for n in "${NS[@]}"; do augs+=(${aug}_n$n); sets+=(seg_aug:${aug}_n$n); done
  augs+=($aug); sets+=(seg_aug:$aug seg:$aug img:$aug)
  stage "B K$k seed $s$tag start"
  local pids=()
  for p in pcb1 pcb2 pcb3; do
    $PY -m AnomalyDetection.ad.fair_ad --k $k --seed $s --pcbs $p --tag $tag --data-root "$DR" \
      --augs "${augs[@]}" --sets "${sets[@]}" "${COMMON[@]}" > $LOGS/fair_ad_K${k}_seed${s}${tag}_$p.log 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  merge $k $s $tag
  stage "B K$k seed $s$tag done"
}

part_b() {
  stage "B 기다림 (run_pinfix_rerun.sh AD 격자 done)"
  until grep -q "### AD 격자 done" $LOGS/run_pinfix_rerun.console.log 2>/dev/null; do sleep 120; done
  stage "B start"
  local RAW=../$GEN/_before_pinfix/ctransgan_diverse cells=()
  for s in 1 2 3 4; do
    if [ ! -f $GEN/ctransgan_diverse/K1_seed$s/manifest.csv ]; then
      (cd Augmentation \
        && { [ -f $RAW/K1_seed$s/manifest.csv ] \
             || $PY -m ctransgan.generate_diverse --k 1 --seed $s --diverse-scratch --device cuda --threads 8 --data-root "$DR" --out $RAW; } \
        && $PY -m ctransgan.review --k 1 --seed $s --folder $RAW/K1_seed$s --data-root "$DR" --out ctransgan/review/diverse_K1_seed$s.png \
        && $PY -m pin_cleanup.clean --manifest $RAW/K1_seed$s/manifest.csv --out-name ctransgan_diverse --k 1 --seed $s --data-root "$DR") \
        > $LOGS/ctransgan_K1_seed$s.log 2>&1 || { stage "B gen seed $s FAILED"; continue; }
      stage "B gen seed $s done"
    fi
    (cd Augmentation && $PY -m gen_subset --aug ctransgan_diverse --k 1 --seed $s --n "${NS[@]}") >> $LOGS/ctransgan_K1_seed$s.log 2>&1
    cell_ctg $s &
    cells+=($!)
  done
  wait "${cells[@]}"
  stage "B done"
}

part_a > $LOGS/run_followups_A.log 2>&1 &
PA=$!
part_b > $LOGS/run_followups_B.log 2>&1 &
PB=$!
wait $PA $PB
grep "^###" $LOGS/run_followups_A.log $LOGS/run_followups_B.log

# ---------- C. 집계 ----------
stage "C 기다림 (run_pinfix_rerun.sh ALL DONE)"
until grep -q "### ALL DONE" $LOGS/run_pinfix_rerun.console.log 2>/dev/null; do sleep 120; done
$PY -m AnomalyDetection.ad.grid_summary > $LOGS/grid_summary.log 2>&1; rc1=$?
$PY -m AnomalyDetection.ad.ppt_summary > $LOGS/ppt_summary_final.log 2>&1; rc2=$?
bash ppt_figures/collect.sh >> $LOGS/ppt_summary_final.log 2>&1
cp AnomalyDetection/results/ppt/grid_visa.png ppt_figures/s14_grid_visa.png
cp AnomalyDetection/results/ppt/grid_smt.png ppt_figures/s14_grid_smt.png
stage "ALL DONE (grid_summary $rc1, ppt_summary $rc2)"
