#!/usr/bin/env bash
# 공정 시험셋 K 곡선 (11/18, ad/fair_ad.py). 실제 불량 K = 1, 2, 4, 8, 10 × 분할 seed 0–4, PCB마다 병렬 프로세스.
#   K=1 seed 0   run_fair_ad.sh가 따로 돌렸다(PatchCore + 생성 4종, 헤드 시드 3)
#   나머지        seg(실제만) / seg_aug(촬영 조건 무작위화) 실제만·+copypaste(·+v3) / 이미지 DevNet, 헤드 시드 2
#   v3           K=1은 seed 0–4, K 2·4·8·10은 seed 0만 (LoRA를 K마다 새로 학습해야 해서). 생성이 끝나는 대로 따로 돌려 _v3 태그로 저장
#   seed마다 흐름 하나 + v3 흐름, 모두 동시에 (11:3x 재시작: 동시 실행 수를 늘려도 한 번 실행 시간이 거의 그대로라서).
#   이미 합친 CSV가 있으면 건너뛰고, 같은 조건의 fair_ad가 이미 돌고 있으면 끝나기를 기다렸다가 합친다. 그래서 언제 다시 띄워도 된다.
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_fair_kcurve.sh \
#       > AnomalyDetection/results/logs/run_fair_kcurve.console.log 2>&1 &
# 결과: AnomalyDetection/results/fair_ad_K{k}_seed{s}[_v3].csv
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOGS=AnomalyDetection/results/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
COMMON=(--seg-seeds 2 --img-seeds 2 --img-kinds devnet)

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

run() {   # k seed tag, fair_ad 인자...
  local k=$1 s=$2 tag=$3; shift 3
  [ -f AnomalyDetection/results/fair_ad_K${k}_seed${s}${tag}.csv ] && { echo "### K$k seed $s$tag 이미 있음"; return; }
  local pat="ad.fair_ad --k $k --seed $s --pcbs pcb[123] --tag $tag --data-root"
  if pgrep -f -- "$pat" > /dev/null; then   # 앞선 러너가 띄운 것이 아직 돈다 → 이어받기
    echo "### K$k seed $s$tag 이어받음 $(date '+%F %T')"
    while pgrep -f -- "$pat" > /dev/null; do sleep 30; done
  else
    echo "### K$k seed $s$tag start $(date '+%F %T')"
    local pids=()
    for p in pcb1 pcb2 pcb3; do
      $PY -m AnomalyDetection.ad.fair_ad --k "$k" --seed "$s" --pcbs $p --tag "$tag" --data-root "$DR" "${COMMON[@]}" "$@" \
        > $LOGS/fair_ad_K${k}_seed${s}${tag}_$p.log 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}"   # 이 PCB 3개만 기다린다 (그냥 wait는 다른 흐름의 서브셸까지 기다려서 멈췄었음)
  fi
  merge "$k" "$s" "$tag"
  echo "### K$k seed $s$tag done $(date '+%F %T')"
}

v3_ready() {   # k seed: 생성 스크립트가 그 조건을 끝냈나
  if [ "$1" = 1 ]; then grep -q "### seed $2 exit" Augmentation/sd_lora/run_v3_seeds.console.log 2>/dev/null
  else grep -q "### K$1 exit" Augmentation/sd_lora/run_v3_kcurve.console.log 2>/dev/null; fi
}

seed_all() {   # seed: K=1(+v3) 다음 K 2·4·8·10
  local s=$1
  until v3_ready 1 $s; do sleep 60; done
  run 1 $s "" --augs sd_lora_v3 copypaste \
    --sets seg:none seg:sd_lora_v3 seg_aug:none seg_aug:copypaste seg_aug:sd_lora_v3 img:none img:sd_lora_v3
  for k in 2 4 8 10; do
    run $k $s "" --augs copypaste --sets seg:none seg_aug:none seg_aug:copypaste img:none
  done
}

# v3 K 2·4·8·10 seed 0
(
  for k in 2 4 8 10; do
    until v3_ready $k 0; do sleep 60; done
    run $k 0 _v3 --augs sd_lora_v3 --sets seg:sd_lora_v3 seg_aug:sd_lora_v3 img:sd_lora_v3
  done
) &
# seed 0 (K=1은 run_fair_ad.sh가 따로 돌렸다)
(
  for k in 2 4 8 10; do
    run $k 0 "" --augs copypaste --sets seg:none seg_aug:none seg_aug:copypaste img:none
  done
) &
for s in 1 2 3 4; do
  seed_all $s &
done
wait
echo "### ALL DONE $(date '+%F %T')"
