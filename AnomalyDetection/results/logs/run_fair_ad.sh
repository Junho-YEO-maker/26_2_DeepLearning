#!/usr/bin/env bash
# 공정 시험셋(촬영 조건 반반) 평가 (11/18, ad/fair_ad.py). PCB마다 병렬 프로세스.
#   seed 0   : PatchCore + 생성 4종(sd_lora_v3, sd_lora v2, anomalydiffusion, copypaste) 비교
#   seed 1–4 : sd_lora_v3, copypaste. v3는 Augmentation/sd_lora/run_v3_seeds.sh가 그 seed를 끝낼 때까지 기다린다.
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_fair_ad.sh \
#       > AnomalyDetection/results/logs/run_fair_ad.console.log 2>&1 &
# 결과: AnomalyDetection/results/fair_ad_K1_seed{s}.csv, 로그 logs/fair_ad_K1_seed{s}_{pcb}.log
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOGS=AnomalyDetection/results/logs
GEN_LOG=Augmentation/sd_lora/run_v3_seeds.console.log
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

run_seed() {   # seed, fair_ad 인자...
  local s=$1; shift
  echo "### seed $s start $(date '+%F %T')"
  for p in pcb1 pcb2 pcb3; do
    $PY -m AnomalyDetection.ad.fair_ad --seed "$s" --pcbs $p --data-root "$DR" "$@" > $LOGS/fair_ad_K1_seed${s}_$p.log 2>&1 &
  done
  wait
  $PY - "$s" <<'EOF'
import sys, pandas as pd
from pathlib import Path
s, r = sys.argv[1], Path('AnomalyDetection/results')
parts = [r / f'fair_ad_K1_seed{s}_{p}.csv' for p in ('pcb1', 'pcb2', 'pcb3')]
missing = [p.name for p in parts if not p.exists()]
if missing:
    print('### 빠진 PCB:', missing)
else:
    pd.concat([pd.read_csv(p) for p in parts]).to_csv(r / f'fair_ad_K1_seed{s}.csv', index=False)
    for p in parts:
        p.unlink()
    print('### 합침:', r / f'fair_ad_K1_seed{s}.csv')
EOF
  grep -h -E "공정|patchcore" $LOGS/fair_ad_K1_seed${s}_pcb*.log | grep -v "시드" | tail -40
  echo "### seed $s done $(date '+%F %T')"
}

until [ -f Dataset/generated/copypaste/K1_seed0/manifest.csv ] && [ -n "$(ls Dataset/generated/copypaste/K1_seed4/images 2>/dev/null | head -1)" ] \
      && ! pgrep -f "copypaste.generate --k 1" > /dev/null; do sleep 30; done
run_seed 0 --patchcore --augs sd_lora_v3 sd_lora anomalydiffusion copypaste
for s in ${SEEDS:-1 2 3 4}; do
  until grep -q "### seed $s exit" $GEN_LOG 2>/dev/null; do sleep 60; done
  grep "### seed $s exit" $GEN_LOG
  run_seed "$s" --augs sd_lora_v3 copypaste
done
echo "### ALL DONE $(date '+%F %T')"
