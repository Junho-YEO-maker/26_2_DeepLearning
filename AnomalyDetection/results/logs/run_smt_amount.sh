#!/usr/bin/env bash
# 생성량 실험 (11/20, 사용자 승인 "엉"): SMT 사전공정, SD+LoRA 합성을 종류당 160장 만들고
# manifest 앞에서 20 / 40 / 80 / 160장을 잘라 위치 seg를 돌린다. (20 ⊂ 40 ⊂ 80 ⊂ 160, 같은 생성에서 자른 것)
#   생성: Augmentation/smt_sd_lora/generate.py --per-type 160 --overgen 2 --tag n160 (320장 중 박스 안 변화 큰 160장, 기존 40장 규칙과 같은 비율)
#         → Dataset/aihub_smt/generated/sd_lora_n160/K{k}_seed{s}/ (AIHub 파생물이라 git 제외)
#         사전공정은 sd_lora_fix도 sd_lora 행을 그대로 쓰므로 생성 규칙이 같다. LoRA는 이미 학습한 runs/K{k}_seed{s}/pre.
#   칸: K 1·4 × seed 0–2. 생성은 GPU에서 하나씩, 생성이 끝난 칸부터 seg를 뒤에서 돌린다.
#   결과: results/smt/seg_ad_pre.csv에 이어 쓰기 → 끝나면 gen_amount --diversity, --compare
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_smt_amount.sh \
#       > AnomalyDetection/results/logs/run_smt_amount.console.log 2>&1 &
#   멈추기: pkill -s <이 스크립트의 PID>
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
SLOG=AnomalyDetection/results/smt/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
NOISE='warn|deprecat|Loading|fetch|safety|unsafe|LoRA keys'
CONDS=(sd_lora_n160:20:0.5 sd_lora_n160:40:0.5 sd_lora_n160:80:0.5 sd_lora_n160:160:0.5)
echo "### start $(date '+%F %T')"
for k in 1 4; do
  for s in 0 1 2; do
    echo "### generate K$k seed $s $(date '+%F %T')"
    (cd Augmentation && $PY -m smt_sd_lora.generate --k $k --seed $s --category pre --per-type 160 --overgen 2 --tag n160) 2>&1 \
      | grep -v -iE "$NOISE" > Augmentation/smt_sd_lora/run_n160_K${k}_seed${s}.log
    if ! grep -q '누수 검사 PASS' Augmentation/smt_sd_lora/run_n160_K${k}_seed${s}.log; then
      echo "### generate K$k seed $s FAILED $(date '+%F %T')"; continue
    fi
    echo "### generate K$k seed $s done $(date '+%F %T')"
    $PY -m AnomalyDetection.smt.seg_ad --category pre --k $k --seeds $s --conds "${CONDS[@]}" \
      > $SLOG/seg_ad_amount_pre_K${k}_seed${s}.log 2>&1 \
      && echo "### seg K$k seed $s done $(date '+%F %T')" || echo "### seg K$k seed $s FAILED $(date '+%F %T')" &
  done
done
wait
$PY -m AnomalyDetection.smt.gen_amount --diversity > AnomalyDetection/results/logs/gen_amount_diversity.log 2>&1
$PY -m AnomalyDetection.smt.gen_amount --compare > AnomalyDetection/results/logs/gen_amount_compare.log 2>&1
echo "### ALL DONE $(date '+%F %T')"
