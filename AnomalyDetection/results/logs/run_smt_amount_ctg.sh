#!/usr/bin/env bash
# 생성량 실험 cTransGAN판 (11/20, 사용자 요청 "c-transgan으로도 예약해놔"): run_smt_amount.sh(SD+LoRA)가 끝나면 시작한다.
# 팀원이 학습해 둔 사전공정 체크포인트(Dataset/aihub_smt/generated/ctransgan/K{k}_seed{s}/pre_model.pt)로 종류당 160장을 새로 뽑고
# (Augmentation/smt_ctransgan/generate_n.py, 학습 없음), 앞 20 / 40 / 80 / 160장으로 위치 seg를 돌린다.
#   칸: K 1·4 × seed 0–2. 생성은 칸당 1–2분, seg는 동시에 3개.
#   결과: results/smt/seg_ad_pre.csv에 이어 쓰기 → gen_amount --diversity, --compare (SD+LoRA와 한 표)
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_smt_amount_ctg.sh <SD 실행 PID> \
#       > AnomalyDetection/results/logs/run_smt_amount_ctg.console.log 2>&1 &
#   멈추기: pkill -s <이 스크립트의 PID>
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
SLOG=AnomalyDetection/results/smt/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
WAIT_PID=${1:-}
CONDS=(ctransgan_n160:20:0.5 ctransgan_n160:40:0.5 ctransgan_n160:80:0.5 ctransgan_n160:160:0.5)
echo "### queued $(date '+%F %T'), SD+LoRA 실행(PID ${WAIT_PID:-없음}) 끝나기를 기다림"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
echo "### start $(date '+%F %T')"
for k in 1 4; do
  for s in 0 1 2; do
    log=Augmentation/smt_ctransgan/run_n160_K${k}_seed${s}.log
    (cd Augmentation && $PY -m smt_ctransgan.generate_n --k $k --seed $s --per-type 160 --tag n160) > $log 2>&1
    if ! grep -q '누수 검사 PASS' $log; then
      echo "### generate K$k seed $s FAILED $(date '+%F %T')"; continue
    fi
    echo "### generate K$k seed $s done $(date '+%F %T')"
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
    $PY -m AnomalyDetection.smt.seg_ad --category pre --k $k --seeds $s --conds "${CONDS[@]}" \
      > $SLOG/seg_ad_amount_ctg_pre_K${k}_seed${s}.log 2>&1 \
      && echo "### seg K$k seed $s done $(date '+%F %T')" || echo "### seg K$k seed $s FAILED $(date '+%F %T')" &
  done
done
wait
$PY -m AnomalyDetection.smt.gen_amount --diversity > AnomalyDetection/results/logs/gen_amount_diversity.log 2>&1
$PY -m AnomalyDetection.smt.gen_amount --compare > AnomalyDetection/results/logs/gen_amount_compare.log 2>&1
echo "### ALL DONE $(date '+%F %T')"
