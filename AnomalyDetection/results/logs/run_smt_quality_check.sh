#!/usr/bin/env bash
# 생성 판정 검증 (11/19, 사용자 승인 "엉"): SMT 합성 40장 중 마스크 안이 덜 튀는 20장(qlow) / 더 튀는 20장(qhigh)으로
# 위치 seg를 다시 돌려, 품질과 무관한 앞 20장(이미 있음)·전부 40장·실제만과 비교한다.
#   세트: AnomalyDetection/smt/quality_subset.py (Dataset/aihub_smt/generated/{sd_lora_fix,ctransgan}_{qlow,qhigh})
#   칸: 공정 2 × K 1·4·16 × seed 0–2 = 18개 프로세스, 각 프로세스가 조건 4개(생성기 2 × qlow·qhigh). 동시에 4개.
#   결과: results/smt/seg_ad_{pre,solder}.csv에 이어 쓰기 → 끝나면 quality_subset --compare
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_smt_quality_check.sh \
#       > AnomalyDetection/results/logs/run_smt_quality_check.console.log 2>&1 &
#   멈추기: pkill -s <이 스크립트의 PID>
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
SLOG=AnomalyDetection/results/smt/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
CONDS=(sd_lora_fix_qlow:20:0.5 sd_lora_fix_qhigh:20:0.5 ctransgan_qlow:20:0.5 ctransgan_qhigh:20:0.5)
echo "### start $(date '+%F %T')"
for k in 1 4 16; do
  for s in 0 1 2; do
    for cat in pre solder; do
      while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done
      $PY -m AnomalyDetection.smt.seg_ad --category $cat --k $k --seeds $s --conds "${CONDS[@]}" \
        > $SLOG/seg_ad_quality_${cat}_K${k}_seed${s}.log 2>&1 \
        && echo "### $cat K$k seed $s done $(date '+%F %T')" || echo "### $cat K$k seed $s FAILED $(date '+%F %T')" &
    done
  done
done
wait
$PY -m AnomalyDetection.smt.quality_subset --compare > AnomalyDetection/results/logs/quality_subset_compare.log 2>&1
echo "### ALL DONE $(date '+%F %T')"
