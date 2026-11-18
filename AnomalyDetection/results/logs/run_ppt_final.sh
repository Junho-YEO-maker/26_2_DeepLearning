#!/usr/bin/env bash
# 발표용 표·그림 최종 집계 예약 (11/18). 공정 K 곡선(run_fair_kcurve.sh)이 다 끝나면 ad/ppt_summary.py를 돌린다.
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_ppt_final.sh \
#       > AnomalyDetection/results/logs/run_ppt_final.console.log 2>&1 &
# 결과: AnomalyDetection/results/ppt/summary.md, fair_kcurve.png, fair_kcurve_pcb.png → ppt_figures/s14_*
set -uo pipefail
cd /home/junho/26_2_DeepLearning
export OMP_NUM_THREADS=4
echo "### wait $(date '+%F %T')"
until grep -q "### ALL DONE" AnomalyDetection/results/logs/run_fair_kcurve.console.log 2>/dev/null; do sleep 120; done
/home/junho/miniforge3/envs/26_2_dl/bin/python -m AnomalyDetection.ad.ppt_summary
rc=$?
bash ppt_figures/collect.sh
echo "### PPT_SUMMARY_DONE exit $rc $(date '+%F %T')"
