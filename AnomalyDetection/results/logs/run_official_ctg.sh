#!/usr/bin/env bash
# 2단계 cTransGAN 추가 (11/20): run_official_gen.sh가 끝나면 seg_aug + ctransgan_official_pinfix만 공식 "불량 조금" 분할로 돌린다.
set -uo pipefail
cd /home/junho/26_2_DeepLearning
until grep -q "ALL DONE" AnomalyDetection/results/logs/run_official_gen.console.log; do sleep 60; done
/home/junho/miniforge3/envs/26_2_dl/bin/python -m AnomalyDetection.ad.official_eval --protocol 2cls_fewshot --pc --models seg_aug \
  --augs ctransgan_official_pinfix --skip-none --tag _ctg > AnomalyDetection/results/logs/official_2cls_ctg.log 2>&1
echo "### ALL DONE $(date '+%F %T')"
