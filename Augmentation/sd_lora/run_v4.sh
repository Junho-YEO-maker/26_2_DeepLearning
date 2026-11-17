#!/usr/bin/env bash
# v4 (11/17): v3 LoRA 그대로, 생성만 바꿈 → 결함 닮음(gen_realism) → AD(label_weight, canon) 확인. SSH가 끊겨도 계속(setsid nohup).
#   - missing: 불량 원본의 실제 빠진 모습으로 먼저 채우고 strength 0.5로 다듬기 (v3 missing 정밀도 0.01)
#   - bent·melt·missing: 바탕에서 비슷한 구조 자리(같은 헤더의 다른 핀, 같은 모양 SMD)로 옮기기 (v3는 원본 1곳에 몰림)
#   cd Augmentation && setsid nohup bash sd_lora/run_v4.sh > sd_lora/run_v4.console.log 2>&1 &
#   SKIP_GEN=1 이면 생성은 건너뛰고 확인만
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
LOGS=/home/junho/26_2_DeepLearning/AnomalyDetection/results/logs

if [ "${SKIP_GEN:-0}" != 1 ]; then
echo "### v4 generate start $(date '+%F %T')"
$PY -m sd_lora.generate --k 1 --seed 0 --runs-dir sd_lora/runs_v3 --tag v4 --diverse --overgen 2 \
    --relocate bent melt missing --erase-missing --data-root "$DR" --force 2>&1 \
    | grep -v -i "warn\|Loading\|fetch\|deprecat\|unsafe\|safety" | tee sd_lora/run_v4_K1_seed0.log
fi
[ -f ../Dataset/generated/sd_lora_v4/K1_seed0/manifest.csv ] || { echo "### v4 generate FAILED $(date '+%F %T')"; exit 1; }

cd ..
echo "### gen_realism start $(date '+%F %T')"
$PY -m AnomalyDetection.ad.gen_realism --augs sd_lora_v3 sd_lora_v4 --data-root "$DR" > $LOGS/gen_realism_v4.log 2>&1 \
  || { echo "### gen_realism FAILED"; tail -20 $LOGS/gen_realism_v4.log; }
grep -v Warning $LOGS/gen_realism_v4.log | tail -8

echo "### label_weight canon start $(date '+%F %T')"
$PY -m AnomalyDetection.ad.label_weight_check --prep canon --aug sd_lora_v4 --diag-aug sd_lora_v3 --data-root "$DR" \
    > $LOGS/label_weight_K1_seed0_canon_v4.log 2>&1 || { echo "### label_weight FAILED"; tail -20 $LOGS/label_weight_K1_seed0_canon_v4.log; exit 1; }
grep -v Warning $LOGS/label_weight_K1_seed0_canon_v4.log | tail -26
echo "### v4 DONE $(date '+%F %T')"
