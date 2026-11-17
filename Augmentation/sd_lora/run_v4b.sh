#!/usr/bin/env bash
# v4b (11/17): v3 + v4 missing만 (지우기 방식 missing의 효과만 따로 보기). 새로 생성하지 않고 manifest만 합친다.
#   먼저 sd_lora_v4가 있어야 한다(run_v4.sh로 생성. 11/17 결과 확인 뒤 삭제함).
#   bent·melt·scratch = sd_lora_v3 그대로 (v4에서 자리 옮기기를 끄면 v3와 같은 결과), missing = sd_lora_v4 (지우기 + 같은 헤더 안 옮기기)
#   cd Augmentation && setsid nohup bash sd_lora/run_v4b.sh > sd_lora/run_v4b.console.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
LOGS=/home/junho/26_2_DeepLearning/AnomalyDetection/results/logs

echo "### v4b manifest $(date '+%F %T')"
$PY - <<'EOF' || exit 1
import csv, collections
from pathlib import Path
g = Path('Dataset/generated')
rows = lambda m: list(csv.DictReader(open(g / m / 'K1_seed0/manifest.csv')))
v3, v4 = rows('sd_lora_v3'), rows('sd_lora_v4')
out = [r for r in v3 if r['defect_type'] != 'missing'] + [r for r in v4 if r['defect_type'] == 'missing']
d = g / 'sd_lora_v4b/K1_seed0'
d.mkdir(parents=True, exist_ok=True)
with open(d / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(v3[0]))
    w.writeheader()
    w.writerows(out)
(d / 'README.txt').write_text('v4b: 이미지는 복사하지 않고 sd_lora_v3(bent/melt/scratch)·sd_lora_v4(missing) 경로를 그대로 가리킨다. 만든 곳: Augmentation/sd_lora/run_v4b.sh\n')
print(len(out), collections.Counter((r['category'], r['defect_type'], r['image'].split('/')[2]) for r in out))
EOF

echo "### label_weight canon start $(date '+%F %T')"
$PY -m AnomalyDetection.ad.label_weight_check --prep canon --aug sd_lora_v4b --diag-aug sd_lora_v3 --data-root "$DR" \
    > $LOGS/label_weight_K1_seed0_canon_v4b.log 2>&1 || { echo "### label_weight FAILED"; tail -20 $LOGS/label_weight_K1_seed0_canon_v4b.log; exit 1; }
grep -v Warning $LOGS/label_weight_K1_seed0_canon_v4b.log | tail -26
echo "### v4b DONE $(date '+%F %T')"
