#!/usr/bin/env bash
# copy-paste 이미지를 K {2,4,8,10} × seed 0–4로 다시 만든다 (11/18, 공정 시험셋 K 곡선용).
# K 1·2·4·8 manifest는 저장소에 있고 같은 코드로 그대로 재현된다(이미지만 이 서버에 없었다). K=10은 새로 만든다.
#   cd Augmentation && setsid nohup bash copypaste/regen_kcurve.sh > copypaste/regen_kcurve.console.log 2>&1 &
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
run() { $PY -m copypaste.generate --k $1 --seed $2 --force --data-root "$DR" > copypaste/regen_K$1_seed$2.log 2>&1; echo "K$1 seed$2 exit $?"; }
for k in 2 4 8 10; do
  for s in 0 1 2 3 4; do run $k $s & done
  wait
done
echo CP_KCURVE_DONE
