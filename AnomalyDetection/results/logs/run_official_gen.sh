#!/usr/bin/env bash
# 공개 수치 비교 2단계용 가짜 불량 (11/20, 사용자 승인 "3단계까지 다"): VisA 공식 "불량 조금" 분할의 학습 불량 20장/PCB만으로
# 생성기를 새로 학습해 만든다. 기존 생성(우리 분할)은 공식 시험 불량과 겹칠 수 있어서 쓰지 않는다.
#   SD+LoRA v3 레시피(run_all.sh 기본값: LoRA 1000 step, rank 16, 정상 10%, --diverse --overgen 2, context 512)
#   cTransGAN(generate_diverse --diverse-scratch, 팀원 설정 그대로)
#   둘 다 bent 핀 잔상 제거(pin_cleanup) → Dataset/generated/{sd_lora,ctransgan}_official_pinfix/K20_seed0
# 분할표 바꿔 끼우기: Augmentation/official_run.py (공유 코드 수정 없음)
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_official_gen.sh \
#       > AnomalyDetection/results/logs/run_official_gen.console.log 2>&1 &
set -uo pipefail
cd /home/junho/26_2_DeepLearning/Augmentation
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB
SPL=/home/junho/26_2_DeepLearning/Dataset/visa/splits_official/2cls_fewshot
GEN=/home/junho/26_2_DeepLearning/Dataset/generated
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OFF="$PY official_run.py 2cls_fewshot"
echo "### start $(date '+%F %T')"
{
  $OFF sd_lora.train --k 20 --seed 0 --steps 1000 --p-normal 0.1 --rank 16 --context 512 --runs-dir sd_lora/runs_official --data-root $DR \
  && $OFF sd_lora.generate --k 20 --seed 0 --runs-dir sd_lora/runs_official --tag official --diverse --overgen 2 --context 512 --data-root $DR \
  && $OFF pin_cleanup.clean --aug sd_lora_official --k 20 --seed 0 --splits $SPL --data-root $DR
} > sd_lora/run_official.log 2>&1 && echo "### sd_lora done $(date '+%F %T')" || echo "### sd_lora FAILED $(date '+%F %T')"
{
  $OFF ctransgan.generate_diverse --k 8 --seed 0 --diverse-scratch --device cuda --threads 8 --data-root $DR --out $GEN/ctransgan_official \
  && $OFF pin_cleanup.clean --aug ctransgan_official --k 20 --seed 0 --splits $SPL --data-root $DR
} > ctransgan/run_official.log 2>&1 && echo "### ctransgan done $(date '+%F %T')" || echo "### ctransgan FAILED $(date '+%F %T')"
echo "### ALL DONE $(date '+%F %T')"
