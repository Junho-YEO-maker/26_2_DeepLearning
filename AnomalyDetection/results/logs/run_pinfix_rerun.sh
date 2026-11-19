#!/usr/bin/env bash
# VisA: 잔상 제거 버전으로 생성기 × K × 합성 장수 N 반복 (11/18 15시, 사용자 승인한 계획).
#
# 고정: VisA PCB1–3, 공정 시험셋(D·DR·ND), 지표 = 과검 5%에서 미검률(전체·종류별), 기준선 = 실제만(N=0, K 곡선 결과), PatchCore
# 반복: 생성기 {SD+LoRA v3, cTransGAN} × K {1, 2, 4, 8} × N(종류당) {1, 5, 10, 20, 40}, seed 0. SD+LoRA K=1만 seed 1–4 추가
#   1) 생성: 실제 K장으로 학습한 생성기의 종류당 40장 → bent 잔상 제거(Augmentation/pin_cleanup) → 누수 검사
#      cTransGAN K8은 여기서 새로 만든다 (나머지는 이미 있음)
#   2) AD: 실제 K장 + 합성 N장으로 위치 seg(촬영 조건 무작위화, seg_aug) 학습. 배치 안 실제:합성 = 반반 고정, 장수만 바뀜
#      N=40에서는 seg(무작위화 없음)와 이미지 DevNet도 같이 (대조용, 발표 K 곡선과 같은 줄)
#      헤드 시드 2 (K 곡선과 같은 설정)
# 바꿔 넣기: 원래 세트는 Dataset/generated/_before_pinfix/로, 이전 결과는 AnomalyDetection/results/_before_pinfix/로 옮긴다(지우지 않음).
#   고친 세트를 같은 이름(sd_lora_v3, ctransgan_diverse)에 두므로 fair_ad·gradcam·ppt_summary 코드는 그대로다.
#   fair_ad_K1_seed{0..4}.csv는 실제만·copy-paste 줄이 섞여 있어 원본만 보관하고 +sd_lora_v3 줄만 뺀다.
#   copy-paste·anomalydiffusion·sd_lora v2는 부록이라 다시 안 돌린다 (copy-paste에도 잔상 있음 → 한계).
# 결과: AnomalyDetection/results/fair_ad_K{k}_seed{s}_grid_{v3|ctg}.csv (칸 전체)
#       SD+LoRA N=40 줄은 fair_ad_K{k}_seed{s}_v3.csv에도 따로 써서 ppt_summary(K 곡선 그림)가 그대로 읽게 한다
# 끝: 생성 품질(gen_realism), Grad-CAM 다시, 공정 K 곡선이 끝나면 ppt_summary + ppt_figures/collect.sh
#
#   cd /home/junho/26_2_DeepLearning && setsid nohup bash AnomalyDetection/results/logs/run_pinfix_rerun.sh \
#       >> AnomalyDetection/results/logs/run_pinfix_rerun.console.log 2>&1 &
# 끝난 단계는 표시·결과 파일을 보고 건너뛰므로 다시 띄워도 된다.
set -uo pipefail
cd /home/junho/26_2_DeepLearning
PY=/home/junho/miniforge3/envs/26_2_dl/bin/python
DR=${DATA_ROOT:-/home/junho/26_2_DeepLearning/Dataset/visa/VisA_official/VisA_PCB}
LOGS=AnomalyDetection/results/logs
GEN=Dataset/generated
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
COMMON=(--seg-seeds 2 --img-seeds 2 --img-kinds devnet)
NS=(1 5 10 20)
MAXJ=2
stage() { echo "### $* $(date '+%F %T')"; }

# ---------- 1. 잔상 제거 (앞서 띄운 정리가 돌고 있으면 기다렸다가, 빠진 것만 채운다) ----------
if [ ! -f $GEN/_before_pinfix/SWAPPED ]; then
  while pgrep -f "pin_cleanup.clean" > /dev/null; do sleep 30; done
  stage "pinfix (빠진 것만)"
  (cd Augmentation \
    && $PY -m pin_cleanup.clean --aug sd_lora_v3 --k 1 --seed 0 1 2 3 4 --data-root "$DR" \
    && $PY -m pin_cleanup.clean --aug sd_lora_v3 --k 2 4 8 10 --seed 0 --data-root "$DR" \
    && $PY -m pin_cleanup.clean --aug ctransgan_diverse --k 1 2 4 --seed 0 --data-root "$DR") \
    || { stage "pinfix FAILED"; exit 1; }
  for m in sd_lora_v3_pinfix/K{1_seed0,1_seed1,1_seed2,1_seed3,1_seed4,2_seed0,4_seed0,8_seed0,10_seed0} \
           ctransgan_diverse_pinfix/K{1,2,4}_seed0; do
    [ -f $GEN/$m/manifest.csv ] || { stage "pinfix 빠짐: $m"; exit 1; }
  done
  $PY - <<'EOF' || { stage "swap FAILED"; exit 1; }
import hashlib, shutil, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, 'Augmentation')
from common.kshot import check_manifest
G, R = Path('Dataset/generated'), Path('AnomalyDetection/results')
GA, RA = G / '_before_pinfix', R / '_before_pinfix'
GA.mkdir(parents=True, exist_ok=True); (RA / 'fair_scores').mkdir(parents=True, exist_ok=True)
emb, emb_a = Path('Dataset/cache/emb512'), GA / 'emb512_bent'
emb_a.mkdir(exist_ok=True)
for name in ('sd_lora_v3', 'ctransgan_diverse'):
    src, fixed = G / name, G / f'{name}_pinfix'
    shutil.move(str(src), str(GA / name))
    shutil.move(str(fixed), str(src))
    for man in sorted(src.glob('K*_seed*/manifest.csv')):
        d = pd.read_csv(man, dtype=str, keep_default_na=False)
        for c in ('image', 'mask'):
            d[c] = d[c].str.replace(f'{G}/{name}_pinfix/', f'{G}/{name}/', regex=False)
        d['condition'] = d['condition'].str.replace(f'"{G}/{name}/', f'"{G}/_before_pinfix/{name}/', regex=False)
        d.to_csv(man, index=False)
        print(check_manifest(man))
        for img in d.loc[d.defect_type == 'bent', 'image']:      # 옛 run.py 임베딩 캐시(경로 키)에서 바뀐 이미지 빼기
            p = emb / (hashlib.sha1(img.encode()).hexdigest()[:16] + '.npy')
            if p.exists():
                shutil.move(str(p), str(emb_a / p.name))
for f in sorted(R.glob('fair_ad_K*_seed*_v3*.csv')) + sorted(R.glob('fair_ad_K*_seed*_ctg*.csv')):
    shutil.move(str(f), str(RA / f.name)); print('보관', f.name)
for f in sorted((R / 'fair_scores').glob('K*_seed*_v3_*.npz')) + sorted((R / 'fair_scores').glob('K*_seed*_ctg_*.npz')):
    shutil.move(str(f), str(RA / 'fair_scores' / f.name))
for s in range(5):
    f = R / f'fair_ad_K1_seed{s}.csv'
    if f.exists():
        shutil.copy2(f, RA / f.name)
        d = pd.read_csv(f)
        d[d.train != '+sd_lora_v3'].to_csv(f, index=False)
        print(f'{f.name}: +sd_lora_v3 {int((d.train == "+sd_lora_v3").sum())}줄 뺌')
for name in ('gen_count_sweep_sd_lora_v3_K1_seed0.csv', 'gradcam_K1_seed0.csv', 'gen_realism_K1_seed0.csv', 'gradcam'):
    if (R / name).exists():
        shutil.move(str(R / name), str(RA / name)); print('보관', name)
for f in Path('Dataset/cache/defect_removal').glob('*sd_lora_v3*'):
    shutil.move(str(f), str(GA / f.name))
EOF
  touch $GEN/_before_pinfix/SWAPPED
  stage "pinfix swapped"
fi

# ---------- 1b. cTransGAN K8 생성 (GPU, AD와 동시에) → 잔상 제거 ----------
ctg_k8() {
  if [ -f $GEN/ctransgan_diverse/K8_seed0/manifest.csv ]; then return 0; fi
  stage "cTransGAN K8 생성 start"
  (cd Augmentation \
    && { [ -f ../$GEN/_before_pinfix/ctransgan_diverse/K8_seed0/manifest.csv ] \
         || $PY -m ctransgan.generate_diverse --k 8 --seed 0 --diverse-scratch --device cuda --threads 8 \
              --data-root "$DR" --out ../$GEN/_before_pinfix/ctransgan_diverse; } \
    && $PY -m ctransgan.review --k 8 --seed 0 --folder ../$GEN/_before_pinfix/ctransgan_diverse/K8_seed0 --data-root "$DR" \
         --out ctransgan/review/diverse_K8_seed0.png \
    && $PY -m pin_cleanup.clean --manifest ../$GEN/_before_pinfix/ctransgan_diverse/K8_seed0/manifest.csv \
         --out-name ctransgan_diverse --k 8 --seed 0 --data-root "$DR") > $LOGS/ctransgan_K8_seed0.log 2>&1 \
    || { stage "cTransGAN K8 FAILED"; return 1; }
  stage "cTransGAN K8 done"
}
ctg_k8 &
CTG8=$!

# ---------- 2. 장수 부분 세트 ----------
subsets() {   # aug k seed...
  local aug=$1 k=$2; shift 2
  (cd Augmentation && $PY -m gen_subset --aug $aug --k $k --seed "$@" --n "${NS[@]}")
}
for k in 1 2 4 8; do subsets sd_lora_v3 $k 0 || exit 1; done
subsets sd_lora_v3 1 1 2 3 4 || exit 1
for k in 1 2 4; do subsets ctransgan_diverse $k 0 || exit 1; done

# ---------- 3. 생성 품질 (슬라이드 9) ----------
if [ ! -f AnomalyDetection/results/gen_realism_K1_seed0.csv ]; then
  stage "gen_realism start"
  $PY -m AnomalyDetection.ad.gen_realism --augs sd_lora_v3 ctransgan_diverse --k 1 --seed 0 --data-root "$DR" \
    > $LOGS/gen_realism_K1_seed0_pinfix.log 2>&1 || stage "gen_realism FAILED (계속)"
  tail -8 $LOGS/gen_realism_K1_seed0_pinfix.log
fi

# ---------- 4. AD 격자 ----------
merge() {   # k seed tag
  $PY - "$1" "$2" "$3" <<'EOF'
import sys, pandas as pd
from pathlib import Path
k, s, tag = sys.argv[1], sys.argv[2], sys.argv[3]
r = Path('AnomalyDetection/results')
parts = [r / f'fair_ad_K{k}_seed{s}{tag}_{p}.csv' for p in ('pcb1', 'pcb2', 'pcb3')]
miss = [p.name for p in parts if not p.exists()]
if miss:
    print('### 빠진 PCB:', miss); sys.exit()
out = r / f'fair_ad_K{k}_seed{s}{tag}.csv'
d = pd.concat([pd.read_csv(p) for p in parts])
d.to_csv(out, index=False)
for p in parts:
    p.unlink()
if tag == '_grid_v3':                    # ppt_summary가 읽는 이름: N=40(+sd_lora_v3) 줄만
    d[d.train == '+sd_lora_v3'].to_csv(r / f'fair_ad_K{k}_seed{s}_v3.csv', index=False)
g = d.groupby(['model', 'train', 'pcb'])[['fair_fnr5', 'fair_fnr_bent']].mean().groupby(['model', 'train']).mean()
print(f'### 합침 {out.name}\n' + g.round(3).to_string())
EOF
}

run() {   # k seed tag, fair_ad 인자...
  local k=$1 s=$2 tag=$3; shift 3
  [ -f AnomalyDetection/results/fair_ad_K${k}_seed${s}${tag}.csv ] && { echo "### K$k seed $s$tag 이미 있음"; return; }
  echo "### K$k seed $s$tag start $(date '+%F %T')"
  local pids=()
  for p in pcb1 pcb2 pcb3; do
    [ -f AnomalyDetection/results/fair_ad_K${k}_seed${s}${tag}_$p.csv ] && continue
    $PY -m AnomalyDetection.ad.fair_ad --k "$k" --seed "$s" --pcbs $p --tag "$tag" --data-root "$DR" "$@" \
      > $LOGS/fair_ad_K${k}_seed${s}${tag}_$p.log 2>&1 &
    pids+=($!)
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  merge "$k" "$s" "$tag"
  echo "### K$k seed $s$tag done $(date '+%F %T')"
}

cell() {   # 생성기(v3|ctg) k seed: N 1·5·10·20(seg_aug) + N 40(seg_aug, seg, DevNet)
  local g=$1 k=$2 s=$3 aug
  [ $g = v3 ] && aug=sd_lora_v3 || aug=ctransgan_diverse
  local augs=() sets=()
  for n in "${NS[@]}"; do augs+=(${aug}_n$n); sets+=(seg_aug:${aug}_n$n); done
  augs+=($aug); sets+=(seg_aug:$aug seg:$aug img:$aug)
  run $k $s _grid_$g --augs "${augs[@]}" --sets "${sets[@]}" "${COMMON[@]}"
}

slots() {   # 공정 K 곡선이 아직 돌면 GPU 메모리 때문에 1칸, 끝나면 MAXJ칸
  pgrep -f run_fair_kcurve.sh > /dev/null && echo 1 || echo $MAXJ
}

submit() {   # 칸을 slots개까지 동시에
  while [ "$(jobs -rp | grep -v -x "$CTG8" | wc -l)" -ge "$(slots)" ]; do wait -n || sleep 30; done
  cell "$@" &
}

stage "AD 격자 start"
for k in 1 2 4; do
  submit v3 $k 0
  submit ctg $k 0
done
submit v3 8 0
for s in 1 2 3 4; do submit v3 1 $s; done
while kill -0 $CTG8 2>/dev/null; do sleep 60; done           # cTransGAN K8 생성이 끝나야
if [ -f $GEN/ctransgan_diverse/K8_seed0/manifest.csv ]; then
  subsets ctransgan_diverse 8 0 && submit ctg 8 0
fi
wait
stage "AD 격자 done"

# ---------- 5. Grad-CAM, 집계 ----------
if [ ! -f AnomalyDetection/results/gradcam_K1_seed0.csv ]; then
  stage "gradcam start"
  bash $LOGS/run_gradcam.sh > $LOGS/run_gradcam_pinfix.console.log 2>&1 || stage "gradcam FAILED (계속)"
fi
stage "K 곡선 기다림"
until grep -q "### ALL DONE" $LOGS/run_fair_kcurve.console.log 2>/dev/null; do sleep 120; done
$PY -m AnomalyDetection.ad.ppt_summary; rc=$?
bash ppt_figures/collect.sh
stage "ALL DONE (ppt_summary exit $rc)"
