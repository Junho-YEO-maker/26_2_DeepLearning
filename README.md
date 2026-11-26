# 적은 불량 사진을 최대한 활용하는 PCB 이상탐지

2026-2 딥러닝 팀 프로젝트. 신규 라인처럼 **정상 사진은 많고 불량 사진은 결함 종류마다 몇 장(K장)뿐일 때**,
불량 사진을 어떻게 써야 탐지가 나아지는지, 그리고 생성 모델로 만든 합성 불량이 **실제 불량 몇 장의 효과**를 내는지를 잰다.

- 최종 발표자료: [`PCB_AD_presentation_v2.pptx`](PCB_AD_presentation_v2.pptx) (본편 24장 + 부록 14장, 발표 메모 포함)
- 데이터: VisA PCB1–3(공개), AIHub 「SMT 공정 개선 멀티모달 데이터」(실제 생산 라인, 저장소에는 없음)
- 생성 모델: cTransGAN(불량 K장만으로 처음부터 학습), SD1.5 인페인팅 + LoRA
- 탐지 모델: PatchCore(정상만), KNN+MLP · DevNet(사진 단위), 위치 모델(seg, 결함 위치를 정답으로 학습)

## 연구 질문과 답

| | 질문 | 답 (발표자료 v2 기준) |
|---|---|---|
| RQ1 | 불량 라벨은 어떻게 써야 하나? | 사진 단위로 배우면 배경(촬영 조건)을 보고 맞힌다. 결함 위치로 배우면 배경 섞은 시험에서 놓친 불량이 87% → 24%로 준다. 여기에 cTransGAN 합성 40장을 더하면 21.7% → 18.0% |
| RQ2 | 합성은 얼마나 늘려야 하나? 실제 몇 장 효과인가? | 종류마다 20장 안팎이면 충분하다. 합성 40장은 실제 1장일 때 1.4장, 2장일 때 2.7장, 4장일 때 6.6장 효과(cTransGAN, VisA) |
| RQ3 | 현장 데이터(SMT)에서도 같은가? | 정상만 쓰는 PatchCore는 불량의 91%를 놓친다. 합성은 실제 1–4장일 때 실제를 1.2–2.6배 모은 효과이고, 16장쯤 모이면 효과가 없거나 해가 된다 |

현장 권고: 불량 사진에 결함 위치를 상자로 표시하고, 위치 모델을 촬영 조건 섞기와 함께 학습한다.
실제 불량이 종류마다 4장 이하면 합성 20–40장을 더하고, 16장쯤 모이면 뺀다.

## 평가 방법 (요약)

- **놓친 불량 비율(미검률)**: 정상 사진의 5%(SMT는 1%)를 잘못 거르도록 기준을 정했을 때 놓친 불량의 비율. 낮을수록 좋다.
- **배경 섞은 시험셋**(코드와 팀 문서에서는 "공정 시험셋", `AnomalyDetection/ad/fair_ad.py`):
  VisA 불량 사진은 정상과 다른 날 다른 조건으로 찍혀서, 모델이 결함 대신 배경을 보고도 원래 시험셋 점수가 높게 나온다.
  그래서 결함을 지운 불량 사진(정상으로 맞혀야 함)과 정상 사진에 결함을 붙인 사진(불량으로 맞혀야 함)을 시험셋에 더했다.
- **나누기**: 실제 불량 K장을 고르는 무작위 분할. 분할 seed 0–4로 5번 반복하고, PCB 3개와 곱해 15단위로 비교한다.
- **실제 몇 장 효과**: 합성을 더한 성적을 실제 불량만 쓴 K 곡선(K = 1–10)에 놓고 읽는다(`AnomalyDetection/results/logs_jh/README.md`).

## 폴더

```
26_2_DeepLearning/
├─ PCB_AD_presentation_v2.pptx  최종 발표자료 (v1: PCB_AD_presentation.pptx)
├─ Dataset/
│  ├─ visa/
│  │  ├─ raw/                원본 VisA PCB1–3 자리. git 제외 → 받는 법: raw/README.md
│  │  ├─ splits/             분할표 (test, val_normal, train_normal, pool, train_K{1,2,3,4,5,6,8,10}_seed{0..4})
│  │  └─ splits_official/    VisA 공식 분할 (공개 수치 비교용)
│  ├─ generated/             합성 불량. {방법}/K{k}_seed{s}/{images,masks,manifest.csv} → README_grid.md
│  └─ aihub_smt/raw/         AIHub SMT 자리. 원본 · 분할표 모두 git 제외 → 받는 법: raw/README.md
├─ Augmentation/             생성 모델 (Augmentation/README.md)
│  ├─ ctransgan/             cTransGAN (VisA)          smt_ctransgan/  cTransGAN (SMT)
│  ├─ sd_lora/               SD 인페인팅 + LoRA (VisA)  smt_sd_lora/    SD + LoRA (SMT)
│  ├─ pin_cleanup/           휜 핀 잔상 제거             gen_subset.py   합성 N장 부분 세트
│  ├─ copypaste/, anomalydiffusion/   기준선, 시도 후 보류
│  └─ common/kshot.py        공통 규칙 (분할 읽기, 결함 위치, manifest · 누수 검사)
├─ AnomalyDetection/         탐지 모델과 평가 (AnomalyDetection/README.md, 14절이 최종 정리)
│  ├─ ad/                    VisA: 모델, 배경 섞은 시험셋(fair_ad.py), 요약 · 그림 스크립트
│  ├─ smt/                   SMT: 위치 모델(seg_ad.py), 합성 양 · 품질 실험
│  └─ results/               결과 csv와 요약 md (results/ppt/*.md, results/smt/, results/logs_jh/)
├─ ppt_build/                발표자료 만드는 코드 (build_deck_v2.py, deck_v2_appendix.py, deck_lib.py)
├─ ppt_figures/              발표 그림 (ppt_figures/README.md)
├─ scripts/                  VisA 분할표 만들기 (make_splits.py, add_k_splits.py)
└─ colab/                    초기 Colab 노트북
```

## 데이터 받기

| 데이터 | 저장소에 있는 것 | 없는 것과 이유 | 받는 법 |
|---|---|---|---|
| VisA PCB1–3 | 분할표, 합성 세트 manifest, 일부 합성 이미지 | 원본 이미지(약 700MB, 용량) | [`Dataset/visa/raw/README.md`](Dataset/visa/raw/README.md) |
| 합성 불량 (VisA) | K1 나누기 0–4, K2·K4·K8 나누기 0의 이미지 · 마스크, 나머지는 manifest만 | 그 밖의 세트 이미지(세트당 약 100MB) | [`Dataset/generated/README_grid.md`](Dataset/generated/README_grid.md)의 명령으로 다시 생성 |
| AIHub SMT | 결과 표(지표만) | 원본, 합성, 목록 · 분할표, SMT 사진이 든 그림 (AIHub 약관상 제3자 제공 금지) | [`Dataset/aihub_smt/raw/README.md`](Dataset/aihub_smt/raw/README.md) (AIHub 활용 신청 필요) |

## 환경

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # CUDA에 맞게
pip install -r requirements.txt
```

python 3.11, A100 80GB 한 장에서 돌렸다. cTransGAN은 CPU로도 돌지만 느리다.

## 실행

모두 저장소 루트에서 실행한다. VisA 원본을 `Dataset/visa/VisA_official/VisA_PCB`에 받았다고 가정한다.
아래 예시는 이미지가 저장소에 없는 칸(K = 2, 나누기 1)을 다시 만드는 경우다.

```bash
DR=$PWD/Dataset/visa/VisA_official/VisA_PCB   # 절대경로로 넘긴다

# 1) 합성 불량 만들기 → Dataset/generated/{ctransgan_diverse,sd_lora_v3}/K2_seed1/ (+ _n1 · _n5 · _n10 · _n20 부분 세트)
cd Augmentation
#   cTransGAN: 생성 → 휜 핀 잔상 제거 → 부분 세트
python -m ctransgan.generate_diverse --k 2 --seed 1 --diverse-scratch --device cuda --data-root $DR \
    --out ../Dataset/generated/_before_pinfix/ctransgan_diverse
python -m pin_cleanup.clean --manifest ../Dataset/generated/_before_pinfix/ctransgan_diverse/K2_seed1/manifest.csv \
    --out-name ctransgan_diverse --k 2 --seed 1 --data-root $DR
python -m gen_subset --aug ctransgan_diverse --k 2 --seed 1 --n 1 5 10 20
#   SD+LoRA v3: LoRA 학습 → 생성 → 휜 핀 잔상 제거 → 부분 세트
python -m sd_lora.train --k 2 --seed 1 --steps 1000 --p-normal 0.1 --rank 16 --context 512 --runs-dir sd_lora/runs_v3 --data-root $DR
python -m sd_lora.generate --k 2 --seed 1 --runs-dir sd_lora/runs_v3 --tag v3raw --diverse --overgen 2 --context 512 --data-root $DR
python -m pin_cleanup.clean --manifest ../Dataset/generated/sd_lora_v3raw/K2_seed1/manifest.csv \
    --out-name sd_lora_v3 --k 2 --seed 1 --data-root $DR
python -m gen_subset --aug sd_lora_v3 --k 2 --seed 1 --n 1 5 10 20
cd ..

# 2) 배경 섞은 시험셋 만들기(처음 한 번) → 위치 모델 + 촬영 조건 섞기, 실제만 vs 합성 40장 추가
python -m AnomalyDetection.ad.fair_ad --build --data-root $DR
python -m AnomalyDetection.ad.fair_ad --k 2 --seed 1 --pcbs pcb1 --tag _demo --data-root $DR \
    --augs ctransgan_diverse --sets seg_aug:none seg_aug:ctransgan_diverse --seg-seeds 2
#    결과: AnomalyDetection/results/fair_ad_K2_seed1_demo_pcb1.csv
#    (--tag를 빼고 PCB 3개를 다 돌리면 저장소의 결과 csv를 덮어쓴다)

# 3) SMT 위치 모델 (AIHub 원본 · 분할표 · 합성 세트가 있을 때만). 결과는 results/smt/seg_ad_pre.csv에 행이 더해진다
python -m AnomalyDetection.smt.seg_ad --category pre --k 1 --seeds 0 1 2 --conds none sd_lora_fix:40:0.5

# 4) 표 · 그림 · "실제 몇 장 효과" (저장소의 결과 csv만 읽음, 데이터 없이 바로 돌아감)
python -m AnomalyDetection.ad.grid_summary
python AnomalyDetection/results/logs_jh/equiv_visa.py ctg
python AnomalyDetection/results/logs_jh/equiv_smt.py

# 5) 발표자료 (Pretendard 글꼴 필요)
python ppt_build/make_dataset_figures.py    # VisA 예시 사진 (VisA 원본 필요)
python ppt_build/build_deck_v2.py           # → PCB_AD_presentation_v2.pptx
```

SD+LoRA 학습 · 생성은 [`Augmentation/sd_lora/README.md`](Augmentation/sd_lora/README.md),
전체 반복 스크립트는 `AnomalyDetection/results/logs/`(팀 실험)와 `AnomalyDetection/results/logs_jh/`(11/25–26 추가 실험)에 있다.
일부 팀 스크립트에는 실험 서버 경로(`/home/junho/...`)가 적혀 있다. `--data-root`나 `DATA_ROOT` 환경변수로 바꿔 넘기면 된다.

## 결과 파일 찾기

| 무엇 | 어디 |
|---|---|
| 최종 정리와 실험별 요약 파일 위치 | `AnomalyDetection/README.md` 14절 |
| 배경 섞은 시험셋 K 곡선, 모델 비교 | `AnomalyDetection/results/ppt/summary.md` |
| 생성기 × 실제 K × 합성 N | `AnomalyDetection/results/ppt/grid_summary.md` |
| 합성 40장 = 실제 몇 장 (VisA, SMT) | `AnomalyDetection/results/logs_jh/README.md` |
| 공개 방법 비교 (PatchCore, DRA, DevNet) | `AnomalyDetection/results/ppt/public_comparison_summary.md` |
| SMT 합성 양 · 품질 | `AnomalyDetection/results/ppt/gen_amount_summary.md`, `gen_quality_smt_summary.md` |

## 규칙 (누수 방지)

- 경로는 전부 저장소 루트 기준 상대경로다. csv 안의 경로도 같다.
- `test.csv`의 이미지는 학습 · 생성에 절대 쓰지 않는다. `val_normal.csv`도 학습 · 생성에 쓰지 않는다.
- 생성 모델은 해당 `train_K{k}_seed{s}.csv`의 불량만 보고 학습하고, 정상 바탕은 `train_normal.csv`에서만 가져온다.
- `manifest.csv` 열: `image, mask, label, source_image, score, category, defect_type, base_normal_image, gen_seed, condition` (`Augmentation/README.md`).
- 분할표는 이미 만들어 고정했다(분할 seed 7017). `scripts/make_splits.py`를 다시 돌리지 않는다.
