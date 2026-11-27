# 적은 불량 사진을 최대한 활용하는 PCB 이상탐지

2026-2 딥러닝 팀 프로젝트. 신규 라인처럼 **정상 사진은 많고 불량 사진은 결함 종류마다 몇 장(K장)뿐일 때**,
불량 사진을 어떻게 써야 탐지가 나아지는지, 그리고 생성 모델로 만든 합성 불량이 **실제 불량 몇 장의 효과**를 내는지를 잰다.

> **처음 보는 분께**: ① 발표자료 `PCB_AD_presentation_v5.pptx` → ② 아래 [연구 질문과 답](#연구-질문과-답) → ③ 실험별 결과 위치는 [결과 파일 찾기](#결과-파일-찾기)와 `AnomalyDetection/README.md` 14절. 결과 표 · 그림 · 발표자료는 데이터 없이 저장소의 결과 csv만으로 다시 만들 수 있다([실행](#실행) 4–5번). `denoising_SEM/`은 같은 저장소에 있는 별도 과제(SEM 이미지 잡음 제거)로, 자체 README가 있다.

- 최종 발표자료: [`PCB_AD_presentation_v5.pptx`](PCB_AD_presentation_v5.pptx) (본편 21장 + 부록 39장, 발표 메모 포함, `ppt_build/build_deck_v5.py`). 15분에 맞춰 v4 본편 31장을 21장으로 줄이고 말을 쉽게 했다. 본편에서 뺀 10장은 부록 20–29로 옮겼다. 부록 15–19는 SMT 사진 단위 모델 비교와 모델이 본 곳, 부록 30–39는 데이터 설명과 결함 유형별 정상 · 불량 예시. SMT 사진이 들어간 발표용은 `PCB_AD_presentation_v5_local.pptx`(AIHub 약관상 git 제외, `ppt_build/make_smt_figures.py` 후 빌드). v4(본편 31장) · v3 · v2는 이전 판
- 데이터: VisA PCB1–3(공개), AIHub 「SMT 공정 개선 멀티모달 데이터」(실제 생산 라인, 원본 이미지는 저장소에 없음)
- 생성 모델: cTransGAN(불량 K장만으로 처음부터 학습), SD1.5 인페인팅 + LoRA
- 탐지 모델: PatchCore(정상만), KNN+MLP · DevNet(사진 단위), 위치 모델(seg, 결함 위치를 정답으로 학습)

## 연구 질문과 답

| | 질문 | 답 (발표자료 v5 기준) |
|---|---|---|
| RQ1 | 불량 라벨은 어떻게 써야 하나? | 사진 단위로 배우면 배경(촬영 조건)을 보고 맞힌다. 결함 위치로 배우면 결함 바꾼 시험에서 놓친 불량이 87% → 24%로 준다. 실제 1장으로는 정상만 쓰는 PatchCore와 비슷하고, 3장부터 확실히 낫다 |
| RQ2 | 합성은 얼마나 늘려야 하나? 실제 몇 장 효과인가? | 종류마다 20장 안팎이면 충분하다. 합성 40장은 실제 1장일 때 1.4장, 2장일 때 2.7장, 4장일 때 6.6장 효과(cTransGAN, VisA) |
| RQ3 | 현장 데이터(SMT)에서도 같은가? | 정상만 쓰는 PatchCore는 불량의 91%를 놓치고, 위치 모델은 불량 1장으로 20%. 합성은 실제 1–4장일 때 실제를 1.2–2.6배 모은 효과. 실제 16장에서는 사전 공정은 계속 좋아지고 납땜 공정은 나빠지는데, 합성 비율을 줄이면 SD+LoRA는 해가 거의 사라지고 cTransGAN(결함을 과장한 세트)은 여전히 나쁘다. 결함이 사진의 20–50%로 커서 사진 단위 모델도 점수는 비슷하지만, 결함을 지운 사진과 모델이 본 곳을 보면 결함을 보는 쪽은 위치 모델이다 |

현장 권고: 불량 사진에 결함 위치를 상자로 표시하고, 위치 모델을 밝기·색 바꾸기와 함께 학습한다.
실제 불량이 종류마다 4장 이하면 합성 20–40장을 더하고, 실제가 많아지면 합성 비중을 줄이고 결함을 과장한 합성 세트는 뺀다.

## 평가 방법 (요약)

- **놓친 불량 비율(미검률)**: 정상 사진의 5%(SMT는 1%)를 잘못 거르도록 기준을 정했을 때 놓친 불량의 비율. 낮을수록 좋다.
- **결함 바꾼 시험셋**(v2 발표자료의 "배경 섞은 시험셋", 코드와 팀 문서에서는 "공정 시험셋", `AnomalyDetection/ad/fair_ad.py`):
  VisA 불량 사진은 정상과 다른 날 다른 조건으로 찍혀서, 모델이 결함 대신 배경을 보고도 원래 시험셋 점수가 높게 나온다.
  그래서 결함을 지운 불량 사진(정상으로 맞혀야 함)과 정상 사진에 결함을 붙인 사진(불량으로 맞혀야 함)을 시험셋에 더했다.
- **나누기**: 실제 불량 K장을 고르는 무작위 분할. 분할 seed 0–4로 5번 반복하고, PCB 3개와 곱해 15단위로 비교한다.
- **실제 몇 장 효과**: 합성을 더한 성적을 실제 불량만 쓴 K 곡선(K = 1–10)에 놓고 읽는다(`AnomalyDetection/results/logs_jh/README.md`).

## 폴더

```
26_2_DeepLearning/
├─ PCB_AD_presentation_v5.pptx  최종 발표자료 (이전 판: v4 · v3 · v2, v1 = PCB_AD_presentation.pptx)
├─ Dataset/
│  ├─ visa/
│  │  ├─ raw/                원본 VisA PCB1–3 자리. git 제외 → 받는 법: raw/README.md
│  │  ├─ splits/             분할표 (test, val_normal, train_normal, pool, train_K{1,2,3,4,5,6,8,10}_seed{0..4})
│  │  └─ splits_official/    VisA 공식 분할 (공개 수치 비교용)
│  ├─ generated/             합성 불량. {방법}/K{k}_seed{s}/{images,masks,manifest.csv} → README_grid.md
│  └─ aihub_smt/             AIHub SMT 목록(index.csv)과 분할표(splits/). 원본(raw/)은 git 제외 → 받는 법: raw/README.md
├─ Augmentation/             생성 모델 (Augmentation/README.md)
│  ├─ ctransgan/             cTransGAN (VisA)          smt_ctransgan/  cTransGAN (SMT)
│  ├─ sd_lora/               SD 인페인팅 + LoRA (VisA)  smt_sd_lora/    SD + LoRA (SMT)
│  ├─ pin_cleanup/           휜 핀 잔상 제거             gen_subset.py   합성 N장 부분 세트
│  ├─ copypaste/, anomalydiffusion/   기준선, 시도 후 보류
│  └─ common/kshot.py        공통 규칙 (분할 읽기, 결함 위치, manifest · 누수 검사)
├─ AnomalyDetection/         탐지 모델과 평가 (AnomalyDetection/README.md, 14절이 최종 정리)
│  ├─ ad/                    VisA: 모델, 배경 섞은 시험셋(fair_ad.py), 요약 · 그림 스크립트
│  ├─ smt/                   SMT: 위치 모델(seg_ad.py), 사진 단위 모델 비교(img_ad.py), 모델이 본 곳(cam.py), 합성 양 · 품질 실험
│  └─ results/               결과 csv와 요약 md (results/ppt/*.md, results/smt/, results/logs_jh/)
├─ ppt_build/                발표자료 만드는 코드 (build_deck_v5.py, deck_v4_appendix.py, deck_lib.py. 이전 판은 build_deck_v4.py · v3.py · v2.py)
├─ ppt_figures/              발표 그림 (ppt_figures/README.md)
├─ scripts/                  VisA 분할표 만들기 (make_splits.py, add_k_splits.py)
├─ colab/                    초기 Colab 노트북
└─ denoising_SEM/            별도 과제: SEM 이미지 잡음 제거 (denoising_SEM/README.md). 이 프로젝트와 코드를 나누지 않는다
```

## 데이터 받기

| 데이터 | 저장소에 있는 것 | 없는 것과 이유 | 받는 법 |
|---|---|---|---|
| VisA PCB1–3 | 분할표, 합성 세트 manifest, 일부 합성 이미지 | 원본 이미지(약 700MB, 용량) | [`Dataset/visa/raw/README.md`](Dataset/visa/raw/README.md) |
| 합성 불량 (VisA) | K1 나누기 0–4, K2·K4·K8 나누기 0의 이미지 · 마스크, 나머지는 manifest만 | 그 밖의 세트 이미지(세트당 약 100MB) | [`Dataset/generated/README_grid.md`](Dataset/generated/README_grid.md)의 명령으로 다시 생성 |
| AIHub SMT | 목록(`index.csv`), 분할표(`splits/`), 결과 표 | 원본, 합성, SMT 사진이 든 그림 (AIHub 약관상 제3자 제공 금지) | [`Dataset/aihub_smt/raw/README.md`](Dataset/aihub_smt/raw/README.md) (AIHub 활용 신청 필요) |

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

# 3) SMT 위치 모델 (AIHub 원본과 합성 세트가 있을 때만). 결과는 results/smt/seg_ad_pre.csv에 행이 더해진다
python -m AnomalyDetection.smt.seg_ad --category pre --k 1 --seeds 0 1 2 --conds none sd_lora_fix:40:0.5
#    같은 데이터로 사진 단위 DevNet · MLP (결과: results/smt/img_ad_pre.csv), 모델이 본 곳 (results/smt/cam_K16_seed0_pre.csv)
python -m AnomalyDetection.smt.img_ad --category pre --ks 1 4 16 --seeds 0 1 2 3 4
python -m AnomalyDetection.smt.cam --category pre --k 16 --seed 0
#    VisA 모델이 본 곳 (PCB마다 따로 돌리고 합침, 결과: results/gradcam_K1_seed0_r2.csv)
python -m AnomalyDetection.ad.gradcam --pcbs pcb1 --tag _r2 --data-root $DR    # pcb2 · pcb3도
python -m AnomalyDetection.ad.gradcam --summarize --tag _r2

# 4) 표 · 그림 · "실제 몇 장 효과" (저장소의 결과 csv만 읽음, 데이터 없이 바로 돌아감)
python -m AnomalyDetection.ad.grid_summary
python AnomalyDetection/results/logs_jh/equiv_visa.py ctg
python AnomalyDetection/results/logs_jh/equiv_smt.py
python AnomalyDetection/results/logs_jh/cmp_img_seg_smt.py   # SMT 사진 단위 vs 위치 모델

# 5) 발표자료 (Pretendard 글꼴 필요)
python ppt_build/make_dataset_figures.py    # VisA 예시 사진 (VisA 원본 필요)
python ppt_build/make_visa_cam_panels.py   # VisA 모델이 본 곳 확대 그림 (ad/gradcam.py --tag _r2 결과 필요)
python ppt_build/make_smt_figures.py       # SMT 사진 그림, git 제외 (AIHub 원본 필요, 없으면 사진 자리만)
python ppt_build/build_deck_v5.py           # → PCB_AD_presentation_v5.pptx
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
| SMT 사진 단위 모델 vs 위치 모델, SMT 모델이 본 곳 | `AnomalyDetection/results/logs_jh/README.md` (표), `results/smt/img_ad_*.csv`, `results/smt/cam_*.csv` |
| VisA 모델이 본 곳 (Grad-CAM vs 위치 모델 지도) | `AnomalyDetection/results/gradcam_K1_seed0_r2.csv`, `results/gradcam/K1_seed0_r2_all.jpg` |

## 규칙 (누수 방지)

- 경로는 전부 저장소 루트 기준 상대경로다. csv 안의 경로도 같다.
- `test.csv`의 이미지는 학습 · 생성에 절대 쓰지 않는다. `val_normal.csv`도 학습 · 생성에 쓰지 않는다.
- 생성 모델은 해당 `train_K{k}_seed{s}.csv`의 불량만 보고 학습하고, 정상 바탕은 `train_normal.csv`에서만 가져온다.
- `manifest.csv` 열: `image, mask, label, source_image, score, category, defect_type, base_normal_image, gen_seed, condition` (`Augmentation/README.md`).
- 분할표는 이미 만들어 고정했다(분할 seed 7017). `scripts/make_splits.py`를 다시 돌리지 않는다.
