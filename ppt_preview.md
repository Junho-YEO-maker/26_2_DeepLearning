# PPT 미리보기 (15분 발표)

작성 2026-11-18. 실험 결과가 나오는 대로 수치와 상태를 바꿔 쓴다.
상태 표기: **[있음]** 바로 쓸 수 있음 / **[진행 중]** 지금 서버에서 돌고 있음 / **[필요]** 아직 안 함

수치는 모두 미검률(과검률 1%에서 놓친 불량 비율, 낮을수록 좋음)이다. 공정 시험셋은 과검률 5% 기준이다.
VisA 숫자는 따로 적지 않으면 K=1, seed 0, PCB1–3 평균이다.

---

## 한 줄 메시지

> 이상 레이블이 적고 불량 사진의 촬영 환경이 다를 때, 레이블은 **위치**까지 쓰고, 딥러닝 생성으로 **늘리고**, 촬영 **환경과 떼어 놓아야** 제대로 활용된다.

## 레이블 활용 3단계 (발표 전체의 뼈대)

| 단계 | 하는 일 | 구현 |
|---|---|---|
| ① 위치 | 이미지 라벨 대신 결함 위치로 학습 | 위치 seg 모델 |
| ② 개수 | K장으로 학습한 생성기로 레이블을 늘림. 생성 결과에 마스크가 같이 나와 위치 라벨도 공짜로 생김 | cTransGAN, SD+LoRA |
| ③ 환경 분리 | 레이블이 촬영 환경과 엮이지 않게 함 | 촬영 조건 무작위화(capture_aug), 공정 시험셋 |

## 모델과 생성기 (발표 범위)

- Aug: **cTransGAN, SD+LoRA** 2종. copy-paste, anomalydiffusion은 부록에만.
- AD: **PatchCore, KNN+MLP, 이미지 DevNet, 위치 seg** 4종. 백본은 모두 같은 고정 WRN50(ImageNet)이라 차이는 레이블 사용법뿐.
- VLM, MVTec, 전처리 변형은 넣지 않음.

---

## 슬라이드별

### 도입

#### 1. 표지 (0:20) [있음]
- 제목 후보: "적은 이상 레이블을 최대한 활용하는 PCB 이상탐지"
- 질문 한 줄: 불량이 몇 장뿐이고 그마저 다르게 찍혔다면, 어떻게 써야 하나?

#### 2. 문제 정의 (0:50) [있음]
- 메시지: 이상탐지에서 적은 비정상 레이블을 어떻게 최대한 활용할까.
- 내용
  - 현장에는 정상 사진이 많고 불량은 종류당 몇 장뿐이다.
  - 표준 방식은 정상만으로 학습한다. 불량 몇 장은 버려지거나 임계값을 정하는 데만 쓰인다.
  - 직관: 불량을 학습에 넣고, 적으니 늘려서 넣으면 더 좋아지지 않을까?

#### 3. 데이터 특성 (1:00) [있음]
- 메시지: 불량은 적고, 그 적은 불량마저 정상과 다른 경로로 모인다.
- ① 레이블이 적다
  - 종류당 최대 10장(PCB2 bent는 8장), K ∈ {1, 2, 4, 8}
  - 현장에서도 피하기 어렵다: 수율이 높으면 불량이 드물고, 새 제품은 이력이 없고, 새 불량 유형은 처음엔 0장이다.
- ② 촬영 환경이 다르다
  - VisA 논문(4절): "The defects were manually generated to produce realistic anomalies."
  - PCB2 불량 사진 평균 밝기 102 vs 정상 88. 불량의 약 2/3가 정상이 한 장도 없는 밝기 구간(103–113)에 모여 있어, 따로 찍은 묶음으로 보인다.
  - PCB1: 정상 표본 40장 중 72%는 기판이 180° 돌아가 있는데 불량 40장은 0%.
  - 결함과 무관한 밝기·위치 통계 8개만으로 정상과 시험 불량을 AUROC 0.93 / 0.85 / 0.76으로 구분. 정상끼리는 0.40–0.53.
  - SMT(현장 수집)도 같다: 같은 기판 자리를 전후로 찍은 62쌍에서 불량 쪽이 더 밝고 차갑게 찍힘. 납땜은 파일명 코드만으로 AUROC 0.92.
- 출처: `Augmentation/figures/capture_difference.png`, `AnomalyDetection/README.md` 13.5절, `AnomalyDetection/ad/prep.py` 설명, `scripts/aihub_smt/README.md`

#### 4. 기존 연구의 활용 방식 (0:40) [있음]
- 메시지: 주류는 정상만 쓰고, 불량을 쓰는 연구도 촬영 환경 차이는 따지지 않았다.
- 내용
  - 정상만: PatchCore 등. VisA 논문의 기본 평가 방식도 정상만 학습(1-class).
  - 불량 사용: DevNet, DRA, AHL(이미지 라벨), PRN(마스크까지). 표준 시험셋 점수로 비교.
  - Baitieva et al.(CVPR 2024): VisA에서 지도 학습 점수가 유난히 높다고 지적 → 우리 결과는 그 이유로 촬영 환경 지름길이라는 다른 설명을 준다.
- 주의: "기존 연구는 불량 레이블을 안 썼다"고 말하지 않는다.

#### 5. 접근 한눈에 (0:50) [필요: 그림]
- 메시지: 레이블 활용 3단계(위치 / 개수 / 환경 분리).
- 그림: K장 → 생성기(Aug) → 생성 불량 + 마스크 → AD 4종 → 표준 시험셋 + 공정 시험셋
- 딥러닝이 쓰인 곳 표시: Transformer GAN(처음부터 학습), 확산 모델 + LoRA(사전학습 + 미세조정), 사전학습 CNN 특징(전이학습), DevNet(deviation loss), 분할 디코더(focal + dice)

### Part 1. Aug: 레이블 개수 늘리기

#### 6. 시행착오 (0:50) [있음]

| 시도 | 결과 | 바꾼 것 |
|---|---|---|
| cTransGAN에 적대 손실만 사용 | 무지개색 노이즈 (32×32, 4000 업데이트) | 결함마다 잠재 벡터 고정 + 재구성 손실 |
| 결함 마스크 안만 다시 그림 (두 생성기 v1) | 원래 핀이 남아 "다리가 하나 더" | 원래 핀 자리까지 다시 그림(`change_region`) |
| cTransGAN v2 | 흐림(선명도가 실제의 약 27%), 지운 자리에 회색 띠 | 256×256을 직접 그림, ±6px 정밀 정렬, 밝기 보정 |
| SD+LoRA v1 | 흰 얼룩 남발, K장 암기 | 정상 기판을 정답으로 주는 정상 샘플 추가(25%) |
| SD+LoRA v2 | 결함을 약하게 그림, 빈 기판 같은 샘플 | 정상 샘플 25% → 10%, 1000 step, rank 16, 2배 만들어 변화 큰 것만 고르기 |
| SD+LoRA v4 (missing 지우기, 자리 옮기기) | 닮음 지표는 오름, AD는 그대로이거나 나빠짐 | v3로 확정 |

- 그림: `ppt_figures/s06_sd_lora_v2_pcb{1,2,3}_sheet.jpg` vs `s08_sd_lora_v3_pcb{1,2,3}_sheet.jpg`, `ppt_figures/s06_sd_lora_v4_base_vs_generated.jpg`, bent 핀 잔상 제거 전후 `ppt_figures/s06_pinfix_before_after.jpg`

#### 7. cTransGAN (1:00) [있음: 구조 / 수치는 채울 실험 0]
- 메시지: K장만으로 처음부터 학습하는 Transformer GAN.
- 원 논문: Wang, C., Huang, G., Huang, Z., & He, W. (2023). [Conditional TransGAN-Based Data Augmentation for PCB Electronic Component Inspection](https://doi.org/10.1155/2023/2024237). *Computational Intelligence and Neuroscience*, Article 2024237.
- VisA PCB 생성 결과 (A100, seed 0): K1·K2·K4 각각 이미지·마스크 480쌍, 총 1,440쌍. 세 조건 모두 manifest 누수 검사 통과.
- 스크래치의 길이·폭·각도 변형은 조건별 120장 중 K1 109장, K2 108장, K4 103장에 적용했다.
- 구조
  - 생성기: 잡음(128) + 조건 → 8×8 → 16×16 → 32×32 토큰(pixel shuffle) → 토큰마다 픽셀 패치로 펼쳐 256×256
  - 판별기: 조건 임베딩을 추가 토큰으로 받음. 생성 결과를 정상 바탕에 끼워 넣은 것을 봄
  - 조건: Transformer 분류기를 K장 결함 크롭으로 학습해 얻은 종류별 평균 특징
  - 손실: 적대 ×0.25 + 다시 그릴 영역 L1 ×35 + 경계 + 고주파 + 크롭 전체 L1
- 데이터 맞춤 조정: 잠재 벡터 고정과 재구성 손실(K장 대응), 원래 핀 자리까지 다시 그림, 기판 정렬 후 같은 부품 자리에 둠, 원본 해상도·JPEG q75로 붙임
- 학습 시간: PCB당 1–4분(GPU, 새 버전)
- 그림: `ppt_figures/s07_ctransgan_K{1,2,4,8}_review.png` (잔상 제거 후)

#### 8. SD+LoRA (1:00) [있음]
- 메시지: 사전학습 확산 모델에 LoRA만 학습해 K장으로 결함을 그린다.
- 구조: SD1.5 인페인팅, UNet에 LoRA(rank 16), PCB별 1개, 결함 종류는 프롬프트로 구분
- 학습: K장 결함 주변 512 크롭, 인페인팅 영역 가림, 정상 샘플 10%(결함 단어 없는 프롬프트), 1000 step
- 생성: 결함 모양·위치 변형(`--diverse`), 2배 만들어 마스크 안 변화가 큰 40장만 남김
- 데이터 맞춤 조정: 정상 사진 위에 그리고 JPEG까지 맞춰 생성 흔적이 지름길이 되지 않게 함
- 그림: `ppt_figures/s08_sd_lora_v3_pcb{1,2,3}_sheet.jpg` (잔상 제거 후), `ppt_figures/s08_real_vs_v3_melt_missing.png`

#### 9. Aug 결과: 생성 품질 (0:40) [진행 중: 채울 실험 0에서 잔상 제거 후 다시 잼]
- 메시지: 두 생성기가 실제 결함을 얼마나 닮았나.
- 실제 결함을 닮은 정도(정밀도 / 재현율, PCB 평균, `AnomalyDetection/results/gen_realism_K1_seed0.csv`)

| | bent | melt | scratch | missing |
|---|---|---|---|---|
| 실제끼리 (상한) | .68 / .68 | .32 / .32 | .30 / .30 | .34 / .34 |
| SD+LoRA v3 (잔상 제거 전) | .37 / .22 | .13 / .15 | .36 / .30 | .01 / .11 |
| cTransGAN (잔상 제거 전) | .33 / .22 | .13 / .15 | .36 / .19 | .02 / .11 |

- 비교 축: 학습량(처음부터 vs LoRA만), 학습 시간, 다양성, 닮음

### Part 2. AD: 레이블을 학습에 쓰는 방식

#### 10. AD 모델 4개 (0:50) [필요: 그림]

| 모델 | 불량 레이블 사용 | 구조 | 역할 |
|---|---|---|---|
| PatchCore | 안 씀 | 정상 패치 특징을 메모리에 저장(1%만 추림), 점수 = 가장 가까운 정상 패치까지 거리 | 기존 방식, 기준선 |
| KNN+MLP | K=0이면 kNN, K>0이면 MLP(이미지 라벨) | 이미지 임베딩(패치 특징의 평균+최대, 2048차원) + kNN 또는 MLP | 같은 특징 위에서 레이블만 넣었다 뺐다 |
| 이미지 DevNet | 이미지 라벨 | 같은 이미지 임베딩 + MLP 헤드, deviation loss | 레이블을 잘 쓰는 이미지 단위 모델 |
| 위치 seg | 위치 라벨(마스크·박스) | WRN50 layer1–3 고정 + 작은 FPN 디코더 → 128×128 결함 지도, focal + dice. 이미지 점수 = 지도 상위 10칸 평균 | 데이터 특성에 맞춘 최종 모델 |

- seg도 점수 하나를 내고, 정상 사진으로 과검률 1% 임계값을 정해 판정한다. 그래서 네 모델을 같은 지표로 비교할 수 있다.

#### 11. 처음 시도: 이미지 단위 (0:50) [있음]
- 메시지: 실제 불량은 효과가 있어 보였는데, 생성을 넣으니 나빠졌다.
- 실제 불량만 (VisA, 3 PCB × 5 seed, `AnomalyDetection/results/results.csv`)
  - PatchCore 0.153
  - DevNet K=1 0.341 → K=8 **0.093** (PatchCore보다 유의미하게 좋음)
  - kNN(K=0) 0.448 → MLP K=1 **0.563**: 불량 1장을 넣으니 오히려 나빠짐(과적합). MLP K=10 0.140
- 생성 추가 (canon 전처리, 헤드 seed 5개, README 13.6절)
  - DevNet 0.281 → +SD+LoRA 0.430 (나빠짐)
  - MLP 0.659 → +SD+LoRA 0.529 (좋아지지만 여전히 DevNet 실제만보다 나쁨. 약한 모델의 과적합을 줄인 효과로 설명)
  - cTransGAN: 채울 실험 0의 N=40 DevNet
- 그림: `ppt_figures/s11_kcurve_standard_test.png`

#### 12. 시행착오: 지름길 추정 (1:10) [있음]
- 메시지: 모델이 결함이 아니라 불량 사진의 촬영 환경을 배웠다.

| 단계 | 시도 | 결과 | 배운 것 |
|---|---|---|---|
| 1 | 생성 품질을 의심 → 생성기 개선(v3, v4) | 닮음 지표는 오르는데 AD는 그대로 | 품질 문제가 아니다 |
| 2 | 임베딩 진단(`gen_check`) | 생성 불량과 바탕 정상의 거리 = 정상끼리 거리의 0.02–0.07배(melt, SD+LoRA v2). 실제 불량 vs 생성 구분 AUROC 0.911 ≈ 실제 불량 vs 정상 0.914 | 결함이 작아서 이미지 하나로 요약하면 묻힌다 |
| 3 | 결함과 무관한 통계 검사(`shortcut_check`) | AUROC 0.93 / 0.85 / 0.76 | 불량 사진은 촬영 환경이 다르다 |
| 4 | 결함 지우기/붙이기(`defect_removal_check`) | 결함 지운 불량도 44% 불량 판정(DR), 정상 + 결함은 16%만 검출(ND) | **지름길 확인** |
| 5 | 입력 정규화(밝기 맞춤, 기판 정렬, 강한 정규화), 결함 지운 사진을 정상 반례로, 생성 라벨 약하게 | 일부만 개선. 강한 정규화는 작은 결함까지 지움(scratch 0.29 → 0.55). 반례 0.427, 약한 라벨 0.269 | 입력만 손봐서는 안 된다 → 학습 단위를 바꿔야 한다 |
| 6 | 패치 단위 DevNet(`patch_ad`) | 실제만 0.299, +생성 0.418 | 위치 쪽으로 가되 결함 지도를 직접 학습하기로 |

- 그림: `ppt_figures/s12_defect_removal_{pcb1_bent,pcb2_melt,pcb3_scratch}.jpg` (D / DR / N / ND), `ppt_figures/s12_shortcut_diagnosis.png`
- Grad-CAM (채울 실험 1, K=1 seed 0): 이미지 DevNet이 DR을 불량이라 할 때, 히트맵 질량 중 지운 자리 안은 0.025로 우연(0.027)과 같다. 질량 절반은 사진의 18%에 퍼진다.
  그림: `ppt_figures/s12_s14_gradcam_vs_seg_map.jpg`

#### 13. 데이터 특성에 맞춘 조정 (0:50) [있음]
- 메시지: 특성 두 개에 조정 두 개.
- 레이블이 적다 → 위치 seg로 학습하고, 생성 불량은 마스크째 넣는다. 배치 = 정상 4 + 불량 4(생성이 있으면 불량의 절반은 실제).
- 촬영 환경이 다르다 →
  - 학습: 촬영 조건 무작위화(밝기·대비, 조명 기울기, 채도, 블러↔샤픈, 잡음)
  - 평가: 공정 시험셋. 시험 불량 D마다 결함을 지운 DR(음성), 정상에 결함을 붙인 ND(양성), 붙이기 대조 NR(음성)을 만들어 촬영 조건 A·B가 반반 섞인 라인을 흉내 낸다.
- 출처: `AnomalyDetection/ad/local_ad.py`, `AnomalyDetection/ad/fair_ad.py` 설명

#### 14. 결과 (1:20) [일부 있음 / 진행 중 / 필요]
- 메시지: 위치로 학습하면 같은 K장에서 더 많이 얻고, 생성도 도움이 된다.
- 실제 불량만, K=1, 분할 5개 평균(`local_ad_{img,seg}_nogen_K1_seed*.csv`)

| | 이미지 DevNet | MLP | 위치 seg |
|---|---|---|---|
| 미검률 | 0.334 | 0.569 | **0.223** |
| 위치 적중률 | — | — | 0.89 |

- 결함을 보는가 (K=1 seed 0, `local_ad_seg_K1_seed0.csv`, README 13.6절)

| | 이미지 DevNet 실제만 | 이미지 DevNet + SD+LoRA | 위치 seg 실제만 |
|---|---|---|---|
| DR (결함 지운 불량을 불량이라 함, 낮을수록 좋음) | 0.44 | 0.09 | 0.14 |
| ND (정상 + 결함을 잡음, 높을수록 좋음) | 0.16 | 0.45 | 0.74 |

- 공정 시험셋 (K=1 seed 0, 잔상 제거 전 숫자. 채울 실험 0의 결과로 바꾼다)

| 학습 | PatchCore | seg | seg + 촬영 조건 무작위화 |
|---|---|---|---|
| 정상만 | 0.188 | | |
| 실제만 | | 0.304 | 0.213 |
| + SD+LoRA | | **0.207** | 0.219 |
| + cTransGAN | | ? | ? |

  - 읽는 법: 생성은 seg에서 0.30 → 0.21로 도움이 된다. 다만 촬영 조건 무작위화를 같이 쓰면 이득이 평균적으로 사라진다(0.213 vs 0.219). 그래서 생성의 주된 역할은 "불량 레이블을 정상 촬영 환경으로 옮겨 지름길을 끊는 것"으로 해석한다.
  - K=1에서는 아직 PatchCore(0.188)가 가장 낫다. 정상만을 넘는 K는 공정 K 곡선으로 확인한다.
- SMT (현장형, 정상이 다양함, 5 seed, `AnomalyDetection/results/smt/summary.txt`)
  - 사전공정: PatchCore 0.910, kNN 0.767 → DevNet K=1 0.213, MLP K=1 0.301
  - 위치 seg(K=1, seed 0–2, `results/smt/seg_ad_*.csv`, 표준 시험셋 과검률 1%, 자리 고치기 전. 채울 실험 0-2로 바꾼다): 실제만 사전 0.199, 납땜 0.802
    - + SD+LoRA 종류당 40장: 사전 0.136, 납땜 0.736 (seed 3/3 개선). 같은 SD+LoRA가 이미지 DevNet에서는 어떤 설정으로도 실제만을 못 넘었다.
    - + sd_real_s30(실제 결함을 붙인 뒤 SD+LoRA로 30% 다시 그림) 10장: 사전 0.167, 납땜 0.636
    - 부록: 실제 결함 옮겨 붙이기 40장 사전 0.100–0.124, 납땜 0.624. 종류당 1장은 효과 없고 5장부터 이득. 표는 `AnomalyDetection/smt/README.md`
- 그림: seg 결함 지도 + 이미지 DevNet Grad-CAM 나란히 [있음] `ppt_figures/s12_s14_gradcam_vs_seg_map.jpg`
  - 시험 불량에서 지도 질량 중 결함 영역 안: DevNet 0.04(우연 0.03) vs seg 0.75. 최고점이 결함 위: 0.11 vs 0.87 (표는 채울 실험 1)
- 채울 실험: 0, 0-2, 0-3 C (그림: 생성기·공정마다 x = 합성 장수 N, 선 = K)
- 그림 [있음]: `ppt_figures/s14_grid_visa.png`, `ppt_figures/s14_grid_smt.png` (숫자는 `AnomalyDetection/results/ppt/grid_summary.md`), 공정 K 곡선 `ppt_figures/s14_fair_kcurve.png`. SMT 생성 예시는 `ppt_figures/local_only/` (AIHub 파생이라 git 제외)

### 마무리

#### 15. PCB 현장에서의 활용 (0:40) [있음]
- 메시지: 우리 상황은 특수한 경우가 아니라 현장의 기본값에 가깝다.
- 왜 이런 상황이 생기나

| 원인 | 어떻게 달라지나 |
|---|---|
| 찍는 장비가 다르다 | 정상은 인라인 카메라로 자동 저장, 불량은 리뷰(재검) 스테이션이나 분석실에서 다시 찍는다 |
| 찍는 시기가 다르다 | 정상은 며칠이면 수천 장, 불량은 몇 달에 걸쳐 모인다. 그 사이 조명·카메라·레시피가 바뀌어 드리프트가 불량 라벨과 섞인다 |
| 불량을 일부러 만든다 | 데이터셋을 만들 때 긁고, 빼고, 녹여서 따로 찍는다(VisA 논문에 명시) |
| 반품 불량 | 현장에서 돌아온 불량은 실험실에서 전혀 다른 조건으로 찍힌다 |
| 검사기에 걸린 것만 라벨이 붙는다 | 기존 검사기가 잘 잡는 불량만 모여 쏠림이 생긴다 |

- 라인에 올리면 정상도 불량도 같은 인라인 카메라로 들어온다. "리뷰 스테이션 조건 = 불량"을 배운 모델은 그 단서가 사라져 불량을 놓친다.
- 사용 조건: **불량 유형이 미리 정해진 라인**(PCB 조립은 IPC-A-610 같은 기준서로 불량 유형이 정해진 경우가 많다)이라면, 불량 몇 장에 박스·마스크만 그려 위치 seg + 생성으로 학습한다.
- 새 불량 유형 대비: 정상만 쓰는 PatchCore를 함께 쓰는 방식을 제안한다(시험은 안 함).

#### 16. 이상탐지 일반 활용 · 차별점 · 유용성 · 확장성 (1:00) [있음]
- 현장의 대응과 우리 결과

| 현장에서 쓰는 대응 | 우리 결과가 주는 것 |
|---|---|
| 불량을 찍을 때 같은 조건의 정상도 함께 찍기 | 같은 조건으로 다시 찍기 어려울 때, 생성 불량을 정상 사진 위에 만들어 "정상 조건의 불량"을 확보할 수 있다(DR 0.44 → 0.09) |
| 전처리로 조건 맞추기 | 일부만 효과. 강하게 맞추면 작은 결함까지 지운다(scratch 0.29 → 0.55) |
| 밝기·색 변환을 정상과 불량에 똑같이 | 촬영 조건 무작위화로 구현. 위치 seg와 함께 쓸 때 효과(0.304 → 0.213) |
| 정상 위주로 학습하고 불량은 임계값에만 | 정상이 균일한 라인(VisA)에서는 강한 기준선. 정상이 다양한 라인(SMT)에서는 무너진다(0.910) |
| 조건만으로 정상/불량이 갈리는지 점검 | 결함 지우기/붙이기(DR/ND)까지 확장해 "모델이 결함을 보는지"를 직접 잰다 |

- 차별점
  1. 같은 K장과 같은 생성 데이터에서 레이블 단위만 바꿨더니, 생성의 효과가 해로움에서 도움으로 뒤집혔다.
  2. 불량 사진의 촬영 환경 차이를 측정하고, 지름길을 뺀 평가(공정 시험셋)를 제안했다.
  3. K장 전용 생성기 두 종류(처음부터 학습하는 GAN vs 사전학습 확산 모델 + LoRA)를 같은 규칙에서 비교했다.
- 유용성
  - 박스·마스크 하나로 같은 레이블의 값어치가 올라간다.
  - 배포 전에 결함 지우기/붙이기로 모델을 점검할 수 있다.
  - 생성 증강에 투자할지 판단하는 기준(위치 학습일 때, 불량이 적을 때)을 준다.
- 확장성 (조건부)
  - 결함이 작고, 불량을 따로 모으고, 불량 유형이 정해진 검사라면 같은 문제와 해법이 적용될 것으로 예상한다. 수율이 높아 불량이 드문 공정이 대표적이다.
  - 근거는 PCB 데이터 두 종류(VisA: 사람이 만든 불량, SMT: 현장 수집)다.
  - 결함 지우기/붙이기 점검은 마스크나 박스만 있으면 어느 데이터에나 쓸 수 있다.

#### 17. 한계와 결론 (0:30) [있음]
- 한계
  - seed 수가 적다(특히 공정 시험셋과 SMT 위치 모델).
  - VisA에서 정상만 학습을 넘는 K는 확인 중이다.
  - 처음 보는 불량 유형은 시험하지 않았다(시험셋이 학습과 같은 4종류).
  - 데이터가 PCB 두 종류다.
- 결론: 한 줄 메시지 반복.

---

## 채울 실험

공통: 프로젝트 루트에서 `PY=/home/junho/miniforge3/envs/26_2_dl/bin/python`, `DR=Dataset/visa/VisA_official/VisA_PCB`(절대경로로), `export OMP_NUM_THREADS=4`.

### 0. VisA: 잔상 제거 후 생성기 × K × 합성 장수 N (슬라이드 9, 11, 14) [진행 중, 11/18 15:20~]
- 왜: bent 생성 이미지에 원래 곧은 핀이 남았다(pcb3 심함, 세 생성기 모두). seg에 생성을 넣으면 bent 미검률만 0.09 → 0.15–0.21로 나빠졌다.
- 잔상 제거 `Augmentation/pin_cleanup/clean.py` (bent만, 규칙은 0-1). 원래 세트·이전 결과는 `Dataset/generated/_before_pinfix/`, `AnomalyDetection/results/_before_pinfix/`에 보관하고, 고친 세트를 같은 이름(`sd_lora_v3`, `ctransgan_diverse`)에 둔다.
- 반복: {SD+LoRA v3, cTransGAN} × K {1, 2, 4, 8} × N(종류당) {0, 1, 5, 10, 20, 40}, seed 0 (SD+LoRA K1은 seed 1–4도).
  위치 seg(촬영 조건 무작위화), 공정 시험셋. 배치 안 실제:합성 = 반반 고정. N장은 `Augmentation/gen_subset.py`(고정 순서의 앞 N장). N=40에서는 seg·DevNet도. N=0은 공정 K 곡선 결과.
- 실행 `AnomalyDetection/results/logs/run_pinfix_rerun.sh` → 결과 `fair_ad_K{k}_seed{s}_grid_{v3|ctg}.csv`, 생성 품질 `gen_realism_K1_seed0.csv`, 끝에 Grad-CAM 다시.
- copy-paste·anomalydiffusion·SD+LoRA v2는 다시 안 돌린다(부록, copy-paste에도 잔상 → 한계).

### 0-1. 생성 흔적을 막는 규칙 (cTransGAN 담당자: 다시 만들 때 생성기 안에 넣기)
**VisA bent 핀 잔상**
- 원인: `generate_diverse.structure_region` → `common/kshot.change_region`이 지울 핀을 (1) **흑백** 밝기 차이로 찾아 은색 핀/초록 기판을 놓치고, (2) 마스크 **20px** 안만 봐서 헤더 너머 핀을 빠뜨린다.
- 규칙 (`pin_cleanup/clean.py` `residue_region`을 import해도 됨)
  1. 핀 = 색으로: 채도 (max−min)/max < 0.30, 가장 밝은 채널 > 110 (2px 흐림 뒤).
  2. 바탕에는 핀이 있고 실제 결함 사진의 같은 자리 ±6px에는 없는 곳만 (정상 핀은 안 건드림).
  3. 마스크(또는 실제 결함 자리)에서 45px 안에 닿는 덩어리, closing 4 → 5px 넓힘.
  4. 실제 결함 사진의 같은 자리(배경)로 채움. 밝기는 주변 3–14px 고리 중앙값으로 맞추고 경계 3px 섞음.
  5. bent만 (missing에 켜면 핀 가장자리를 파먹는다).
- 확인: 생성 뒤 `pin_cleanup.clean` 로그의 "고친 장수"가 거의 0, review 그림 pcb3 bent에 곧은 핀이 없음.
- 남은 한계: K=1이면 종류당 마스크 모양이 하나라 40장의 휜 핀 길이·각도가 같다.

**SMT 결함 자리** (SMT cTransGAN을 만들 때. `smt_sd_lora/generate_fix.py`의 `choose_site`, `place_fixed`)
- 원인: 결함 박스를 실제 결함 크기 그대로, 무작위 '정상_{종류}' 자리의 한가운데에 둔다 → 납땜 nm·ms는 부품이 반쯤 남고, rs·sh·ws는 큰 IC 위에 무늬.
- 규칙
  1. 자리: 실제 결함 박스와 크기·가로세로비가 가까운 상위 25%(최소 8개)에서 무작위 (nm은 크기 무시). 박스 방향을 자리 방향에 맞춤.
  2. ms = 자리 전체 + 6%. nm = 자리 긴 축 한쪽 끝에 수직인 띠(두께 = 실제 결함 짧은 변 × 0.9–1.1, 폭 = 부품 폭 × 1.06). 나머지 = 자리 중심 ± 5%.
  3. 바탕과 같은 JPEG 양자화 테이블로 저장, 누수 검사, 정답 = `condition.box`.
- 결과 폴더 `Dataset/aihub_smt/generated/ctransgan/K{k}_seed{s}/`(K 1·4·16 × seed 0–2, 종류당 40장),
  AD: `python -m AnomalyDetection.smt.seg_ad --category {pre|solder} --k K --seeds 0 1 2 --conds none ctransgan:{1,5,10,20,40}:0.5`

### 0-2. SMT: 고친 SD+LoRA로 K × N (슬라이드 14 SMT 줄) [진행 중, 11/18 16:12~]
- `sd_lora_fix` = `sd_lora`에서 납땜 nm·ms·rs·sh·ws만 0-1 규칙으로 다시 만든 것. ms는 SD가 부품을 다시 그리는 장이 많아 한계로 두고 0-3 A에서 따로 본다.
- 공정(사전 6종, 납땜 12종) × K {1, 4, 16} × N {0, 1, 5, 10, 20, 40} × seed 0–2, 위치 seg(`smt/seg_ad.py`), 합성 비율 0.5 고정. 지표: 미검률 @ 과검 1%, 유형별 미검률.
- 실행 `AnomalyDetection/results/logs/run_smt_fix_grid.sh` → `results/smt/seg_ad_{pre,solder}.csv` (cond `sd_lora_fix:{N}:0.5`). 11/19 새벽 예상.

### 0-3. 예약 [11/18 16:29, `run_followups.sh`, 앞 작업이 끝나면 저절로 시작]
- **A. SMT ms 따로 실험** (0-2 뒤): `smt_sd_lora/generate_ms_real.py`로 ms만 실제 빈 패드(자리 전체, 기판 색 맞춤, LoRA strength 0.3)로 바꾼 `sd_lora_fix_msreal`, 납땜 seg N 5·40. ms 유형 미검률을 `sd_lora_fix`·실제만과 비교.
- **B. VisA cTransGAN K1 seed 1–4** (0번 AD 격자 뒤): 생성 → 잔상 제거 → 부분 세트 → AD 칸. 보관함에 manifest가 이미 있으면 생성은 건너뛴다.
- **C. 집계** (모두 끝난 뒤): `ad/grid_summary.py` → `results/ppt/grid_{visa,smt}.png`, `grid_summary.md` (x = N, 선 = K / bent / SMT 유형 × N / ms 비교), 이어서 `ppt_summary` + `ppt_figures/collect.sh`, 그림은 `ppt_figures/s14_grid_{visa,smt}.png`.

### 1. Grad-CAM과 결함 지도 (슬라이드 12, 14) [있음, 11/18 10:46. 0번 끝에 잔상 제거 SD+LoRA로 다시]
- `AnomalyDetection/ad/gradcam.py`(`run_gradcam.sh`). K=1 seed 0, 공정 시험셋 D·DR·ND 같은 사진. DevNet·MLP는 Grad-CAM(layer3), seg는 출력 지도.
- 결과 (`gradcam_K1_seed0.csv`, 결함 영역은 사진의 2.8% = 우연 수준)

| D(시험 불량) 기준 | 결함 영역 안 질량 | 최고점이 결함 위 | 질량 절반 면적 | DR 불량 판정 | ND 검출 |
|---|---|---|---|---|---|
| 이미지 DevNet 실제만 | 0.04 | 0.11 | 16% | 0.58 | 0.17 |
| 이미지 DevNet + SD+LoRA | 0.05 | 0.21 | 12% | 0.25 | 0.43 |
| 이미지 MLP 실제만 | 0.06 | 0.16 | 16% | 0.29 | 0.06 |
| 위치 seg 실제만 | **0.75** | **0.87** | **2.3%** | 0.21 | 0.74 |
| 위치 seg + SD+LoRA | **0.85** | **0.88** | **0.2%** | 0.11 | 0.78 |

- DevNet이 DR을 불량이라 할 때 지운 자리 안 질량 0.025(우연 0.027), 질량 절반이 사진의 18%에 퍼짐 → 결함이 아닌 기판 전체를 본다. LayerCAM도 0.05–0.06.
- 그림 `AnomalyDetection/results/gradcam/K1_seed0_all.jpg` (흰 선 = 결함 영역, 자홍 원 = 최고점). 한계: seed 0 하나, 설명 방법이라 DR/ND 숫자를 보강하는 근거로만.

### 하지 않을 것
- 새 생성기, VLM, MVTec, 전처리 변형, anomalydiffusion·copy-paste·transplant 추가 분석

---

## 주장 수위

| 확실히 말해도 됨 | 조건을 붙여서 | 말하면 안 됨 |
|---|---|---|
| 같은 K장에서 위치 라벨이 이미지 라벨보다 훨씬 낫다 | VisA에서 정상만 학습을 넘는 K (공정 K 곡선 결과로) | "PCB 불량 사진은 꼭 다르게 찍힌다" |
| 이미지 단위 모델은 촬영 환경 지름길을 배운다 | 생성이 지름길 끊기를 넘어 새 지식을 더한다 | "기존 연구는 불량 레이블을 안 썼다" |
| 정상이 다양한 라인(SMT)에서는 불량 레이블이 필수다 | 다른 산업으로의 확장 | cTransGAN이 AD에 도움이 된다 (결과 전까지) |
| 우리가 본 PCB 데이터 두 개는 불량 사진의 촬영 환경이 달랐다 | 위치 seg를 처음 보는 불량 유형에 쓰기 | |

## 참고 자료
- VisA 논문 (Zou et al., ECCV 2022): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf
- Baitieva et al., CVPR 2024 보충자료: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Baitieva_Supervised_Anomaly_Detection_CVPR_2024_supplemental.pdf
- PRN (CVPR 2023): https://arxiv.org/abs/2212.02031
- cTransGAN 원논문 (Wang et al., 2023): https://onlinelibrary.wiley.com/doi/10.1155/2023/2024237
- 리뷰 스테이션 구조 (US 5369431): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5369431
- 생산 AOI 데이터의 드리프트: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10847760/


### SMT cTransGAN 비교 완료 (11/19)

- 공정 pre·solder, K={1,4,16}, seed={0,1,2}, 종류당 생성 40장으로 총 6,480쌍을 생성했다.
- diffusion과 같은 분할·정상 부품 자리 규칙, 합성 비율 0.5, N={0,1,5,10,20,40}으로 위치 탐지 모델을 비교했다.
- 과검률 1%의 평균 미검률(실제만 → 생성 종류당 40장), 3 seed 평균이다.

| 공정 | K1 | K4 | K16 |
|---|---|---|---|
| pre | 0.199 → 0.163 | 0.106 → 0.063 | 0.034 → 0.019 |
| solder | 0.802 → 0.725 | 0.564 → 0.509 | 0.409 → 0.728 |

- 납땜 K16은 모든 N에서 악화됐다. 일부 K16 GAN 학습의 판별기 손실이 거의 0으로 떨어졌으며 생성 분포의 불안정성을 추가 검토해야 한다.
- 생성 패치가 학습 원본과 유사하게 재구성될 수 있다. 위 수치는 원본과 다른 결함 모양을 생성했다는 증거를 대신하지 않는다.
- 원본 수치 108행과 전체 N 표는 [ctransgan_comparison.csv](AnomalyDetection/results/smt/ctransgan_comparison.csv), [ctransgan_summary.md](AnomalyDetection/results/smt/ctransgan_summary.md)에 있다.
