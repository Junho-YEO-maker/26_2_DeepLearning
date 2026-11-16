# sd_lora — SD 인페인팅 + LoRA

Stable Diffusion 1.5 인페인팅 모델에 **K장만으로** LoRA를 학습해서 결함을 그려 넣는다.
조건 (K, seed)마다 PCB별 LoRA 1개(결함 4종류는 프롬프트로 구분). 실행은 **GPU 서버**에서 한다(Colab·Kaggle 노트북은 11/16 삭제).

## 실행 (GPU 서버)
```bash
cd Augmentation
pip install -r sd_lora/requirements.txt                  # torch는 서버 CUDA에 맞게 먼저
export DATA_ROOT=/abs/path/VisA_PCB                      # pcb1–3 원본의 부모 (없으면 Dataset/visa/raw)
python -m sd_lora.train --k 1 --seed 0 --pcb pcb1 --steps 100 --data-root $DATA_ROOT --runs-dir sd_lora/runs_test  # 짧은 시험
bash sd_lora/run_all.sh                                  # v3 단계 B, K=1 seed 0: 학습 → 생성 → gen_check
KS="1 2 4" bash sd_lora/run_all.sh                       # 합격하면 1단계 전체
```
- 결과: `Dataset/generated/sd_lora_v3/K{k}_seed{s}/`, 로그 `sd_lora/run_v3_K{k}_seed{s}.log`(끝에 gen_check 표).
- AD 실험: 저장소 루트에서 `python -m AnomalyDetection.ad.run --models knn_mlp devnet --ks 1 --seeds 0 --augs sd_lora_v3 --res 512 --data-root $DATA_ROOT`.
  GPU에서 PCB당 몇 분(이 PC CPU에서는 3 PCB에 48분).
- 이미 끝난 조건은 건너뛴다. 다시 하려면 `--force`. 서버로 옮길 것: 이 저장소, VisA pcb1–3 원본(분할표 해시와 대조됨). HF 베이스 모델(약 4GB)은 자동 다운로드.

## 결과 기록

### v2 (recipe `v2-region+normal`, Kaggle T4 ×2, K1_seed0, 11/13)
`Dataset/generated/sd_lora/K1_seed0/` 480장, 누수 검사 PASS. AD 결과는 `AnomalyDetection/README.md` 13.5절.

| FNR@FPR1% 평균 (PCB1–3, seed 0) | none | classic | copypaste | sd_lora v2 |
|---|---|---|---|---|
| KNN+MLP | 0.687 | **0.527** | 0.628 | 0.638 |
| DevNet | 0.332 | **0.331** | 0.563 | 0.596 |

**진단** (`python -m AnomalyDetection.ad.gen_check --aug sd_lora --k 1 --seed 0`, 지표 뜻은 그 파일 docstring):

| 지표 | 값 | 뜻 |
|---|---|---|
| edit (생성 ↔ 자기 바탕, 정상끼리 거리 = 1) | melt **0.02–0.07**, missing 0.04–0.15, bent 0.12–0.79, scratch 0.20–0.39 | AD가 보는 이미지 임베딩에서는 결함이 거의 안 보인다 |
| novelty (생성 / 실제 시험 불량) | melt 1.02–1.05 / 1.12–1.25 | 실제 불량보다 정상 쪽에 붙어 있다 |
| sep (실제 시험 불량 vs 생성) | **0.911** (대조: vs 정상 0.914) | 생성 불량이 특징 공간에서 정상과 구별되지 않는다 |
| 픽셀 change (마스크 안 \|생성 − 바탕\|) | melt 44–52, scratch 76–85, bent 26–66 | 픽셀로는 꽤 그렸다 → 문제는 "안 그림"만이 아니다 |
| 같은 종류 40장 다양성 | 거의 같음 | K=1 결함 1장을 같은 자리 ±5–30px에만 둠 |

**원인 (11/16 확인): VisA 불량 사진은 촬영 조건이 정상과 다르다.** 결함과 무관한 전역 통계 8개(밝기·RGB·대비·기판 위치)만으로
실제 시험 불량 vs 정상을 AUROC **0.93 / 0.85 / 0.76** (PCB1/2/3)으로 구분한다(정상 train vs 정상 test는 0.40–0.53).
PCB2 불량 사진은 평균 회색 88 → 102로 밝고(d = 1.9), PCB1은 기판이 아래로 치우쳐 있다(d = 2.1).
실제 불량으로 학습한 AD는 이 촬영 차이도 불량 신호로 쓰는데, 생성 불량은 정상 사진을 바탕으로 하므로 그 신호가 없다.
copypaste(실제 결함 픽셀을 그대로 붙임)도 똑같이 해로웠던 이유가 이것으로 설명된다. → **생성 품질만 올려서는 AD 점수가 오르지 않는다.**
AD 쪽 처리는 `AnomalyDetection/README.md` 13.5절.

## v3 방향 (GPU 서버)

판정은 매번 `gen_check`로 먼저 한다(PCB당 몇 분). 전체 `ad.run`은 합격한 것만.

| 순서 | 무엇 | 코드 | 합격 기준 (gen_check) |
|---|---|---|---|
| 0 | **AD 쪽 밝기 지름길 제거가 먼저** (13.5절). 이게 없으면 아래 어떤 개선도 FNR에 안 보일 수 있다 | AD 팀 | — |
| A | 생성만 바꿈: 기존 v2 LoRA로 `--diverse --overgen 2`. LoRA 재학습 없이 싸게 효과 분리 | `STAGE=A bash sd_lora/run_all.sh` (v2 LoRA를 `sd_lora/runs/`에 복사해 둘 것: Kaggle Output의 `lora/`) | div 상승, edit 하위 25% 제거, sep < sep_normal |
| B | 학습도 바꿈: 정상 샘플 25% → **10%**, 500 → **1000 step**, rank 8 → **16** | `bash sd_lora/run_all.sh` (기본) | melt·missing edit ≥ 0.2, novelty가 실제 불량(real_novelty)에 근접 |
| C | B 합격 시 K {1, 2, 4} × seed 0 → `ad.run --augs sd_lora_v3` | `KS="1 2 4"` | none 대비 FNR 개선 (6.2절 기준) |

바뀐 것 (11/16):
- **`--diverse`** (`common/kshot.py` `augment_mask`, `board_place`): melt는 원래 부품 자리에서 결함 덩어리마다 좌우 반전·크기 0.75–1.35·회전 ±30°
  (pcb1 melt처럼 덩어리가 여럿이면 각자 중심 기준, 기판 밖으로 밀리면 다시 뽑음). scratch는 기판 위 임의 위치 + 회전 ±180°.
  bent·missing은 핀 자리에 묶여 있어 그대로. 인페인팅은 마스크 안을 새로 그리므로 copy-paste와 달리 모양이 실제로 달라진다.
- **`--overgen N`**: 종류마다 40×N장을 만들고 마스크 안 픽셀 변화(`change`)가 큰 40장만 남긴다. manifest `score` 열에 기록.
  v2에서 change가 낮은 샘플(bent 13, missing 12, scratch 9)은 사실상 빈 기판이었다.
- **`--tag v3`**: 결과를 `Dataset/generated/sd_lora_v3/`에 따로 쓴다. AD `run.py`는 이제 `Dataset/generated/` 아래 아무 폴더 이름이나 aug로 받는다
  (v2 결과 `sd_lora`는 results.csv에 그대로 남는다).
- 위 코드는 CPU에서 가짜 파이프라인으로 끝까지 돌려 확인했다(72개 후보 → 36장, 누수 검사 PASS). 실제 디퓨전 출력은 서버에서 처음 본다.

하지 않기로 한 것:
- 생성 이미지 밝기를 불량 사진 쪽으로 맞추기: 시험셋 지름길을 생성에 심는 것이라 결과 해석이 망가진다. 지름길은 AD 쪽에서 없앤다.
- 바탕 정상을 AD 정상 학습셋에서 빼기: 해 봤는데 효과 없음(DevNet 0.596 → 0.627).
- pcb1 missing: 시험 불량 면적 중앙값 4.4%인데 K1 donor는 0.11%. K=1에서는 어떻게 해도 못 맞춘다. 결과 해석 때 언급만.

## 하는 일 (v2 기준, v3는 위 옵션만 추가)
| 단계 | 내용 |
|---|---|
| 학습 | 그 PCB의 K장 불량(종류당 K장)에서 결함 주변 512×512 크롭을 매번 다른 위치로 자르고, 인페인팅 영역(8px 넓힘)을 가린 뒤 실제 불량을 다시 그리게 학습. 배치의 일부(`--p-normal`)는 정상 샘플 |
| 생성 | K장 중 하나의 결함 마스크를 train_normal 바탕에 기판 정렬(평행이동 + PCB1의 180° 회전) 후 같은 부품 자리 ± 종류별 jitter(bent·missing 5px, melt·scratch 30px)로 두고(copy-paste와 같은 함수), 주변 512 크롭을 인페인팅한 뒤 **인페인팅 영역 안만** 원본 해상도 바탕에 붙인다 |
| 결과 | `Dataset/generated/sd_lora[_{tag}]/K{k}_seed{s}/` (형식은 [Augmentation/README.md](../README.md)) |

### 학습 방식 (recipe `v2-region+normal`)
- **인페인팅 영역 = 결함 마스크, bent·missing은 + 원래 핀 자리.** VisA 주석은 휜 핀 토막·빠진 자리만 덮어서, 마스크만 다시 그리면 정상 바탕의 원래 핀이 남아 "다리가 하나 더" 생겼다(v1). 실제 불량과 정상 기판을 정렬·비교해 크게 다른 덩어리 중 마스크에 닿는 것을 영역에 더한다(`common/kshot.py`의 `change_region`). 학습과 생성이 같은 함수를 쓴다. 정답 마스크(출력 `masks/`)는 VisA 주석 그대로.
- **정상 샘플 (`--p-normal`, v2 25%, v3 10%).** 같은 영역을 정상 기판(train_normal, 불량 1장당 3장 `--refs`)의 같은 자리에 씌우고 결함 단어 없는 프롬프트로 정상 기판을 정답으로 준다. 흰 얼룩 남발과 K장 암기를 줄이지만, 많으면 결함을 약하게 그린다(v2 진단).
- **프롬프트는 종류마다 보이는 모양을 쓴다**: bent `a metal header pin bent and tilted out of line`, melt `a component melted into a white burnt blob`, scratch `thin white scratch lines on the board surface`, missing `a pin or component is missing, empty spot`, 정상 `clean, intact, all parts in place` (모두 `close-up photo of sks circuit board, ...`).
- 밝기·대비 ±5%. 뒤집기는 안 한다(실크 글자가 거울상이 됨).
- 옛 방식(v1) LoRA는 train이 건너뛰지 않고 다시 학습하고, generate는 거부한다(`train.json`의 `recipe`). v3 단계 B는 학습 설정만 바뀌어 recipe는 같으므로 `--runs-dir sd_lora/runs_v3`로 폴더를 나눈다.

- 결함이 512보다 크면(pcb1 missing 등) 크롭을 키운 뒤 512로 줄여 넣는다.
- 기본값: 500 step, batch 4, lr 1e-4, rank 8, 샘플링 30 step, guidance 7.5. bf16(compute capability ≥ 8) 또는 fp16 + gradient checkpointing 자동. T4 실측 LoRA 1개당 약 50분(500 step).
- LoRA 가중치: `sd_lora/runs*/K{k}_seed{s}/{pcb}/` (git 제외). `train.json`에 recipe, 학습 소스 K장, 쓴 정상 이미지, 영역 넓이 비(`region_vs_mask`), 시간 기록.
- 베이스 모델: `stable-diffusion-v1-5/stable-diffusion-inpainting` (Hugging Face에서 자동 다운로드, 약 4GB).
- `_compat.py`: 오래된 torchao(< 0.16)가 깔려 있으면 peft가 LoRA를 만들다 멈추는 것을 막는다. torchao가 없거나 새 버전이면 아무것도 안 하므로 서버에서도 그대로 둔다.

## 확인할 것
- 생성 이미지 몇 장을 눈으로 보고, 결함이 마스크 안에 실제로 그려졌는지(빈 기판이 아닌지) 확인. manifest `score`(change)가 낮은 것부터 보면 빠르다.
- 누수 검사는 generate가 끝에 자동으로 한다(`common.kshot.check_manifest`). AD `run.py`도 학습 전에 한 번 더 한다.
