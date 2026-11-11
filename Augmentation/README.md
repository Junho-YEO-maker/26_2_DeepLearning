# Augmentation 팀

불량 K장(`Dataset/visa/splits/train_K{k}_seed{s}.csv`)만 소스로 써서 생성 불량 이미지를 만들고
`Dataset/generated/{방법}/K{k}_seed{s}/` 에 쌓는다. AD 팀은 이 폴더를 그대로 가져다 쓴다.

## 공유 분할표 (`Dataset/visa/splits/`, 2026-11-11 생성, 고정)
`scripts/make_splits.py`로 만들었다. 규칙은 AnomalyDetection/README.md 2.2절. 분할 seed 7017.

| 파일 | 내용 | 생성에 써도 되나 |
|---|---|---|
| `train_K{k}_seed{s}.csv` | PCB별·종류별 불량 K장 (K ∈ {1,2,4,8}, s ∈ 0–4, 같은 s 안에서 K끼리 중첩) | **결함 소스는 여기서만** |
| `train_normal.csv` | PCB별 정상 500장 | **정상 바탕은 여기서만** |
| `pool.csv` | K장을 뽑는 후보 전체(종류별 10장, pcb2 bent 8장) | 통째로 쓰면 안 됨 |
| `val_normal.csv`, `test.csv` | 임계값용 정상, 시험셋(불량 182장 + 정상 911장) | **금지** |

- 열: `category, image, mask, label, split, defect_type, image_sha256`
- 경로는 `Dataset/visa/raw/...`로 통일. 원본이 다른 곳에 있으면 코드에 `--data-root`로 넘긴다.
- 마스크 픽셀값 = 종류 ID: 1=bent, 2=melt, 3=scratch, 4=missing

## 폴더 (방법마다 따로)
```
Augmentation/
├─ common/kshot.py     모든 방법이 같이 쓰는 것: 공유 분할 읽기(test/val 차단), 결함 위치 규칙, 원본에 다시 붙이기, manifest 기록·누수 검사
├─ visa_pcb/           copypaste (src/generate/k_copypaste.py), ctransgan, 33장 비교 실험
├─ sd_lora/            SD 인페인팅 + LoRA           → sd_lora/README.md
└─ anomalydiffusion/   AnomalyDiffusion 공식 코드 어댑터 → anomalydiffusion/README.md
```
결과는 방법과 상관없이 전부 `Dataset/generated/{방법}/K{k}_seed{s}/` 한 곳에 같은 형식으로 쌓인다.

## 방법
| 방법 (aug 이름) | 성격 | 범위 | 상태 |
|---|---|---|---|
| `copypaste` | 학습 없음, 기준선 | 모든 K × seed 0–4 | 코드 완료 |
| `ctransgan` | GAN | K {1,2,4} × seed 0 | 1단계 (AD 팀 합의 ⑤), 진행 중 |
| `sd_lora` | 범용 diffusion + LoRA | K {1,2,4} × seed 0 | 코드 완료, A100 실행 대기 |
| `anomalydiffusion` | 불량 생성 전용 diffusion | K {1,2,4} × seed 0 | 어댑터 완료, **보류**: copy-paste·sd_lora 결과에서 생성 쪽 이득이 보일 때만 실행 |

- 생성 모델은 조건마다 **해당 K장으로만** 새로 학습한다(33장·풀 전체로 학습한 체크포인트 재사용 금지).
- 결함 위치·모양은 모든 방법이 같은 규칙을 따른다: K장 결함 마스크를 기판 정렬(평행이동 + PCB1의 180° 회전) 후 같은 부품 자리 ± 종류별 jitter(bent·missing 5px, melt·scratch 30px). 규칙은 `visa_pcb/src/generate/k_copypaste.py` 한 곳에 있고 `common/kshot.py`가 그대로 가져다 쓴다. 방법끼리는 "마스크 안을 무엇으로 채우나"만 다르다.
- 장수는 방법끼리 같게, 조건당 종류별 40장(PCB당 160장)으로 맞춘다(9절 ⑦). 장수 변화 실험(10/40/160)은 선택.

## 실행 (copy-paste)
`Augmentation/visa_pcb`에서:
```bash
python -m src.generate.k_copypaste --k 2 --seed 0                 # 조건 하나
python -m src.generate.k_copypaste --all                          # K {1,2,4,8} × seed 0–4 = 20개 조건
python -m src.generate.k_copypaste --all --data-root /content/VisA_PCB   # 원본이 다른 곳에 있을 때
python -m src.data.shared_splits --check ../../Dataset/generated/copypaste/K2_seed0/manifest.csv   # 누수 검사
```
- 조건마다 PCB 3개 × 종류 4개 × 40장 = 480장. 로컬 CPU에서 조건당 약 5분.
- 이미 만든 조건은 건너뛴다. 다시 만들려면 `--force`.
- 결함은 기판 정렬(평행이동 + PCB1의 180° 회전) 후 같은 부품 자리 ± 종류별 jitter(bent·missing 5px, melt·scratch 30px)에 붙이고, 패치에 약한 변형(반전, ±3°, 크기 ±10%, 밝기·대비 ±5%)을 건다(`--placement same`, 기본). JPEG는 원본과 같은 q75 4:2:0. 2단계 기준선처럼 임의 위치·회전으로 붙이려면 `--placement random`.
- 새 생성 방법도 `src/data/shared_splits.py`의 `SharedSplits`로만 원본을 읽는다. test/val 이미지는 경로·해시 둘 다 막힌다.

## 출력 형식 (AD 팀 합의, 9절 ⑧)
```
Dataset/generated/{방법}/K{k}_seed{s}/
├─ images/      {pcb}_{종류}_{번호}.jpg   원본 해상도, 원본처럼 JPEG
├─ masks/       {pcb}_{종류}_{번호}.png   픽셀값 = 종류 ID(1–4)
└─ manifest.csv
```
| 열 | 값 |
|---|---|
| `image`, `mask` | 프로젝트 루트 기준 상대경로 |
| `label` | 1 |
| `source_image` | 결함을 가져온 불량. 반드시 같은 PCB의 `train_K{k}_seed{s}.csv` 안 |
| `score` | 품질 필터 점수 (없으면 빈칸) |
| `category`, `defect_type` | pcb1–3, bent/melt/scratch/missing |
| `base_normal_image` | 결함을 붙인 정상 바탕. 반드시 `train_normal.csv` 안 |
| `gen_seed` | 생성 난수 시드 (아래) |
| `condition` | 붙인 위치·면적 등 JSON |

- images/, masks/ 는 git 제외 (Drive), manifest.csv 만 git
- 이미지를 JPEG로 저장하는 이유: 실제 이미지가 JPEG라서, PNG로 두면 "압축 흔적이 없으면 생성"이라는 지름길을 탐지 모델이 배울 수 있다.

## 시드 규칙
- 팀 시드는 7017. 폴더 이름의 `seed{s}`는 AD 팀의 K 샘플링 seed(0–4)로, **어떤 불량 K장을 뽑았는지**만 뜻한다.
- 생성 난수는 `visa_pcb/src/seeds.py`가 7017과 (방법, PCB, K, s)를 묶어 만든다. 조건마다 난수열이 달라진다.
  - `condition_seed(method, pcb, k, s)`: torch 등에 넣는 조건별 정수 시드 → manifest `gen_seed`
  - `sample_rng(method, pcb, k, s, index)`: 샘플별 numpy 난수 생성기

## 금지
- `test.csv`, `val_normal.csv` 이미지를 소스·바탕·학습·참조에 쓰지 않는다
- 풀 전체나 불량 100장으로 학습한 생성 모델을 K장 조건에 돌려쓰지 않는다
- 같은 (방법, K, seed) 결과를 덮어쓸 때는 manifest 도 같이 갱신한다

## 검사
```bash
python -m tests.smoke_k_copypaste   # 더미 데이터로 분할 → 생성 → 누수 검사
python -m tests.smoke_seeds
```

## 실행 환경
- Colab T4 기준 SD 1.5, 512px, fp16. SDXL 은 메모리 부족
- 추가 패키지는 여기에 기록:

## VisA PCB A팀 생성 모델 비교 (33장 실험)

[1~3단계 실행 코드와 실측 결과](visa_pcb/README.md) · [1~6단계 전체 계획](visa_pcb/PLAN.md)

이 실험은 자체 분할(PCB당 gen_train 불량 33장, seed 7017)을 쓴다. 이 분할은 공유 시험셋과 겹치므로 **여기서 만든 결과물(`outputs/`)은 K장 실험에 넘기지 않는다.** 생성 방법끼리의 품질 비교에만 쓴다.


## B팀 전달용 K1_seed0 결과

Google Drive의 `내 드라이브/DL_TP/runs/shared_30e32db/Dataset/generated/` 아래에 두 결과가 있다.

| 방법 | Drive 하위 폴더 | GitHub manifest |
|---|---|---|
| Copy-paste | [Copy-paste 결과 폴더](https://drive.google.com/drive/folders/1iq6_Do7gF-LdFMsdR8yTaWZt2MD13dd5) | [manifest.csv](../Dataset/generated/copypaste/K1_seed0/manifest.csv) |
| cTransGAN | [cTransGAN 결과 폴더](https://drive.google.com/drive/folders/1f-NIgH42YJ3bB72eTGDdtvYpmS42AOOn) | [manifest.csv](../Dataset/generated/ctransgan/K1_seed0/manifest.csv) |

각 폴더에 생성 이미지와 마스크 480쌍이 있다. 링크가 있는 모든 사용자가 뷰어로 열 수 있다.
학습할 때 각 폴더의 `images/`, `masks/`를 레포의 `Dataset/generated/{방법}/K1_seed0/` 아래에 두면 manifest의 상대 경로와 일치한다.
