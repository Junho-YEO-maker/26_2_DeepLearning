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

## 방법
- `copypaste` : 불량 패치를 정상 이미지에 붙여넣기 (기준선). 모든 K × seed 0–4. **K장 실험용 코드 완료**
- `ctransgan` : **생성 모델 1단계 (AD 팀 합의, 9절 ⑤)**. PCB1–3 × K {1, 2, 4} × seed 0, 조건 9개마다 해당 K장으로 새로 학습한다(33장 체크포인트 재사용 금지). T4에서 조건당 약 13분. K=1을 먼저 돌려 품질을 확인한다.
- `sd_lora`   : SD inpainting + LoRA. **선택** — 1단계를 마치고 시간이 남으면 추가한다.
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
- 결함은 원래 있던 자리 ±30px에 붙인다(`--placement same`, 기본). 같은 PCB는 기판 위치가 거의 같아서 결함이 기판 밖 배경에 떨어지지 않는다. 2단계 기준선처럼 임의 위치·회전으로 붙이려면 `--placement random`.
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
