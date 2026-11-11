# anomalydiffusion — AnomalyDiffusion (Hu et al., AAAI 2024)

불량 생성 전용 diffusion. [공식 코드](https://github.com/sjtuplayer/anomalydiffusion)(MIT)를 `upstream/`에 받아
그대로 돌리고, 입력·출력만 우리 실험에 맞게 바꿔 끼운다(`run.py`).

## 공식 코드를 그대로 못 쓰는 이유와 바꾼 점
| 공식 코드 | 문제 | 어댑터가 하는 일 |
|---|---|---|
| `test/<종류>/` 이미지 중 **앞쪽 1/3만** 학습 | K장을 넣으면 일부만 학습됨 | 진짜 이미지 뒤에 사본을 붙여 개수를 맞춤 → 정확히 진짜 K장만 학습 (사본은 공식 검증 자리만 채움) |
| 이미지 전체를 **256px**로 줄여 생성 | PCB 결함(이미지의 0.1–6%)이 거의 사라짐 | 결함 주변 512 크롭을 샘플로 넣고 512로 생성, 원본 해상도 바탕에 다시 붙임 |
| 마스크 생성 모델을 (샘플, 종류)마다 **30,001 step** 학습 | 우리 조건이면 수십 번 학습 | K장 결함 마스크를 종류별 jitter로 옮겨 씀 (copy-paste, sd_lora와 같은 규칙) |
| 학습 step 기본 1,000,000 | 너무 김 | `--steps` (기본 6000, K=1로 재고 정함) |

샘플 하나 = K장 중 결함 하나의 주변 크롭(`{pcb}-{종류}-{j}`). 모델은 조건마다 1개(12·24·48개 샘플을 한 번에 학습).

정상 바탕 크롭은 기판을 그 결함 이미지에 맞춰(평행이동 + 180°, copy-paste의 `board_transform`) 같은 부품 자리를 잘라 온다. PCB1은 바탕의 상당수가 180° 돌아가 있어서 정렬 없이 자르면 엉뚱한 부분이 나온다. export에서 같은 변환을 거꾸로 적용해 바탕 원본 방향으로 붙인다.

> **보류**: copy-paste·sd_lora 결과에서 생성 쪽 이득이 보일 때만 실행한다(가장 무겁고 불확실성이 큼).

## 설치 (A100 서버, 한 번)
```bash
cd Augmentation/anomalydiffusion
bash setup.sh        # 공식 코드 clone(커밋 기록), conda 환경 Anomalydiffusion(python 3.8, torch 1.12.1), LDM 체크포인트 다운로드
```

## 처음 실행 점검 (K=1로 반드시 먼저)
공식 코드는 GPU 없이 검증하지 못했다. 아래를 순서대로 확인한다.
```bash
cd Augmentation
python -m anomalydiffusion.run prepare  --k 1 --seed 0
python -m anomalydiffusion.run train    --k 1 --seed 0 --steps 200   # ① 짧게: 에러 없이 도는지, step당 시간
```
1. 학습이 끝나면 `anomalydiffusion/work/K1_seed0/logs/anomaly-checkpoints/checkpoints/`에 `embeddings.pt`, `spatial_encoder.pt`가 생겼는지 본다. 없으면 run.py가 멈추며 알려 준다. 저장 위치가 다르면 그 경로를 run.py의 `train()`에 맞춘다.
2. 로그에서 step당 시간을 보고 `--steps`를 정한다(조건 3개를 하루 안에 끝낼 수 있게).
3. 생성 1개 샘플만 시험: `python -m anomalydiffusion.run generate --k 1 --seed 0` 을 돌리다가 첫 샘플 폴더 `work/K1_seed0/generated_dataset/<샘플>/<종류>/image/`가 생기면 열어 본다. 결함이 마스크 자리에 그려졌는지 확인.
   - `size=256`을 못 찾는다는 오류가 나면 `--gen-res 256`으로 돌린다(그 경우 해상도가 절반).
4. 문제가 없으면 본학습: `python -m anomalydiffusion.run all --k 1 --seed 0 --force --steps <정한 값>`

## 본 실행
```bash
cd Augmentation
STEPS=<정한 값> bash anomalydiffusion/run_all.sh        # K {1,2,4} × seed 0
```
- 단계별: `prepare → train → generate → export` (`all`은 넷 다). 원본 위치가 다르면 `--data-root`.
- 결과: `Dataset/generated/anomalydiffusion/K{k}_seed{s}/` (형식은 [Augmentation/README.md](../README.md)), export가 끝에 누수 검사.
- 작업 폴더 `anomalydiffusion/work/`, 공식 코드 `upstream/`은 git 제외.
- 공식 코드가 현재 폴더 기준 경로(`name-anomaly.txt`, `logs/`, `generated_mask/` …)를 쓰므로 조건마다 `work/K{k}_seed{s}/`에서 따로 돌린다.

## 기록할 것
- `upstream_commit.txt` (setup.sh가 만듦): 어느 커밋으로 돌렸는지
- 조건별 학습 step, 시간 (`run_K*_seed0.log`)
- manifest `condition`의 `base_match_margin`: 생성 결과가 어느 정상 바탕에서 나왔는지 찾을 때의 확신 정도. 0에 가까운 행이 많으면 알려 줄 것
