# sd_lora — SD 인페인팅 + LoRA

Stable Diffusion 1.5 인페인팅 모델에 **K장만으로** LoRA를 학습해서 결함을 그려 넣는다.
조건 (K, seed)마다 PCB별 LoRA 1개(결함 4종류는 프롬프트로 구분), 9개 조건이면 LoRA 27개.

## 실행 (Colab, 권장)
[colab/sd_lora_kshot.ipynb](../../colab/sd_lora_kshot.ipynb)를 Colab에서 열고 GPU 런타임(T4 또는 A100) → **모두 실행**.
LoRA와 결과는 Drive `MyDrive/DL_TP/runs/sd_lora/`에 저장되고, 세션이 끊기면 다시 모두 실행해서 이어 간다.
T4에서는 fp16 + gradient checkpointing으로 자동 전환된다(A100은 bf16).

## 실행 (Kaggle, Colab 할당량이 끝났을 때)
[kaggle/sd_lora_kshot.ipynb](../../kaggle/sd_lora_kshot.ipynb)를 Kaggle에 Import → Settings에서 **GPU T4 x2**, **Internet On** → **Run All**
(브라우저를 닫아도 돌게 하려면 Save Version → Save & Run All).
- 원본은 VisA 공식 tar에서 pcb1–3만 받아 온다. 데이터를 따로 올릴 필요가 없고, 해시는 분할표와 대조한다.
- GPU 2장이면 PCB를 나눠 동시에 학습한다(pcb1·pcb3 / pcb2).
- 결과는 `/kaggle/working/sd_lora/`, 조건별 zip `sd_lora_K{k}_seed{s}.zip`을 Output 탭에서 받는다.
- 이전 LoRA(이 노트북의 이전 Output이나, Colab Drive의 `lora/` 폴더를 Kaggle Dataset으로 올린 것)를 Add Input으로 붙이면 학습을 건너뛴다.

## 실행 (GPU 서버)
```bash
cd Augmentation
pip install -r sd_lora/requirements.txt          # torch는 서버 CUDA에 맞게 먼저
python -m sd_lora.train --k 1 --seed 0 --pcb pcb1 --steps 100   # ① 짧은 시험: 시간·메모리 확인
python -m sd_lora.train --k 1 --seed 0           # ② K=1 본학습 (pcb1–3)
python -m sd_lora.generate --k 1 --seed 0        # ③ 생성 480장 → 결과 확인 후
bash sd_lora/run_all.sh                          # ④ K {1,2,4} × seed 0 전체
```
- 원본이 `Dataset/visa/raw`에 없으면 `--data-root /경로/VisA_PCB`를 붙인다.
- 이미 끝난 조건은 건너뛴다. 다시 하려면 `--force`.

## 하는 일
| 단계 | 내용 |
|---|---|
| 학습 | 그 PCB의 K장 불량(종류당 K장)에서 결함 주변 512×512 크롭을 매번 다른 위치로 자르고, 결함 영역(8px 넓힘)을 가린 뒤 다시 그리게 학습. 프롬프트 `... sks printed circuit board with a {bent pin / melted component / scratch / missing part} defect` |
| 생성 | K장 중 하나의 결함 마스크를 train_normal 바탕에 기판 정렬(평행이동 + PCB1의 180° 회전) 후 같은 부품 자리 ± 종류별 jitter(bent·missing 5px, melt·scratch 30px)로 두고(copy-paste와 같은 함수), 주변 512 크롭을 인페인팅한 뒤 **마스크 안만** 원본 해상도 바탕에 붙인다 |
| 결과 | `Dataset/generated/sd_lora/K{k}_seed{s}/` (형식은 [Augmentation/README.md](../README.md)) |

- 결함이 512보다 크면(pcb1 missing 등 5장) 크롭을 키운 뒤 512로 줄여 넣는다.
- 기본값: 1000 step, batch 4, lr 1e-4, rank 8, 샘플링 30 step, guidance 7.5. ①에서 결과가 약하면 step부터 늘린다.
- LoRA 가중치: `sd_lora/runs/K{k}_seed{s}/{pcb}/` (git 제외). `train.json`에 학습 소스 K장 목록과 시간 기록.
- 베이스 모델: `stable-diffusion-v1-5/stable-diffusion-inpainting` (Hugging Face에서 자동 다운로드, 약 4GB).

## 확인할 것
- 생성 이미지 몇 장을 눈으로 보고, 결함이 마스크 안에 실제로 그려졌는지(빈 기판이 아닌지) 확인.
- `python -c "from common.kshot import check_manifest; print(check_manifest('../Dataset/generated/sd_lora/K1_seed0/manifest.csv'))"` 로 누수 검사 (generate가 끝에 자동으로도 함).
