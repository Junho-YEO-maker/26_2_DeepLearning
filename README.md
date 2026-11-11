# 26_2_DeepLearning

불량 샘플 K장 시나리오에서 생성 데이터(copy-paste / SD LoRA inpaint)가
실제 불량 몇 장을 대신하는지 측정하는 과제. 데이터는 VisA pcb1~3 (pcb4 제외, AnomalyDetection/README.md 2.1절).

## 폴더 구조
```
26_2_DeepLearning/
├─ Dataset/                 # 두 팀이 공유하는 모든 데이터
│  ├─ visa/
│  │  ├─ raw/               # VisA 원본 pcb1~3. git 제외, Drive에만 존재
│  │  └─ splits/            # 한 번 만들고 고정. git 포함 (분할 seed 7017)
│  │     ├─ test.csv        # 고정 시험셋(불량 182 + 정상 911). 학습·생성 사용 금지
│  │     ├─ val_normal.csv  # 임계값 선택용 정상 (PCB당 200)
│  │     ├─ train_normal.csv  # 정상 학습셋, 생성 바탕 (PCB당 500)
│  │     ├─ pool.csv        # K장 후보 (종류별 10, pcb2 bent 8)
│  │     └─ train_K{1,2,4,8}_seed{0..4}.csv  # PCB별·종류별 K장
│  └─ generated/            # 생성 결과. 이미지는 Drive, manifest.csv만 git
│     └─ {방법}/K{k}_seed{s}/{images,masks,manifest.csv}
├─ Augmentation/            # 생성 팀 코드
├─ AnomalyDetection/        # 탐지 팀 코드
│  ├─ configs/
│  └─ results/results.csv
├─ scripts/
│  └─ make_splits.py        # 분할표 생성 (한 번만 실행)
└─ colab/                   # 노트북
```

## 규칙
- 경로는 전부 프로젝트 루트 기준 상대경로. csv 안의 경로도 같음.
- `test.csv`에 있는 이미지는 학습·생성 소스로 절대 쓰지 않는다.
- 생성 모델은 해당 `train_K{k}_seed{s}.csv` 안의 불량만 소스로 쓴다.
- 정상 바탕은 `train_normal.csv`에서만 가져온다. `val_normal.csv`도 학습·생성에 쓰지 않는다.
- `manifest.csv` 열: `image, mask, label, source_image, score, category, defect_type, base_normal_image, gen_seed, condition` (Augmentation/README.md)

## Colab
1. 레포 clone, Drive 마운트
2. `Dataset/visa/raw`, `Dataset/generated` 를 Drive 폴더로 심볼릭 링크
3. 세션마다 raw zip을 `/content`로 복사 후 해제 (Drive 직접 읽기는 느림)

## 분할표 만들기
```bash
python scripts/make_splits.py --raw Dataset/visa/raw --out Dataset/visa/splits
```
이미 만들어 git에 올렸으므로 다시 실행하지 않는다(`--force`는 팀 합의 후에만).
