# Anomaly Detection 팀

같은 분할표로 K=0(비지도) ~ K=30(지도) 를 학습하고, 생성 데이터를 섞었을 때 지표가
실제 불량 몇 장분에 해당하는지(등가 샘플 수) 측정한다.

## 실험 축
- 비지도 (K=0): PatchCore, MLP+KNN 등. `train_normal.csv` 만 사용
- 지도 (K>0): `train_normal.csv` + `train_K{k}_seed{s}.csv`
- 지도 + 생성: 위에 `Dataset/generated/{방법}/K{k}_seed{s}/manifest.csv` 추가
- K ∈ {1,3,5,10,20,30}, seed ∈ {0,1,2}, 카테고리별 학습 권장 (메모리)

## 지표 (`test.csv` 고정)
- 픽셀 IoU
- 과검률 1% 고정 시 미검률
- 이미지 AUROC (보조)

## 설정
- `configs/base.yaml` 하나만 두고 K / seed / 생성 방법은 CLI 인자로 넘긴다
- 경로는 프로젝트 루트 기준 상대경로, yaml 안에 `../` 쓰지 않는다

## 결과 기록 `results/results.csv` (append-only)
```
method, category, K, seed, extra_data, iou, miss_rate_at_fpr1, auroc, n_generated, git_commit, timestamp
```
- `extra_data` 는 생성 방법 이름 또는 none
- 생성 데이터를 쓴 경우 기록 전 누수 검사(생성 source 가 test 에 없는지) 통과 필수

## 금지
- `test.csv` 로 학습·하이퍼파라미터 선택 금지
- 추가 패키지는 여기에 기록: 
