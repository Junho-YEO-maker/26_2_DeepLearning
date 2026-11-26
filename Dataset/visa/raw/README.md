# VisA 원본 이미지 (git 미포함)

이 폴더에는 VisA 데이터셋의 PCB1–3 원본 이미지와 마스크가 들어간다.
용량이 커서(PCB1–3만 약 700MB, 전체 tar 1.9GB) 저장소에는 넣지 않았다. 팀 내부에서는 Google Drive로 공유했다.
분할표(`Dataset/visa/splits/`, `Dataset/visa/splits_official/`)와 생성 이미지 manifest는 git에 들어 있으므로,
아래 방법으로 원본만 받으면 실험을 그대로 재현할 수 있다.

- 출처: Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV 2022
  (https://github.com/amazon-science/spot-diff)
- 라이선스: CC BY 4.0
- 사용 범주: `pcb1`, `pcb2`, `pcb3` (`pcb4`는 제외. 이유는 `AnomalyDetection/README.md` 2.1절)

## 받는 방법

프로젝트 루트에서 실행한다.

```bash
curl -L -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
# 크기 확인: 1,929,840,640 bytes

# 1) 분할표(Dataset/visa/splits)가 쓰는 경로
tar xf VisA_20220922.tar -C Dataset/visa/raw pcb1 pcb2 pcb3

# 2) 공식 분할 평가(AnomalyDetection/ad/official_eval.py 등)가 쓰는 경로
mkdir -p Dataset/visa/VisA_official/VisA_PCB
tar xf VisA_20220922.tar -C Dataset/visa/VisA_official/VisA_PCB pcb1 pcb2 pcb3 split_csv
```

두 경로의 내용은 같다(2는 공식 `split_csv/`가 더 있을 뿐). 디스크를 아끼려면 2만 풀고
`ln -s ../VisA_official/VisA_PCB/pcb1 Dataset/visa/raw/pcb1` 처럼 pcb1–3을 링크해도 된다.
`/home/junho/...`가 들어간 실행 스크립트는 `DATA_ROOT` 환경변수로 2의 절대경로를 넘기면 된다.

## 풀고 난 뒤 모양

```
Dataset/visa/raw/
├─ pcb1/
│  ├─ Data/Images/{Normal,Anomaly}/*.JPG
│  ├─ Data/Masks/Anomaly/*.png
│  └─ image_anno.csv
├─ pcb2/  (같은 구조)
└─ pcb3/  (같은 구조)
```

## 확인

분할표의 `image_sha256`과 실제 파일을 대조한다. `missing 0, bad 0`이 나오면 된다.

```bash
python - <<'EOF'
import csv, glob, hashlib, os
miss = bad = n = 0
for f in glob.glob('Dataset/visa/splits/*.csv'):
    for r in csv.DictReader(open(f, encoding='utf-8')):
        n += 1
        if not os.path.exists(r['image']): miss += 1; continue
        if hashlib.sha256(open(r['image'], 'rb').read()).hexdigest() != r['image_sha256']: bad += 1
print('rows', n, 'missing', miss, 'bad', bad)
EOF
```
