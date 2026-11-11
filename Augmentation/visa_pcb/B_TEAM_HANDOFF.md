# B팀 데이터 전달

각 결과 ZIP을 같은 폴더에 풀면 DL_TP 아래에 PCB별 결과가 배치됩니다.

```
DL_TP/outputs/copy_paste/{pcb}/
  images/000000.png ... 000499.png
  masks/000000.png ... 000499.png
  meta.csv
```

이미지 500장은 모두 합성 불량으로 분류 라벨을 1로 둡니다. 같은 이름의 마스크는 L 모드 PNG이며 0은 배경, 255는 합성 결함 영역입니다. CSV는 UTF-8 BOM으로 저장했습니다. `pandas.read_csv` 또는 `encoding='utf-8-sig'`로 읽으세요.

| 컬럼 | 값 |
|---|---|
| filename | 이미지와 마스크의 공통 파일명 |
| source_normal | VisA 기준 원본 정상 상대경로 |
| source_anomaly, source_mask | 결함 제공 이미지와 마스크의 상대경로 |
| condition | 회전, 위치, 패치 배율, 결함 면적과 종류 등 JSON |
| seed, sample_index | seed 7017과 PCB 내 샘플 번호 |
| resize, original_width/height, output_width/height | 리사이즈 방법과 크기 |
| image_sha256, mask_sha256 | 파일 무결성 검사값 |

저장 해상도는 PCB1·2 512×390, PCB3 512×315, PCB4 512×416입니다. 긴 변을 512로 줄이고 종횡비를 정수 반올림 범위에서 유지했습니다. 원본 정상과 실제 불량에도 분류기의 동일한 입력 전처리를 적용할 수 있습니다.

코드 ZIP의 `splits/{pcb}_split.csv`가 진짜 데이터 분할 기준입니다. 합성에는 gen_train 정상·불량만 사용했습니다. B팀 평가에는 heldout 정상·불량을 쓰고, 학습·검증용 진짜 데이터는 gen_train 안에서 선택합니다.

현재 합성물은 PCB별 진짜 불량 33장을 제공 풀로 사용했습니다. 총 진짜 불량 예산을 k장으로 제한하는 실험에는 해당 k장만 제공 풀에 넣고 합성해야 합니다. 현재 파일로 k<33을 실험하면 분류기가 직접 보는 진짜 불량 수를 바꾼 조건입니다.

```python
import csv
from pathlib import Path
from PIL import Image

base = Path("DL_TP/outputs/copy_paste/pcb1")
with (base / "meta.csv").open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))
image = Image.open(base / "images" / rows[0]["filename"]).convert("RGB")
mask = Image.open(base / "masks" / rows[0]["filename"])
label = 1
```

데이터 출처와 변형 사실은 각 PCB ZIP의 ATTRIBUTION.txt에 기록했습니다.

## cTransGAN 출력

GPU 학습과 생성 후에는 같은 열 구조로 outputs/ctransgan/{pcb}에 저장합니다. 해당 마스크는 삽입 영역이며 condition에 의미적 검증 여부를 기록합니다. 이번 패키지의 cTransGAN 본학습 및 B팀용 500장 출력은 미실행입니다.
