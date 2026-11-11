# VisA PCB 이상 데이터 증강

PCB 종류별 고정 gen_train 불량 33장으로 생성 방법을 비교하는 A팀 실험입니다. VisA의 pcb1~pcb4를 사용하고 seed를 7017로 고정합니다. 전체 작업 순서와 B팀 K장 실험 연계 조건은 [실험 계획](PLAN.md)에 있습니다.

| 단계 | 내용 | 검증 결과 |
|---|---|---|
| 1 | 원본 라벨·마스크 검사, 층화 분할, heldout 누수 방지 | [데이터 검사](reports/data_inspection.md), [분할표](splits) |
| 2 | 마스크 복사-붙여넣기, PCB별 500장 생성, 공통 평가 | [방법](STAGE2_METHOD.md), [실측 CSV](results/baseline_summary.csv) |
| 3 | Conditional TransGAN, PCB별 500장 생성과 Colab T4 평가 | [방법 및 원논문 대비 변경점](STAGE3_CTRANSGAN.md), [실측 결과](reports/ctransgan_t4_results.md), [실측 CSV](results/ctransgan_t4_summary.csv) |

## Colab 실행

[노트북](notebooks/A_augmentation.ipynb)을 Colab에서 열고 GPU 런타임을 선택한 후 순서대로 실행합니다. 코드, 체크포인트 및 재실행 결과는 `MyDrive/DL_TP/runs/github_stage3/DL_TP`에 저장합니다. 이미 받은 VisA PCB 원본은 `MyDrive/DL_TP/data/VisA_stage3_20261111`에서 재사용합니다. 분할 해시, 원본과 마스크 해시, heldout 교집합 검사를 먼저 수행합니다.

로컬에서는 이 디렉터리를 작업 경로로 사용합니다. PyTorch/torchvision을 실행 환경에 맞게 설치한 뒤 아래 명령으로 소형 입력에서 파이프라인을 검증할 수 있습니다.

```bash
python -m pip install -r requirements-stage2.txt
python -m tests.smoke_stage1
python -m tests.smoke_stage2
python -m tests.smoke_stage3
```

`src.generate.run`은 `generate(pcb, n, seed)`를 호출합니다. 모델과 PCB마다 `outputs/{model}/{pcb}/{images,masks,meta.csv}`를 생성하며, 출력 비율과 원본 정상·불량 마스크·생성 조건·seed를 기록합니다. 실제 PNG, 비교 그림, 모델 체크포인트는 Drive의 기존 실행 폴더에 있습니다. [B팀 파일 규약](B_TEAM_HANDOFF.md)

## 원논문 대비 변경점

- VisA 클래스 ID 마스크의 연결 영역에서 PCB별 학습 패치를 만들었습니다. 전체 원본의 다중 불량 라벨은 그대로 보존합니다.
- cTransGAN의 클래스 특징 평균 임베딩과 Transformer 생성기·판별기를 유지하고, 폭·깊이 및 업데이트 수를 Colab용으로 줄였습니다.
- 32×32 결함 패치를 생성한 후 정상 이미지의 마스크 영역에 삽입하고, 원본 종횡비를 유지한 긴 변 512px PNG로 저장합니다. 출력 마스크는 합성 영역입니다.

## 출처 및 라이선스

- [VisA 데이터 및 클래스 매핑](https://github.com/amazon-science/spot-diff). 데이터 CC BY 4.0, 클래스 매핑 Apache-2.0. 라이선스는 `upstream/`에 있습니다.
- [Conditional TransGAN 논문](https://doi.org/10.1155/2023/2024237), Wang et al., 2023. [저자 저장소](https://github.com/long-deep/pcb-detect)는 조사에 참고했습니다. 논문 기반 독립 구현을 제공합니다.
- [SeaS](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_SeaS_Few-shot_Industrial_Anomaly_Image_Generation_with_Separation_and_Sharing_ICCV_2025_paper.pdf)의 고정 학습/평가 분리에서 착안해 VisA 다중 라벨 층화와 정상 80/20 분할을 적용했습니다.
