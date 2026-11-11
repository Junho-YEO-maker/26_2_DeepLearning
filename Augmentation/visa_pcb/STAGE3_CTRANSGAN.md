# 3단계 cTransGAN 구현과 실행 상태

논문은 Wang et al., Conditional TransGAN-Based Data Augmentation for PCB Electronic Component Inspection, Computational Intelligence and Neuroscience, 2023, DOI 10.1155/2023/2024237입니다. 요청에서 적힌 Sensors 표기를 이 문서에서 정정합니다.

저자 저장소의 cTransGAN.ipynb를 조사했습니다. Generator 초기화 시 EncoderBlock의 필수 인자가 누락되어 TypeError가 발생했고, PixelShuffle.forward와 학습 데이터, 일부 변수도 미완성입니다. 커밋과 노트북 해시, 실제 초기화 오류는 reports/stage3_source_audit.json에 있습니다. 공개 저장소에 별도 LICENSE 파일은 확인되지 않았습니다. 이번 src 코드는 논문의 방법을 바탕으로 새로 작성했고 저자 소스와 데이터를 패키지에 복사하지 않았습니다.

- 논문과 공개 코드 링크는 다음과 같습니다. https://onlinelibrary.wiley.com/doi/10.1155/2023/2024237
- 저자 저장소의 조사 커밋은 93ea4e1f69e5d5929cd9f2ae08d5bff7138e19bb입니다. https://github.com/long-deep/pcb-detect

## 구현한 흐름

1. PCB별 고정 gen_train 불량 33장을 사용합니다. 각 마스크의 종류 ID와 4방향 연결 영역을 구분해 하나의 종류를 갖는 ROI를 만듭니다. 전체 이미지의 다중 라벨을 대표 라벨 하나로 바꾸지 않습니다.
2. 2픽셀 문맥을 포함한 RGB ROI를 32×32로 변환해 Transformer 분류기를 먼저 학습합니다.
3. 분류기 최종 분류층 직전 특징을 ROI별로 추출하고 종류별 평균을 조건 임베딩으로 저장합니다.
4. 잡음과 조건 임베딩을 연결한 생성기, 이미지 토큰과 조건 토큰을 연결한 판별기를 BCE 적대 손실로 학습합니다.
5. 생성한 32×32 패치를 원래 주석 위치의 gen_train 정상 이미지에 삽입합니다. 원본 마스크의 해당 연결 영역 안쪽을 블렌딩하고, 밖의 픽셀은 유지합니다. 결과를 긴 변 512픽셀과 원본 비율로 저장합니다.

기본 표본추출은 ROI별 균등 추출입니다. 연결 영역이 많은 이미지의 비중이 커질 수 있습니다. 한 이미지에서 얻은 여러 ROI는 같은 원본에 종속된 표본입니다. 진짜 불량 예산은 여전히 PCB별 33장입니다.

## 원논문 대비 변경점

| 항목 | 논문 또는 저자 공개 초안 | 이번 Colab 구현 |
|---|---|---|
| 데이터 | 부품 데이터와 DeepPCB, bbox 기반 ROI | VisA PCB 4종, 마스크 종류별 연결 영역 |
| 조건 | 분류기에서 추출한 클래스 특징 | ROI 특징의 종류별 평균, 이 원리를 유지 |
| 생성기 규모 | 공개 초안의 1024→256→64 폭, 5/4/2 블록 | 256→64→16 폭, 2/2/1 블록 |
| 분류기 특징 폭 | 공개 초안의 384 | 128 |
| 판별기 | 공개 초안의 깊이 7, 미완성 코드 | 4×4 패치 토큰, 깊이 3 |
| attention | TransGAN의 다단계/grid 기법을 참조 | 단계별 전체 attention, 두 번의 PixelShuffle |
| 조건 주입 | 이미지·잡음 벡터와 클래스 특징 연결 | G의 잡음 연결, D의 추가 조건 토큰 연결 |
| 학습 일정 | 논문 GAN 240 epochs, batch 64, lr 1e-4 | 분류기 200 업데이트, GAN 4,000 업데이트, lr 1e-4 |
| T4 배치 | 논문과 하드웨어가 다름 | micro batch 8 × 누적 2, fp16, gradient checkpointing |
| A100 배치 | 이번 프로젝트의 하드웨어 프로필 | micro batch 32 × 누적 1, 별도 checkpoint 경로 |
| 주석 | bbox를 사용한 패치 재삽입 | 마스크 연결 영역 삽입과 이진 support mask |
| 위치 | 원래 샘플의 ROI 위치 | 해당 PCB의 다른 gen_train 정상에 같은 좌표 사용 |

32×32로 학습하는 ROI 자체에는 정사각형 변환이 있습니다. 전체 출력의 종횡비와 변환 내역은 meta.csv에 보존합니다. 4,000 업데이트는 사전 지정한 실행 예산이며, heldout으로 선택한 값은 없습니다. 좁은 전체 attention 구조는 이 구현의 축소 변경점입니다.

## 출력 마스크의 해석

이 모델은 RGB 결함 패치를 학습합니다. 출력 PNG 마스크는 실제로 합성한 영역을 가리킵니다. 생성된 내용이 그 종류의 결함을 정확하게 표현하는지는 별도 정성·B팀 평가가 필요합니다. condition에 semantic_mask_verified=false를 기록했습니다. 이 영역을 사람이 검증한 의미적 분할 정답으로 해석하면 한계가 있습니다.

이미지와 마스크는 outputs/ctransgan/{pcb}/{images,masks}/, 메타데이터는 meta.csv에 저장합니다. 열 이름은 2단계 규약을 유지합니다. source_anomaly/source_mask는 모양과 위치를 제공한 gen_train 원본이고, condition은 클래스 임베딩, 연결 영역, 좌표와 체크포인트 해시를 포함합니다.

## 실행과 재개

공통 설정과 원본 다운로드 셀을 실행한 뒤 notebooks/A_augmentation.ipynb의 3단계 셀을 실행합니다. 터미널에서는 다음 순서입니다.

```bash
python -m tests.smoke_stage3
python -m src.train.ctransgan --config configs/ctransgan_t4.yaml
python -m src.generate.run --config configs/ctransgan_t4.yaml
python -m src.metrics.evaluate --config configs/ctransgan_t4.yaml
python -m src.metrics.stage3_summary
```

A100이면 config 인자를 configs/ctransgan_a100.yaml로 변경합니다. 두 프로필의 배치 크기가 다르므로 서로 다른 학습 실행으로 기록합니다. 같은 실행을 재개할 때 설정을 유지해야 합니다.

체크포인트는 classifier/G/D 가중치, 옵티마이저 상태, 클래스 임베딩, 진행 단계, AMP scaler, Python/NumPy/PyTorch/CUDA 난수 상태, 손실과 시간을 저장합니다. 입력 데이터와 학습 코드·설정 해시가 바뀌면 재개를 중단합니다. 각 분류기 또는 G/D 업데이트가 완료된 경계에서 저장하며, 세션이 중간에 끊기면 마지막 저장 지점 이후 업데이트를 재실행합니다. 기본 저장 간격은 100 업데이트입니다.

학습 시간은 분류기/GAN의 업데이트 구간을 따로 합산하며, 원본 읽기·초기 모델 생성·체크포인트 쓰기 시간은 제외합니다. GPU peak는 PyTorch max_memory_allocated 값이며 재개 이전 최고값과 합쳐 최대값을 보존합니다. 생성 GPU peak는 생성 시 따로 기록합니다.

누수 검사는 기존 A팀 접근 게이트를 사용합니다. 생성은 학습 계획이 끝난 체크포인트를 요구합니다. 평가 참조는 해당 PCB gen_train 불량 33장이고, 지표 설정은 2단계와 같습니다. 지표와 그림은 results/ctransgan 및 figures/ctransgan에 저장해 모델별 경로를 유지합니다. 비교표의 PSNR/SSIM 열에도 원본 정상과의 구조 보존이라는 주석을 넣었습니다.

## 이번 실행 상태

CPU에서 더미 end-to-end 검증을 완료했습니다. 4종 모두 분류기·GAN 업데이트, 생성, 재개가 동작했고, 재개 후 파라미터와 손실은 연속 학습과 일치했습니다. 사전학습 특징 기반 평가까지 더미 데이터로 연결했습니다.

실제 데이터에서는 PCB별 33장의 마스크로 ROI를 추출하고 T4 프로필과 같은 네트워크 크기로 CPU 1회 분류기·1회 GAN 업데이트를 실행했습니다. 이는 실제 입력에 대한 기능 검사입니다. 임시 체크포인트는 폐기했고 B팀용 이미지는 내보내지 않았습니다.

Colab Tesla T4 본학습, PCB별 500장 생성 및 평가를 완료했습니다. [실측 보고서](reports/ctransgan_t4_results.md)와 [실측 CSV](results/ctransgan_t4_summary.csv)에 설정과 결과를 기록했습니다. `results/stage3_comparison.csv`는 재실행 시 작성됩니다. A100 실행은 미실행입니다.
