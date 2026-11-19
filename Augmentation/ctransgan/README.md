# ctransgan — Conditional TransGAN (K장 실험용 축소 구현)

Wang et al., *Conditional TransGAN-Based Data Augmentation for PCB Electronic Component Inspection*,
Computational Intelligence and Neuroscience, 2023 ([DOI 10.1155/2023/2024237](https://onlinelibrary.wiley.com/doi/10.1155/2023/2024237))
의 구조(분류기 특징을 조건으로 쓰는 Transformer 생성기·판별기)를 축소해서, K장 결함을 256×256 크롭으로 그리고
정상 바탕의 **결함 자리만 다시 그리도록** 맞춘 생성기.
조건 (K, seed)마다 PCB별로 그 PCB의 K장(종류당 K장)으로만 새로 학습한다. GPU 없이 CPU로 돈다.

## 서버 실험 반영 (11/19)

- VisA A100 실행은 `generate_diverse.py`에 `--diverse-scratch`를 적용했다. K1·K2·K4, seed 0 각각 480쌍을 생성했고 공용 누수 검사를 통과했다.
- 생성 패치와 마스크에 같은 길이·폭·각도 변환을 적용한다. 기판 안 배치와 변형 조건을 만족한 scratch는 조건별 120장 중 K1 109장, K2 108장, K4 103장이다. 나머지는 기존 모양을 사용한다.
- bent·missing·melt의 형태 변형은 이번 서버 실행에 추가하지 않았다. 아래 11/15 CPU 검수 수치는 당시 결과이며 새 GPU 결과의 검수 수치로 사용하지 않는다.
- 서버 생성물은 `/home/junho/ctransgan_runs/diverse_full_u9lv51T7/`에 있다. 기존 `review/K*_seed0.png`는 11/15 결과다.
- SMT 비교는 [smt_ctransgan](../smt_ctransgan/README.md)에서 실행한다. [최종 결과표](../../AnomalyDetection/results/smt/ctransgan_summary.md)는 pre와 solder K1·K4의 개선, solder K16의 악화를 모두 포함한다.

```bash
cd Augmentation
python -m ctransgan.generate_diverse --k 2 --seed 0 --diverse-scratch --device cuda --data-root /path/to/VisA_PCB --out /home/junho/ctransgan_runs/new_run
```

## 실행
```bash
cd Augmentation
pip install -r ctransgan/requirements.txt                     # torch, scipy (GPU 없어도 됨)
python -m ctransgan.generate --k 2 --seed 0                   # 조건 하나 (PCB 3개 × 종류 4개 × 40장 = 480장)
python -m ctransgan.generate --all                            # K {1,2,4} × seed 0
bash ctransgan/run_all.sh                                     # 위와 같음 + 로그(run_K*_seed0.log) + 확인용 시트
python -m ctransgan.review --k 2 --seed 0                     # 확인용 시트 → ctransgan/review/K2_seed0.png
```
- 원본이 `Dataset/visa/raw`에 없으면 `--data-root /경로/VisA_PCB`를 붙인다.
- 이미 끝난 조건은 건너뛴다. 다시 하려면 `--force`. 짧은 시험은 `--pcb pcb1 --per-type 4 --out /tmp/pilot`.
- GPU가 있으면 자동으로 쓴다(`--device cpu`로 고정 가능).

## 하는 일
| 단계 | 내용 |
|---|---|
| 다시 그릴 영역 | melt·scratch는 결함 마스크. **bent·missing은 마스크 + 원래 핀·부품 자리**를 6px 넓힌 것: `common/kshot.py`의 `change_region`(sd_lora v2와 같은 함수)에, 그 주변 30px 안에서 임계값을 낮춰(30) 구한 차이를 closing·opening 한 것을 더한다(`structure_region`). VisA 주석은 휜 핀 토막·빠진 자리만 덮어서 마스크만 다시 그리면 바탕의 곧은 핀이 남아 "다리가 하나 더" 생기고, 어둡게 찍힌 핀은 `change_region`만으로는 반만 잡힌다. 정답 마스크(`masks/`)는 VisA 주석 모양 그대로 |
| 크롭 | K장 결함마다 다시 그릴 영역 + 12px 여백(최소 64px) 정사각형 크롭을 256×256으로 맞춘다. 크롭이 320px보다 크면 떨어진 덩어리별로(최대 4개) 나눠 각각 256으로 그린다 |
| 조건 임베딩 | Transformer 분류기(4×4 패치 토큰, 폭 128, 깊이 2)를 결함 크롭 32×32로 60회 학습 → 분류층 직전 특징의 종류별 평균 |
| 생성기 | 잡음(128) + 조건 임베딩 → 8×8 토큰(폭 1024) → pixel shuffle → 16×16(폭 256, 블록 2) → pixel shuffle → 32×32(폭 64, 블록 1) → 토큰마다 겹치는 16×16 픽셀 패치로 펼쳐 256×256 RGB (`model.py`) |
| 판별기 | 조건 임베딩을 추가 토큰으로 받는 Transformer. 생성 결과를 기판 정렬한 정상 바탕 크롭(결함마다 3장)에 다시 그릴 영역만 끼워 넣어 32×32로 줄인 것을 본다 |
| 손실 | 적대(softplus) ×0.25 + 다시 그릴 영역(+12px) L1 ×35 + 경계 L1 ×14 + 고주파(3·7px) L1 ×18 + 크롭 전체 L1 ×6. 생성 크롭 1개당 125 업데이트(batch 4, Adam, 생성기 5e-4 → cosine, 판별기 1e-4) |
| 생성 | 결함마다 고정 잠재 벡터(anchor) + 잡음 0.12. K장을 돌아가며 종류별 40장. 장마다 train_normal 5장을 뽑아 결함 주변이 잘 맞는 순으로 보다가, 다시 그릴 영역을 생성 크롭이 95% 이상 덮는 첫 바탕을 쓴다(같은 결함에 같은 바탕은 두 번 쓰지 않는다). 학습 바탕 과반에서 다시 그릴 영역이 아니었던 곳(+12px 밖)은 그리지 않고 바탕 그대로 둔다 |
| 위치 | `common/kshot.py`의 기판 정렬(평행이동 + PCB1 180°) 뒤, melt·scratch는 ± 30px jitter(copy-paste·sd_lora와 같음). **bent·missing은 jitter 대신 ±6px 안에서 바탕의 부품 자리에 정밀하게 맞춘다**(`refine.py`). 다시 그릴 영역에 핀·기판 선이 들어가서 몇 px만 어긋나도 이중으로 보이기 때문 |
| 붙이기 | 다시 그릴 영역만 원본 해상도 바탕에 붙인다(2–3px 넓혀 같은 폭으로 페더). bent·missing은 영역 주변에서 잰 바탕과 결함 원본의 밝기 차이를 영역 안으로 부드럽게 퍼뜨려 더한다(±25 한도). JPEG q75 4:2:0 |
| 결과 | `Dataset/generated/ctransgan/K{k}_seed{s}/` (형식은 [Augmentation/README.md](../README.md)), 끝에 누수 검사 자동 |

manifest `condition`: 기판 정렬값(`align`), 실제 이동량(`applied_shift_xy`), 정밀 정렬 이동(`local_refinement_xy`) 또는 jitter(`offset_xy`),
붙인 창(`crop_boxes`), 마스크 면적(`mask_area`), 다시 그린 면적(`paint_px`)과 그중 생성 크롭이 덮은 비율(`painted_frac`),
밝기 보정 평균(`color_offset`), 바탕 주변 오차(`base_context_mae`)와 몇 번째 후보였는지(`base_rank`), 학습 설정.
옵션: `--no-structure`(bent·missing도 마스크 안만 다시 그림), `--no-color-match`(밝기 보정 끔), `--res 128`.

## 원논문 대비 바꾼 점
| 항목 | 논문 | 여기 |
|---|---|---|
| 데이터 | 부품 bbox ROI, 수백 장 | VisA PCB1–3, 종류당 K장(1/2/4)의 결함 마스크 |
| 출력 | 32×32 ROI 패치 | 256×256 결함 크롭(마지막 토큰을 픽셀 패치로 펼침)을 그려, 정상 바탕의 다시 그릴 영역에만 붙임 |
| 조건 | 분류기 특징 | 같음(종류별 평균 특징). G는 잡음에 이어 붙이고 D는 추가 토큰으로 받음 |
| 규모 | G 1024→256→64, 블록 5/4/2, D 깊이 7, 240 epoch | G 1024→256→64, 블록 0/2/1, D 깊이 2, 크롭당 125 업데이트 |
| 손실 | 적대 손실 | 적대 + 결함마다 고정한 잠재 벡터에 대한 재구성(L1·경계·고주파). K장만으로 적대 손실만 쓰면 무지개색 노이즈만 나온다(32×32, 4000 업데이트로 확인) |
| 정상 바탕 | 쓰지 않음 | D가 보는 합성(바탕 + 생성 영역), 바탕 선택, 위치 맞춤, 밝기 보정에 씀 |

## 소요 시간 (CPU, GPU 없음)
생성 크롭 수에 비례한다(크롭 1개당 125 업데이트, 1스레드에서 업데이트당 약 0.5초). 생성은 PCB당 6–9분.
11/15 실행은 조건 두 개를 동시에(각 1스레드, `--threads 1`) 돌렸고 중간에 검수도 같이 돌아서, 아래 시간은 단독 실행보다 길다.
로그: `run_K{1,2,4}_seed0.log`.

| 조건 | PCB별 생성 크롭 수 | PCB별 학습(초) | PCB별 생성(초) | 합계 | 누수 검사 |
|---|---|---|---|---|---|
| K1_seed0 | 6 / 5 / 4 | 476 / 381 / 309 | 404 / 337 / 444 | 39분 | PASS 480행 |
| K2_seed0 | 10 / 9 / 8 | 1258 / 685 / 631 | 698 / 518 / 542 | 72분 | PASS 480행 |
| K4_seed0 | 18 / 17 / 18 | 1401 / 1894 / 1405 | 472 / 457 / 409 | 101분 | PASS 480행 |

## 검수 (11/15, 1,440장 전수)
| 항목 | 결과 |
|---|---|
| 형식 | 전부 JPEG, 원본과 같은 양자화표(q75), 바탕과 같은 해상도. 마스크 값은 0과 종류 ID만, 면적 = manifest `mask_area`. PCB·종류별 40장, 파일 이름 중복 없음 |
| 누수 | `check_manifest` PASS (소스는 train_K, 바탕은 train_normal만) |
| 다시 그린 영역 밖 변화 | 0장 (영역 10px 밖에서 바탕과 30 넘게 다른 픽셀 없음) |
| 결함이 보이는가 | 마스크 안에서 (생성–바탕 차이)/(실제 불량–바탕 차이) 중앙값 0.96–0.99, 0.4 미만 0장 |
| 선명도 | 마스크 안 라플라시안 표준편차, 생성/실제 불량의 중앙값: bent 0.67 · missing 0.69 · melt 1.48 · scratch 1.44 (v2는 0.24 · 0.24 · 0.37 · 0.45). K별 bent/missing: K1 0.66/0.66, K2 0.72/0.72, K4 0.75/0.76 |
| 실제 불량과의 차이 | 마스크 안 (생성–실제 차이)/(실제–바탕 차이) 중앙값 0.07–0.15. K장 결함을 거의 그대로 다시 그린다 |
| 원래 핀 잔존 (bent·missing 720장) | 원래 핀·부품 자리의 절반 넘게 바탕 그대로인 장 0장, 30% 넘는 장 7장 (v1은 전부 남음) |
| 다시 그린 배경 | 질감(라플라시안 표준편차)이 바로 옆 바탕의 0.87–0.89 |
| 생성 크롭이 덮은 비율 | 다시 그릴 영역의 90% 미만만 그린 장 12장 (pcb1 missing 8, pcb1 bent 2, pcb2 missing 2). 나머지 부분은 바탕 그대로 |
| 중복 | 같은 (결함 소스, 바탕) 쌍 0개. 82%는 첫 번째 바탕 후보를 그대로 씀 |

알려진 한계 (눈으로 확인한 것과 수치):
- **bent·missing의 질감은 실제보다 조금 부드럽다**(선명도 0.66–0.76). 특히 pcb1 missing(핀이 빠진 자리의 카펫 질감)과 pcb3 bent의 LED. melt·scratch는 실제와 같은 수준이다.
- **pcb3의 LED(bent·missing):** 투명한 LED 몸통은 바탕과 차이가 작아 다 잡히지 않아서, 원래 LED 윤곽이 희미하게 남는 장이 있다.
- **pcb2 missing 소스 054·059:** 이 불량 사진은 기판이 살짝 기울어 있어 기판 정렬 점수가 낮다(0.23–0.43, K2·K4의 38장). ±6px 정밀 정렬로 핀 자리는 맞췄지만 주변이 몇 px 어긋날 수 있다.
- **큰 결함:** 한 덩어리가 320px보다 큰 결함(pcb1의 큰 부품 missing, 긴 scratch 등)은 256으로 줄여 그리므로 그만큼 덜 선명하다.
- K장이 적어서 생성 결과는 K장 결함을 바탕에 맞게 다시 그린 것이다. 11/15 버전의 다양성은 바탕(장마다 다른 정상)·위치(melt·scratch jitter)·잡음에서 나온다. 11/19 서버 실행에서는 위 스크래치 형태 변형을 추가했다.
- 마스크는 VisA 주석 모양을 옮긴 것이다. bent·missing은 마스크보다 넓은 영역(원래 핀 자리)이 바뀌어 있다.

## 이전 버전 (11/15에 차례로 교체)
- v1: 32×32 생성 결과를 128로 키워 conv로 다듬고 마스크 안만 붙임 → bent에 원래 핀이 그대로 남고("다리가 하나 더") 흐림.
- v2: bent·missing에서 원래 핀 자리까지 다시 그림 → 핀은 지워졌지만 여전히 흐림(선명도가 실제 불량의 약 27%), 지운 자리에 회색 띠.
- v3(지금): 생성기가 256×256을 직접 그리고, bent·missing 정밀 정렬·밝기 보정, 큰 결함은 덩어리별 크롭.
- 옛 `visa_pcb/` 구현(d66851e에서 삭제)으로 만든 K1 결과는 Drive에 있다(Augmentation/README.md 마지막 절).
