# "합성 40장 = 실제 불량 몇 장" 실험 (11/25–11/26)

발표자료 v2의 RQ2(VisA)와 RQ3(SMT) "실제 몇 장" 슬라이드에 들어간 숫자를 만든 실행 스크립트, 분석 스크립트, 로그가 이 폴더에 있다.
모델과 지표는 팀 실험과 같다. 실험을 새로 설계하지 않았고, 비어 있던 칸(나누기, 실제 장수)만 채웠다.

## 무엇을 쟀나

실제 불량 K장에 합성 N장을 더한 성적을, **실제 불량만 쓴 곡선**에 놓고 "실제만으로 몇 장 쓴 것과 같은가"로 바꿔 읽는다.

- 실제만 곡선을 장수에 대해 "실제를 더 넣으면 나빠지지 않는다"는 조건으로 고르게 다듬은 뒤(PAVA, log2 K 위), 두 점 사이를 직선으로 잇는다.
- `<1`은 실제 최소 장수보다 나쁨, `>10`·`>32`는 곡선의 끝보다 좋음이다.
- VisA는 (나누기, PCB) 15단위를 2000번 복원 추출해 95% 구간을 냈다. SMT는 PCB 구분이 없어 나누기가 단위다.

| | VisA | SMT |
|---|---|---|
| 모델 | 위치 모델 + 촬영 조건 섞기 (`AnomalyDetection/ad/fair_ad.py`, `seg_aug`) | 위치 모델 (`AnomalyDetection/smt/seg_ad.py`, 2000 step) |
| 지표 | 배경 섞은 시험셋에서 과검 5%일 때 놓친 불량 (`fair_fnr5`) | 시험셋에서 과검 1%일 때 놓친 불량 (`fnr`) |
| 실제만 곡선 | K = 1, 2, 3, 4, 5, 6, 8, 10 × 나누기 5번 | K = 1, 2, 4, 8, 16, 32 × 나누기 5번 |
| 합성 | cTransGAN, SD+LoRA v3 × K 1·2·4 × 나누기 5번 (K8은 1번) | 사전 공정 SD+LoRA 나누기 5번, 나머지 3번 |

## 결과 요약 (합성 40장)

VisA (놓친 불량 %, 괄호 = 실제 몇 장 효과와 95% 구간)

| 실제 K | 실제만 | + cTransGAN 40 | + SD+LoRA 40 |
|---|---|---|---|
| 1 | 21.7 | 18.0 (1.4장, 1.1–2.0) | 20.5 (1.1장, 1 미만–1.4) |
| 2 | 13.3 | 10.7 (2.7장, 2.3–3.8) | 11.5 (2.5장, 2.0–3.2) |
| 4 | 7.9 | 6.3 (6.6장, 4.5–7.8) | 6.7 (6.1장, 4.0–7.5) |

SMT (놓친 불량 %, 괄호 = 실제 몇 장 효과)

| 공정 | 실제 K | 실제만 | + SD+LoRA 40 | + cTransGAN 40 |
|---|---|---|---|---|
| 사전 | 1 | 25.9 / 19.9 | 14.8 (2.4장) | 16.3 (2.6장) |
| 사전 | 4 | 10.1 / 10.6 | 6.5 (6.0장) | 6.3 (6.3장) |
| 사전 | 16 | 3.5 / 3.4 | 2.3 (곡선 끝보다 좋음, 1%p 남짓) | 1.9 (같음) |
| 납땜 | 1 | 80.2 | 77.2 (1.2장) | 72.5 (1.5장) |
| 납땜 | 4 | 56.4 | 49.4 (5.9장) | 50.9 (5.4장) |
| 납땜 | 16 | 40.9 | 49.6 (나빠짐) | 72.8 (크게 나빠짐) |

사전 공정 실제만은 SD+LoRA(나누기 5번) / cTransGAN(나누기 3번) 짝의 평균이라 두 값이다.

- 실제가 적을 때(1–4장) 합성 40장은 실제를 1.1–2.6배 모은 효과가 있다. VisA와 SMT가 같은 흐름이다.
- 실제가 16장쯤 모이면 효과가 없거나(사전) 해가 된다(납땜).
- SD+LoRA는 VisA 실제 1장에서 cTransGAN보다 2.6–4.0%p 더 놓친다(`ctg_vs_sd_visa.txt`). 2장부터는 둘이 비슷하다.

## 파일

| 파일 | 내용 |
|---|---|
| `run_visa_equiv.sh` | VisA 실행. 실제만 K 3·5·6, 합성 세트 K 2·4 × 나누기 1–4 생성(cTransGAN, SD+LoRA) → 휜 핀 잔상 제거 → 부분 세트 → AD 평가. 결과 csv나 세트 manifest가 이미 있으면 그 칸은 건너뛴다 |
| `run_smt_equiv.sh` | SMT 실행. 실제만 K 2·8·32 × 나누기 0–4, K 1·4·16 × 나누기 3–4, 사전 공정 SD+LoRA 나누기 3–4 |
| `equiv_visa.py` | VisA 실제 몇 장 효과. `python equiv_visa.py ctg` → `equiv_visa_ctg.txt`, `v3` → `equiv_visa_v3.txt` |
| `equiv_smt.py` | SMT 실제 몇 장 효과 → `equiv_smt.txt` |
| `ctg_vs_sd.py` | VisA cTransGAN과 SD+LoRA를 같은 단위끼리 짝지어 비교 → `ctg_vs_sd_visa.txt` |
| `metrics_cmp_visa.py` | VisA 합성 효과를 여러 지표로 본 것 → `metrics_cmp_visa_{ctg,v3}.txt` |
| `model_cmp_visa.py` | VisA 모델별(PatchCore, DevNet, MLP, 위치 모델) · 실제 장수별 비교 → `model_cmp_visa.txt` |
| `*.log`, `*.console.log` | 실행 로그 (`fair_ad_*` VisA 평가, `ctg_*`·`sd_*` 생성, `smt_*` SMT 평가) |

결과 csv는 팀 결과와 같은 곳에 쌓인다.

- VisA: `AnomalyDetection/results/fair_ad_K{3,5,6}_seed{0-4}.csv`(실제만), `fair_ad_K{2,4}_seed{1-4}_grid_{ctg,v3}.csv`(합성)
- SMT: `AnomalyDetection/results/smt/seg_ad_{pre,solder}.csv`에 행이 더해졌다(열 `steps` = 2000)

## 다시 돌리기

저장소 루트에서 실행한다. 파이썬은 `PY`로 넘긴다(기본 `python`).

```bash
PY=python bash AnomalyDetection/results/logs_jh/run_visa_equiv.sh   # VisA 원본이 Dataset/visa/VisA_official/VisA_PCB에 있어야 함. 다시 돌릴 칸의 결과 csv · manifest를 먼저 지운다
PY=python bash AnomalyDetection/results/logs_jh/run_smt_equiv.sh    # AIHub SMT 원본과 생성 세트가 있어야 함 (원본은 AIHub에서 받고, 생성 세트는 다시 만든다)
python AnomalyDetection/results/logs_jh/equiv_visa.py ctg
python AnomalyDetection/results/logs_jh/equiv_smt.py
```

분석 스크립트는 저장소에 있는 결과 csv만 읽으므로 데이터 없이도 바로 돌아간다. 저장된 `*.txt`와 같은 값이 나오는 것을 확인했다.
A100 80GB 한 장에서 VisA 실행은 약 14시간, SMT 실행은 약 1시간 걸렸다.
