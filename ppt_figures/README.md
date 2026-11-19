# 발표 슬라이드 그림

슬라이드 번호는 `ppt_preview.md` 기준이다. `bash ppt_figures/collect.sh`가 원본에서 복사한다. 원본을 고치면 이 스크립트를 다시 돌린다.

11/19 갱신: 생성 예시(s06 전후, s07, s08)는 bent 핀 잔상을 지운 세트로 다시 만들었다. 생성기 × K × 합성 장수 N 그림(s14_grid)과 공정 K 곡선(s14_fair)도 잔상 제거 후 결과다.

| 파일 | 슬라이드 | 보여주는 것 | 원본 |
|---|---|---|---|
| `s03_capture_difference.png` | 3 데이터 특성 | PCB별 정상 사진 평균 vs 불량 사진 평균, 밝기 분포. 불량 사진은 다르게 찍혔다 | `Augmentation/figures/capture_difference.png` |
| `s03_defect_size_resolution.png` | 3 데이터 특성 | 결함 크기. 원본, 512(모델 입력), 256에서 확대한 모습 | `Augmentation/figures/resolution_check.png` |
| `s06_sd_lora_v2_pcb{1,2,3}_sheet.jpg` | 6 시행착오 | SD+LoRA v2 생성 결과. 결함을 약하게 그린다 | `Augmentation/figures/` |
| `s06_sd_lora_v4_base_vs_generated.jpg` | 6 시행착오 | v4(missing 지우기, 자리 옮기기) | `Augmentation/figures/v4_base_vs_generated.jpg` |
| `s06_pinfix_before_after.jpg` | 6 시행착오 | bent 핀 잔상 제거 전후. PCB마다 [실제 결함 \| 전 \| 후], 왼쪽 SD+LoRA v3, 오른쪽 cTransGAN (K1 seed 0) | `Augmentation/figures/pinfix_before_after.jpg` (`pin_cleanup/figures.py`) |
| `s07_ctransgan_K{1,2,4,8}_review.png` | 7 cTransGAN | 잔상 제거한 cTransGAN. PCB·종류마다 [K장 실제 \| 정상 바탕 \| 생성 \| 마스크] | `Augmentation/ctransgan/review/fixed_K{k}_seed0.png` (`ctransgan.review`) |
| `s08_sd_lora_v3_pcb{1,2,3}_sheet.jpg` | 8 SD+LoRA | 잔상 제거한 v3. 종류마다 예 2개 [실제 K=1 \| 정상 바탕 \| 생성] (전과 같은 장) | `Augmentation/figures/sd_lora_v3_fixed_pcb{p}_sheet.jpg` (`pin_cleanup/figures.py`) |
| `s08_real_vs_v3_melt_missing.png` | 8 SD+LoRA | 실제 결함 vs v3 생성 (melt, missing. 잔상 제거는 bent만이라 그대로) | `Augmentation/figures/real_vs_v3_melt_missing.png` |
| `s11_kcurve_standard_test.png` | 11 이미지 단위 | 원래 시험셋 K 곡선, 실제만 (PatchCore, DevNet, kNN/MLP) | `AnomalyDetection/results/kcurve_none_512.png` |
| `s12_shortcut_diagnosis.png` | 12 지름길 추정 | 결함과 무관한 통계로 정상/불량 구분 | `Augmentation/figures/shortcut_diagnosis_K1_seed0.png` |
| `s12_defect_removal_{pcb1_bent,pcb2_melt,pcb3_scratch}.jpg` | 12 지름길 추정 | 결함 지우기/붙이기 [D \| DR \| N \| ND] | `AnomalyDetection/results/defect_removal_examples/` |
| `s12_s14_gradcam_vs_seg_map.jpg` | 12, 14 | 같은 D / DR / ND 사진에서 이미지 DevNet Grad-CAM vs 위치 seg 결함 지도 | `AnomalyDetection/results/gradcam/K1_seed0_all.jpg` (`ad/gradcam.py`) |
| `s13_fair_test_examples.png` | 13 데이터 맞춤 조정 | 공정 시험셋 구성: D / DR / N / ND / NR을 결함 주변으로 확대 | `AnomalyDetection/results/ppt/fair_test_examples.png` (`ad/fair_figure.py`) |
| `s14_fair_kcurve.png` | 14 결과 | K 곡선. 왼쪽 공정 시험셋, 오른쪽 원래 시험셋 (둘 다 과검률 5%) | `AnomalyDetection/results/ppt/` (`ad/ppt_summary.py`) |
| `s14_fair_kcurve_pcb.png` | 14 결과 | PCB별 공정 K 곡선 | `AnomalyDetection/results/ppt/` (`ad/ppt_summary.py`) |
| `s14_grid_visa.png` | 14 결과 | VisA 위치 seg, 생성기별 x = 합성 장수 N, 선 = K (공정 시험셋 미검률 @ 과검 5%) | `AnomalyDetection/results/ppt/grid_visa.png` (`ad/grid_summary.py`) |
| `s14_grid_smt.png` | 14 결과 | SMT 위치 seg, 위 줄 SD+LoRA·아래 줄 cTransGAN, 공정별 x = N, 선 = K (미검률 @ 과검 1%) | `AnomalyDetection/results/ppt/grid_smt.png` (`ad/grid_summary.py`) |
| `local_only/s14_smt_sd_lora_fix_sheet.jpg` | 14 (SMT) | SMT 납땜 ms·nm 자리 고치기 전후 | `Dataset/aihub_smt/generated/sd_lora_fix/K1_seed0/sheet_fix.jpg` |
| `local_only/s14_smt_ms_real_pads_sheet.jpg` | 14 (SMT) | ms를 실제 빈 패드로 만든 것 | `Dataset/aihub_smt/generated/sd_lora_fix_msreal/K1_seed0/sheet_ms.jpg` |

`local_only/`는 AIHub SMT 사진에서 나온 그림이다. AIHub 약관상 제3자 제공 금지라 git에 올리지 않는다(`.gitignore`). 서버에서만 받아 쓴다.

표에 들어갈 숫자는 `AnomalyDetection/results/ppt/summary.md`(공정 K 곡선), `AnomalyDetection/results/ppt/grid_summary.md`(생성기 × K × N)에 있다.

아직 없는 그림: 슬라이드 5(접근 한눈에), 10(AD 모델 4개 구조).
