# 발표 슬라이드 그림

슬라이드 번호는 `ppt_preview.md` 기준이다. `bash ppt_figures/collect.sh`가 원본에서 복사한다. 원본을 고치면 이 스크립트를 다시 돌린다.

공정 K 곡선(s14)은 seed 1–4 실행이 끝나면 `AnomalyDetection/results/logs/run_ppt_final.sh`가 다시 그리고 이 폴더로 복사한다. 그 전에 받은 그림은 seed 0만 반영된 것이다.

| 파일 | 슬라이드 | 보여주는 것 | 원본 |
|---|---|---|---|
| `s03_capture_difference.png` | 3 데이터 특성 | PCB별 정상 사진 평균 vs 불량 사진 평균, 밝기 분포. 불량 사진은 다르게 찍혔다 | `Augmentation/figures/capture_difference.png` |
| `s03_defect_size_resolution.png` | 3 데이터 특성 | 결함 크기. 원본, 512(모델 입력), 256에서 확대한 모습 | `Augmentation/figures/resolution_check.png` |
| `s06_sd_lora_v2_pcb{1,2,3}_sheet.jpg` | 6 시행착오 | SD+LoRA v2 생성 결과. 결함을 약하게 그린다 | `Augmentation/figures/` |
| `s06_sd_lora_v4_base_vs_generated.jpg` | 6 시행착오 | v4(missing 지우기, 자리 옮기기) | `Augmentation/figures/v4_base_vs_generated.jpg` |
| `s07_ctransgan_K{1,2,4}_review.png` | 7 cTransGAN | 새 버전 생성 결과 | `/home/junho/ctransgan_runs/diverse_full_u9lv51T7/` (프로젝트 밖) |
| `s08_sd_lora_v3_pcb{1,2,3}_sheet.jpg` | 8 SD+LoRA | 최종 v3 생성 결과 | `Augmentation/figures/` |
| `s08_real_vs_v3_melt_missing.png` | 8 SD+LoRA | 실제 결함 vs v3 생성 (melt, missing) | `Augmentation/figures/real_vs_v3_melt_missing.png` |
| `s11_kcurve_standard_test.png` | 11 이미지 단위 | 원래 시험셋 K 곡선, 실제만 (PatchCore, DevNet, kNN/MLP) | `AnomalyDetection/results/kcurve_none_512.png` |
| `s12_shortcut_diagnosis.png` | 12 지름길 추정 | 결함과 무관한 통계로 정상/불량 구분 | `Augmentation/figures/shortcut_diagnosis_K1_seed0.png` |
| `s12_s14_gradcam_vs_seg_map.jpg` | 12, 14 | 같은 D / DR / ND 사진에서 이미지 DevNet Grad-CAM vs 위치 seg 결함 지도 | `AnomalyDetection/results/gradcam/K1_seed0_all.jpg` (`ad/gradcam.py`) |
| `s13_fair_test_examples.png` | 13 데이터 맞춤 조정 | 공정 시험셋 구성: D / DR / N / ND / NR을 결함 주변으로 확대 | `AnomalyDetection/results/ppt/fair_test_examples.png` (`ad/fair_figure.py`) |
| `s14_fair_kcurve.png` | 14 결과 | K 곡선. 왼쪽 공정 시험셋, 오른쪽 원래 시험셋 (둘 다 과검률 5%) | `AnomalyDetection/results/ppt/` (`ad/ppt_summary.py`) |
| `s14_fair_kcurve_pcb.png` | 14 결과 | PCB별 공정 K 곡선 | `AnomalyDetection/results/ppt/` (`ad/ppt_summary.py`) |

표에 들어갈 숫자는 `AnomalyDetection/results/ppt/summary.md`에 있다.

아직 없는 그림: 슬라이드 5(접근 한눈에), 10(AD 모델 4개 구조).
