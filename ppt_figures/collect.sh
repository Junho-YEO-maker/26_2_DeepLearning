#!/usr/bin/env bash
# 발표 슬라이드에 넣을 그림을 이 폴더로 모은다 (ppt_preview.md 슬라이드 번호 기준). 원본이 바뀌면 다시 돌린다.
#   bash ppt_figures/collect.sh
# 이름 = s{슬라이드}_{내용}. 원본 위치와 설명은 ppt_figures/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=ppt_figures
CTG=/home/junho/ctransgan_runs/diverse_full_u9lv51T7
cp_() { if [ -f "$1" ]; then cp -p "$1" "$OUT/$2"; echo "  $2"; else echo "  (없음) $1"; fi; }
cp_ Augmentation/figures/capture_difference.png                 s03_capture_difference.png
cp_ Augmentation/figures/resolution_check.png                   s03_defect_size_resolution.png
for p in 1 2 3; do cp_ Augmentation/figures/sd_lora_v2_pcb${p}_sheet.jpg s06_sd_lora_v2_pcb${p}_sheet.jpg; done
cp_ Augmentation/figures/v4_base_vs_generated.jpg               s06_sd_lora_v4_base_vs_generated.jpg
for k in 1 2 4; do cp_ $CTG/review_K${k}_seed0.png              s07_ctransgan_K${k}_review.png; done
for p in 1 2 3; do cp_ Augmentation/figures/sd_lora_v3_pcb${p}_sheet.jpg s08_sd_lora_v3_pcb${p}_sheet.jpg; done
cp_ Augmentation/figures/real_vs_v3_melt_missing.png            s08_real_vs_v3_melt_missing.png
cp_ AnomalyDetection/results/kcurve_none_512.png                s11_kcurve_standard_test.png
cp_ Augmentation/figures/shortcut_diagnosis_K1_seed0.png        s12_shortcut_diagnosis.png
cp_ AnomalyDetection/results/gradcam/K1_seed0_all.jpg           s12_s14_gradcam_vs_seg_map.jpg
cp_ AnomalyDetection/results/ppt/fair_test_examples.png         s13_fair_test_examples.png
cp_ AnomalyDetection/results/ppt/fair_kcurve.png                s14_fair_kcurve.png
cp_ AnomalyDetection/results/ppt/fair_kcurve_pcb.png            s14_fair_kcurve_pcb.png
