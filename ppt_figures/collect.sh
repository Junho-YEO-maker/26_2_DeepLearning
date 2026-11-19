#!/usr/bin/env bash
# 발표 슬라이드에 넣을 그림을 이 폴더로 모은다 (ppt_preview.md 슬라이드 번호 기준). 원본이 바뀌면 다시 돌린다.
#   bash ppt_figures/collect.sh
# 이름 = s{슬라이드}_{내용}. 원본 위치와 설명은 ppt_figures/README.md.
# 11/19: 생성 예시(s06 전후, s07, s08)는 잔상 제거한 세트로 다시 만든 것 (Augmentation/pin_cleanup/figures.py, ctransgan.review).
# local_only/ = AIHub SMT 파생 이미지. AIHub 약관상 제3자 제공 금지라 git에 올리지 않는다(.gitignore).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=ppt_figures
mkdir -p $OUT/local_only
cp_() { if [ -f "$1" ]; then cp -p "$1" "$OUT/$2"; echo "  $2"; else echo "  (없음) $1"; fi; }
cp_ Augmentation/figures/capture_difference.png                 s03_capture_difference.png
cp_ Augmentation/figures/resolution_check.png                   s03_defect_size_resolution.png
for p in 1 2 3; do cp_ Augmentation/figures/sd_lora_v2_pcb${p}_sheet.jpg s06_sd_lora_v2_pcb${p}_sheet.jpg; done
cp_ Augmentation/figures/v4_base_vs_generated.jpg               s06_sd_lora_v4_base_vs_generated.jpg
cp_ Augmentation/figures/pinfix_before_after.jpg                s06_pinfix_before_after.jpg
for k in 1 2 4 8; do cp_ Augmentation/ctransgan/review/fixed_K${k}_seed0.png s07_ctransgan_K${k}_review.png; done
for p in 1 2 3; do cp_ Augmentation/figures/sd_lora_v3_fixed_pcb${p}_sheet.jpg s08_sd_lora_v3_pcb${p}_sheet.jpg; done
cp_ Augmentation/figures/real_vs_v3_melt_missing.png            s08_real_vs_v3_melt_missing.png
cp_ AnomalyDetection/results/kcurve_none_512.png                s11_kcurve_standard_test.png
cp_ Augmentation/figures/shortcut_diagnosis_K1_seed0.png        s12_shortcut_diagnosis.png
for e in pcb1_bent pcb2_melt pcb3_scratch; do
  cp_ AnomalyDetection/results/defect_removal_examples/${e}_1_D-DR-N-ND.jpg s12_defect_removal_${e}.jpg
done
cp_ AnomalyDetection/results/gradcam/K1_seed0_all.jpg           s12_s14_gradcam_vs_seg_map.jpg
cp_ AnomalyDetection/results/ppt/fair_test_examples.png         s13_fair_test_examples.png
cp_ AnomalyDetection/results/ppt/fair_kcurve.png                s14_fair_kcurve.png
cp_ AnomalyDetection/results/ppt/fair_kcurve_pcb.png            s14_fair_kcurve_pcb.png
cp_ AnomalyDetection/results/ppt/grid_visa.png                  s14_grid_visa.png
cp_ AnomalyDetection/results/ppt/grid_smt.png                   s14_grid_smt.png
cp_ Dataset/aihub_smt/generated/sd_lora_fix/K1_seed0/sheet_fix.jpg         local_only/s14_smt_sd_lora_fix_sheet.jpg
cp_ Dataset/aihub_smt/generated/sd_lora_fix_msreal/K1_seed0/sheet_ms.jpg   local_only/s14_smt_ms_real_pads_sheet.jpg
