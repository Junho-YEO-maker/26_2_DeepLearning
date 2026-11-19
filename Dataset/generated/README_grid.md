# 생성기 × K × 합성 장수 N 실험용 생성 세트 (VisA PCB, 11/19)

다른 곳에서 같은 조건으로 AD 모델을 돌릴 수 있게 올린 세트다. 결과 표와 그림은 `AnomalyDetection/results/ppt/grid_summary.md`, `grid_visa.png`.

## 세트

| 이름 | 생성기 | 올린 칸 (K = 종류당 실제 불량 수, seed = 분할 seed) |
|---|---|---|
| `sd_lora_v3/` | SD1.5 인페인팅 + LoRA (`Augmentation/sd_lora`, v3) | K1 seed 0–4, K2·K4·K8 seed 0 |
| `ctransgan_diverse/` | cTransGAN (`Augmentation/ctransgan/generate_diverse.py --diverse-scratch`) | K1 seed 0–4, K2·K4·K8 seed 0 |

- 칸마다 PCB 3개 × 종류 4개(bent, melt, scratch, missing) × 40장 = 480장. `images/`(생성 불량, 원본 해상도 JPEG), `masks/`(종류 id 마스크 PNG), `manifest.csv`(`image, mask, label, source_image, base_normal_image, condition` …), `review.jpg`(bent 잔상 제거 전후).
- 생성기는 K장(`Dataset/visa/splits/train_K{k}_seed{s}`)만 보고 학습했고, 정상 학습 이미지 위에 결함을 그렸다. 시험 이미지는 쓰지 않았다(`check_manifest` 누수 검사 PASS).
- **bent 핀 잔상 제거**: 두 생성기 모두 bent에서 원래 곧은 핀이 남는 문제가 있어 `Augmentation/pin_cleanup/clean.py`로 지웠다. 고친 장은 `condition.pinfix`에 기록. 고치기 전 세트는 서버에만 보관(`_before_pinfix/`, git 제외).
- 그 밖의 세트(`sd_lora_v3/K10_seed0`, `copypaste/`)는 이미지 없이 manifest만 있다.

## 합성 장수 N 부분 세트

`{sd_lora_v3,ctransgan_diverse}_n{1,5,10,20}/K{k}_seed{s}/manifest.csv`. 이미지는 복사하지 않고 위 세트의 경로를 가리킨다.
PCB·종류마다 고정 무작위 순서의 앞 N장이다(`Augmentation/gen_subset.py`, 순서는 (생성기, K, seed)마다 고정, N=1 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 40). N=40은 원래 세트 전체.

    cd Augmentation && python -m gen_subset --aug sd_lora_v3 --k 1 --seed 0 --n 1 5 10 20   # 다시 만들 때

## 실험 조건 (우리 결과와 맞추려면)

- 학습: 실제 K장 + 생성 N장(종류당). 위치 seg는 배치 = 정상 절반 + 불량 절반, 불량 중 실제:생성 = 반반 고정이라 N은 생성 장수만 바꾼다. `seg_aug`는 학습 이미지에 촬영 조건 무작위화(capture_aug)를 더한 것.
- 평가: 공정 시험셋(`AnomalyDetection/ad/fair_ad.py` 설명). 시험 불량 D마다 결함 지운 DR, 정상에 결함 붙인 ND, 붙이기 대조 NR을 VisA 원본에서 결정적으로 만든다(이미지는 올리지 않음, 처음 한 번 `--build`). 주 지표 `fair_fnr5` = 과검률 5%에서 놓친 비율, 종류별 `fair_fnr_{bent,…}`.
- 기준: N=0(실제만)은 같은 모델을 `--augs` 없이 돌린 것(`fair_ad_K{k}_seed{s}.csv`의 `seg_aug,none`), PatchCore(정상만) 0.188.

칸 하나 (K1 seed 0, cTransGAN, PCB 하나):

    DR=/path/to/VisA_PCB   # pcb1~3 원본
    python -m AnomalyDetection.ad.fair_ad --build --data-root $DR          # 공정 시험셋, 처음 한 번
    A=ctransgan_diverse
    python -m AnomalyDetection.ad.fair_ad --k 1 --seed 0 --pcbs pcb1 --tag _grid_ctg --data-root $DR \
      --augs ${A}_n1 ${A}_n5 ${A}_n10 ${A}_n20 $A \
      --sets seg_aug:${A}_n1 seg_aug:${A}_n5 seg_aug:${A}_n10 seg_aug:${A}_n20 seg_aug:$A seg:$A img:$A \
      --seg-seeds 2 --img-seeds 2 --img-kinds devnet

전체 반복은 `AnomalyDetection/results/logs/run_pinfix_rerun.sh`(SD+LoRA 전 칸, cTransGAN K1·2·4·8 seed 0)와 `run_followups.sh` B(cTransGAN K1 seed 1–4). 결과 csv: `AnomalyDetection/results/fair_ad_K{k}_seed{s}_grid_{v3|ctg}.csv`, 표·그림: `python -m AnomalyDetection.ad.grid_summary`.

## 올리지 않은 것

- AIHub SMT 생성 불량(`Dataset/aihub_smt/generated/`)과 원본·분할표: AIHub 이용약관상 제3자 제공 금지. SMT 실험은 서버에서만 돌린다(코드는 `AnomalyDetection/smt/`, `Augmentation/smt_sd_lora/`).
