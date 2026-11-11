# A팀 3단계 cTransGAN 실제 실행 결과

Colab Tesla T4에서 PCB별 분류기 200회·GAN 4,000회 업데이트, 각 500장 생성과 평가를 완료했습니다. seed=7017입니다. 데이터 및 더미 스모크 테스트는 PASS이고 평가 파일은 PCB별 500장·참조 gen_train 불량 33장·마스크 밖 변경 0건 검증을 통과했습니다.

**전달 완료:** Google Drive의 `DL_TP/runs/ctransgan_20261111/deliverables`에 PCB별 데이터 ZIP 4개, 코드·평가 결과 ZIP, 학습 체크포인트 ZIP을 저장했습니다. Colab에서 ZIP CRC와 복사 전후 SHA256을 검증했고, 동기화 완료 후 Drive 웹 화면에서 파일명과 크기를 확인했습니다.

|PCB|KID 평균 ± 표준편차|FID|LPIPS|학습 시간(분)|생성(초/장)|GPU peak allocated(GiB)|
|---|---:|---:|---:|---:|---:|---:|
|pcb1|0.047823 ± 0.010476|56.2319|0.152694|12.90|0.2540|0.628|
|pcb2|0.005490 ± 0.002511|26.8686|0.133917|13.44|0.2556|0.708|
|pcb3|0.005000 ± 0.001665|25.1349|0.067232|13.58|0.2607|0.708|
|pcb4|0.006312 ± 0.001354|17.8267|0.147263|12.68|0.2586|0.708|

학습 업데이트 시간 합계는 52.60분입니다. 데이터 읽기·모델 초기화·체크포인트 쓰기를 제외한 값입니다. GPU peak는 torch.cuda.max_memory_allocated이며 전체 장치 사용량과 다릅니다.

KID는 부분집합 25장, 100회 반복의 평균±표준편차입니다. heldout은 학습·조건 선택·평가 참조에 사용하지 않았습니다. FID의 실제 참조 수는 33장으로 작습니다. LPIPS는 전체 이미지의 생성 표본 1,000쌍으로 계산했으며 배경 다양성도 포함합니다.

복사-붙여넣기 대비 KID는 pcb1·2·4에서 낮아지고 pcb3에서 높아졌습니다. 반복 편차가 겹치므로 통계적으로 유의한 개선이라고 단정하지 않습니다. FID는 네 PCB 모두 낮아졌고 LPIPS는 네 PCB 모두 낮아졌습니다. 낮은 전체 이미지 KID가 결함 자체의 사실성을 보장하지는 않습니다.

이 모델은 원논문의 축소 구현입니다. 클래스 특징 평균 조건과 Transformer G/D를 유지하고 폭·깊이와 학습량을 줄였습니다. 결함 ROI를 32×32로 생성해 gen_train 정상 이미지에 삽입합니다. 전체 출력은 긴 변 512px, 원본 종횡비를 보존합니다. 마스크는 합성 삽입 영역이며 semantic_mask_verified=false입니다. PSNR/SSIM은 원본 정상에 대한 구조 보존 보조값입니다. A100과 4·5단계는 미실행입니다.

## Drive 전달 파일

|파일|바이트|SHA256|
|---|---:|---|
|DL_TP_stage3_code_results.zip|9605126|c69a34b513345d8650973e673d5ea773bf638a6980d762da84bb17a274a52e1f|
|DL_TP_ctransgan_pcb1.zip|172754384|8c68b313f838fb8479f158d8f00fd7a66d8b913de3c57b7ad4e4f0dd022f4aa9|
|DL_TP_ctransgan_pcb2.zip|170640763|b8dc697db3d60ff9310bc5bfab66e7abc28a0d541ee2f2035a73715bca69d22e|
|DL_TP_ctransgan_pcb3.zip|110385299|18b0ea3bd732d6e2de2adc449b1697db02371234d562b74d6c2e33147367902b|
|DL_TP_ctransgan_pcb4.zip|150722981|b05e3a6a6792f60a783fc254e71b18deb287968e30e59fcee39b0fecade5ba14|

PCB별 ZIP에는 DL_TP/outputs/ctransgan/{pcb}/images, masks, meta.csv, records, cost.json, run.json과 분할 CSV·방법 문서가 포함됩니다. 코드 ZIP에는 src/configs/splits/notebooks/tests/results/figures/reports가 포함됩니다. 체크포인트는 ZIP에 포함하지 않았습니다.

## 파일 사용

B팀은 필요한 PCB ZIP을 같은 폴더에 풀면 `DL_TP/outputs/ctransgan/{pcb}/images`, `masks`, `meta.csv`를 그대로 사용할 수 있습니다. `DL_TP_ctransgan_checkpoints.zip`(317,871,242바이트)에는 PCB별 분류기 200회·GAN 4,000회 완료 체크포인트 4개가 들어 있습니다. 파일별 해시는 같은 Drive 폴더의 `storage_verified.json`에서 확인할 수 있습니다.

논문: Wang et al., Conditional TransGAN-Based Data Augmentation for PCB Electronic Component Inspection, Computational Intelligence and Neuroscience, 2023, DOI 10.1155/2023/2024237. 저자 저장소 long-deep/pcb-detect는 미완성 초기화 오류와 LICENSE 부재를 확인하여 코드를 재배포하지 않고 독립 구현했습니다. 상세 변경점은 기존 STAGE3_CTRANSGAN.md 및 README에 기록했습니다.
