# Augmentation 팀

불량 K장(`Dataset/visa/splits/train_K{k}_seed{s}.csv`)만 소스로 써서 생성 불량 이미지를 만들고
`Dataset/generated/{방법}/K{k}_seed{s}/` 에 쌓는다. AD 팀은 이 폴더를 그대로 가져다 쓴다.

## 방법
- `copypaste` : 불량 패치를 정상 이미지에 붙여넣기 (기준선)
- `sd_lora`   : SD inpainting + LoRA 학습 후 마스크 영역 생성
- 품질 필터로 걸러진 결과만 manifest에 남긴다

## 출력 형식 (고정)
```
Dataset/generated/{방법}/K{k}_seed{s}/
├─ images/      생성 이미지
├─ masks/       결함 마스크 (이미지와 같은 파일명, png)
└─ manifest.csv image, mask, label, source_image, score
```
- `image`, `mask`, `source_image` 는 프로젝트 루트 기준 상대경로
- `source_image` 는 반드시 해당 `train_K{k}_seed{s}.csv` 안의 불량 이미지
- `label` 은 1 (불량)
- `score` 는 품질 필터 점수 (필터 없으면 빈칸)
- images/, masks/ 는 git 제외 (Drive), manifest.csv 만 git

## 금지
- `test.csv` 이미지를 소스·학습·참조에 쓰지 않는다
- 같은 (방법, K, seed) 결과를 덮어쓸 때는 manifest 도 같이 갱신한다

## 실행 환경
- Colab T4 기준 SD 1.5, 512px, fp16. SDXL 은 메모리 부족
- 추가 패키지는 여기에 기록: 
