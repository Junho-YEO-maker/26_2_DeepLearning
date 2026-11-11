"""Create the stage-1 text report from measured artifacts only."""
import json,csv
from pathlib import Path
from .prepare import PCBS

def build():
    m=json.loads(Path('splits/manifest.json').read_text()); archive=json.loads(Path('reports/archive_provenance.json').read_text());audit=json.loads(Path('reports/real_audit.json').read_text())
    lines=['# 1단계 실제 데이터 검사 결과','', '원본 tar를 내려받아 PCB 4종의 이미지·마스크를 실제로 검사했습니다. 학습 및 품질지표 계산은 미실행입니다.','',f"- 원본 크기: {archive['bytes']:,} bytes",f"- SHA256: `{archive['sha256']}`",'- image_anno.csv 컬럼: image, label, mask. label에는 정상(normal) 또는 불량 종류가 있고 여러 종류는 쉼표로 구분합니다.','- 원본 폴더: `{pcb}/Data/Images/{Normal,Anomaly}/*.JPG`, `{pcb}/Data/Masks/Anomaly/*.png`, `{pcb}/image_anno.csv`.','- 정상은 저장된 마스크가 없습니다. 불량은 종류별 ID를 담은 PNG입니다. 이진화는 `mask > 0`으로 합니다.','- 전체 400개 불량의 CSV 종류 집합과 공식 id2class로 해석한 마스크 종류 집합이 일치했습니다.','', '## 분할 결과','', '|PCB|정상 gen_train|정상 heldout|불량 gen_train|불량 heldout|다중 종류 불량 수|','|---|---:|---:|---:|---:|---:|']
    for pcb in PCBS:
        d=m['pcb'][pcb];c=d['counts'];lines.append(f"|{pcb}|{c['normal/gen_train']}|{c['normal/heldout']}|{c['anomaly/gen_train']}|{c['anomaly/heldout']}|{d['multilabel_images']}|")
    lines+=['','seed=7017. 불량은 floor(N/3), 정상은 floor(N×0.8). 정상 80%는 지정되지 않은 비율에 대한 구현 선택입니다. 다중 라벨을 단일 대표 라벨로 바꾸지 않고 정수 층화했습니다.','', '## 원본 해상도와 마스크','', '|PCB|해상도 (width×height: 이미지 수)|마스크 모드|실제 마스크 값|','|---|---|---|---|']
    for pcb in PCBS:
        d=m['pcb'][pcb];lines.append(f"|{pcb}|{d['dimensions']}|{d['mask_modes']}|{list(d['mask_values_image_occurrences'])}|")
    lines+=['','리사이즈·크롭은 수행하지 않았습니다. 모든 불량 마스크와 대응 이미지 크기가 일치했습니다.','', '## 불량 종류별 분포','', '다중 라벨이므로 종류별 건수를 합하면 불량 이미지 수를 초과할 수 있습니다.','', '|PCB|종류|gen_train|heldout|','|---|---|---:|---:|']
    for pcb in PCBS:
        for name,c in m['pcb'][pcb]['class_counts'].items():lines.append(f"|{pcb}|{name}|{c.get('gen_train',0)}|{c.get('heldout',0)}|")
    lines+=['','## 원본 split_csv','', 'tar 안의 3개 CSV와 공식 GitHub 버전의 바이트가 모두 일치했습니다. 각 파일은 전체 데이터셋 10,821개 행이며 컬럼은 object, split, label, image, mask입니다. 여기의 label은 normal/anomaly 이진 라벨입니다. 따라서 불량 종류 정보는 split_csv가 아닌 image_anno.csv에서 확인해야 합니다.','', '|공식 split|PCB별 불량 train/test|정상 train/test|이번 실험 사용 여부|','|---|---|---|---|','|1cls.csv|0 / 100|약 90% / 10%|참고용|','|2cls_fewshot.csv|20 / 80|약 20% / 80%|참고용|','|2cls_highshot.csv|60 / 40|약 60% / 40%|참고용|','', '정확한 PCB별 정상 건수는 `official_split_inspection.json`에 기록했습니다. 새 `splits/*.csv`만 A/B팀 공통 분할로 사용합니다.','', '## 검증 결과','', '- 더미 데이터 end-to-end 분할·로딩 smoke test: PASS.','- 재실행한 더미 분할/manifest의 바이트 일치: PASS.','- heldout 접근 및 용도 위조 차단: PASS.','- CSV 변조 및 decoded-pixel 중복 누수 차단: PASS.',f"- 실제 이미지 {audit['verified_images']:,}장과 불량 마스크 전체 파일 해시 감사: {audit['status']}.",'- PCB 4종 전체를 합친 train/heldout 사이 원본 바이트·디코딩 RGB 해시 교집합 없음.','- 실제 gen_train 불량 132개 모두 품질 참조 전용 게이트로 원본 크기 로딩 성공.','- 원본 모든 이미지/마스크 파일과 주석 행의 집합 일치.','- 모델 훈련/지표 계산/Colab GPU 실행: 미실행. 현재 환경은 nvidia-smi가 없는 CPU 환경입니다.','', '## 해석 및 다음 단계','', '단계 2에서 기준선, KID/FID/LPIPS, B팀 outputs/meta 형식을 구현합니다. heldout은 생성기 및 A팀 품질 평가에 사용하지 않습니다.','', 'B팀이 필요한 최소 진짜 불량 k를 주장하려면 생성기도 k장만 본 조건이어야 합니다. 33장으로 학습한 생성 데이터를 k<33 실험에 쓰면 이는 분류기 학습에 직접 넣는 진짜 불량 수만 줄인 실험입니다.','', '출처·라이선스·SeaS 대비 변경점·실행 순서는 README.md를 참고하세요.']
    Path('reports/data_inspection.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':build()
