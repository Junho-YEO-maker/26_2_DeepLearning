"""VisA 공식 분할(split_csv/1cls.csv, 2cls_fewshot.csv)을 우리 분할표 형식으로 바꾼다 (11/20, 공개 수치 비교용).

우리 실험은 직접 만든 분할(Dataset/visa/splits, 종류당 K장)이라 공개 논문 수치와 바로 비교할 수 없다.
VisA가 함께 내놓은 공식 분할로 같은 모델을 다시 돌리기 위해, 공식 행을 우리 csv 형식(category, image, mask, label, split,
defect_type, image_sha256)으로 옮긴다. 결함 종류와 해시는 우리 분할표(pool·test·train_normal·val_normal)에서 가져온다
(PCB1–3의 모든 이미지가 그 안에 있다).

  1cls          학습 = 정상 약 900장/PCB, 시험 = 불량 100 + 정상 약 100                     (정상만 학습하는 표준 평가)
  2cls_fewshot  학습 = 불량 20장 + 정상 약 200장/PCB, 시험 = 불량 80 + 정상 약 800          (불량 조금 쓰는 공식 평가)
결과 폴더 Dataset/visa/splits_official/{protocol}/
  test.csv, train_normal.csv, val_normal.csv(비움: 공식 분할에 검증 정상이 없다),
  pool.csv · train_K20_seed0.csv (2cls_fewshot의 학습 불량 20장. 생성기 코드가 K장 파일 이름을 그대로 읽도록)

    python -m AnomalyDetection.ad.official_splits
"""
import csv

from .data import PCBS, RAW_PREFIX, ROOT, read_csv

OFFICIAL = ROOT / 'Dataset/visa/VisA_official/VisA_PCB/split_csv'
OUT = ROOT / 'Dataset/visa/splits_official'
FIELDS = ['category', 'image', 'mask', 'label', 'split', 'defect_type', 'image_sha256']


def ours():
    rows = []
    for f in ('pool', 'test', 'train_normal', 'val_normal'):
        rows += read_csv(ROOT / 'Dataset/visa/splits' / f'{f}.csv')
    return {r['image']: r for r in rows}


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def convert(protocol, known):
    out = OUT / protocol
    out.mkdir(parents=True, exist_ok=True)
    parts = {'test': [], 'train_normal': [], 'train_defect': []}
    for r in read_csv(OFFICIAL / f'{protocol}.csv'):
        if r['object'] not in PCBS:
            continue
        img = RAW_PREFIX + r['image']
        k = known[img]
        lab = '1' if r['label'] == 'anomaly' else '0'
        assert lab == k['label'], img
        row = dict(category=r['object'], image=img, mask=(RAW_PREFIX + r['mask']) if r['mask'] else '', label=lab,
                   split=r['split'], defect_type=k['defect_type'] if lab == '1' else '', image_sha256=k['image_sha256'])
        if r['split'] == 'test':
            parts['test'].append(row)
        elif lab == '0':
            parts['train_normal'].append(row)
        else:
            parts['train_defect'].append(dict(row, split='train_K20_seed0'))
    write(out / 'test.csv', parts['test'])
    write(out / 'train_normal.csv', parts['train_normal'])
    write(out / 'val_normal.csv', [])
    write(out / 'pool.csv', parts['train_defect'])
    write(out / 'train_K20_seed0.csv', parts['train_defect'])
    for pcb in PCBS:
        n = {k: sum(r['category'] == pcb for r in v) for k, v in parts.items()}
        t = {}
        for r in parts['train_defect']:
            if r['category'] == pcb:
                t[r['defect_type']] = t.get(r['defect_type'], 0) + 1
        print(f'{protocol} {pcb}: 시험 {n["test"]} (불량 {sum(r["category"] == pcb and r["label"] == "1" for r in parts["test"])}), '
              f'학습 정상 {n["train_normal"]}, 학습 불량 {n["train_defect"]} {t}')
    assert not ({r['image'] for r in parts['test']} & {r['image'] for r in parts['train_normal'] + parts['train_defect']})


if __name__ == '__main__':
    known = ours()
    for p in ('1cls', '2cls_fewshot'):
        convert(p, known)
    print('저장:', OUT)
