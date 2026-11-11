"""공유 분할표 → K장 copy-paste → 누수 검사를 소형 더미 데이터로 끝까지 돌린다."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.shared_splits import SharedSplits, check_manifest, read_rows
from src.generate.k_copypaste import generate_condition
from tests.smoke_stage1 import must_fail

REPO = Path(__file__).resolve().parents[3]
TYPES = ['bent', 'melt', 'scratch', 'missing']


def make_dummy(raw, pcbs=('pcb1', 'pcb2', 'pcb3')):
    rng = np.random.default_rng(7017)
    for pcb in pcbs:
        base = raw / pcb / 'Data'
        for d in ('Images/Normal', 'Images/Anomaly', 'Masks/Anomaly'):
            (base / d).mkdir(parents=True)
        rows = []
        for i in range(8):
            Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(base / f'Images/Normal/{i:04d}.JPG')
            rows.append((f'{pcb}/Data/Images/Normal/{i:04d}.JPG', 'normal', ''))
        labels = [[t] for t in TYPES for _ in range(4)] + [['melt', 'scratch']]
        for i, types in enumerate(labels):
            Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(base / f'Images/Anomaly/{i:03d}.JPG')
            m = np.zeros((40, 48), np.uint8)
            for j, t in enumerate(types):
                m[5 + 10 * j:12 + 10 * j, 10:20] = TYPES.index(t) + 1
            Image.fromarray(m).save(base / f'Masks/Anomaly/{i:03d}.png')
            rows.append((f'{pcb}/Data/Images/Anomaly/{i:03d}.JPG', ','.join(types),
                         f'{pcb}/Data/Masks/Anomaly/{i:03d}.png'))
        with open(raw / pcb / 'image_anno.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'label', 'mask'])
            w.writerows(rows)


def run():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw, splits, out = td / 'raw', td / 'splits', td / 'gen'
        make_dummy(raw)
        subprocess.run([sys.executable, str(REPO / 'scripts/make_splits.py'), '--raw', str(raw), '--out', str(splits),
                        '--ks', '1', '2', '--seeds', '0', '1', '--pool-max', '2', '--test-min', '1',
                        '--train-normal', '4', '--val-normal', '2'], check=True, capture_output=True)
        s = SharedSplits(splits, raw)
        manifest = generate_condition(s, 2, 1, out, per_type=3)
        rows = read_rows(manifest)
        assert len(rows) == 3 * 4 * 3
        for r in rows:
            m = np.asarray(Image.open(r['mask']))
            assert set(np.unique(m)) == {0, TYPES.index(r['defect_type']) + 1}, 'mask must hold type id'
            assert Image.open(r['image']).size == (48, 40), 'original resolution'
        assert check_manifest(manifest, splits)['status'] == 'PASS'

        # 재실행은 건너뛰고 결과가 같다
        before = manifest.read_bytes()
        generate_condition(s, 2, 1, out, per_type=3)
        assert manifest.read_bytes() == before

        # test 이미지를 소스로 바꿔치기하면 누수 검사가 잡는다
        test_img = next(r['image'] for r in read_rows(splits / 'test.csv') if r['label'] == '1')
        rows[0]['source_image'] = test_img
        with open(manifest, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        must_fail(lambda: check_manifest(manifest, splits))

        # test 행은 열 수 없다
        test_row = next(r for r in read_rows(splits / 'test.csv'))
        must_fail(lambda: s.open(test_row))
    return {'status': 'PASS'}


if __name__ == '__main__':
    print(run())
