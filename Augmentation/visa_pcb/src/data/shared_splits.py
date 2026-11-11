"""두 팀 공유 분할표(Dataset/visa/splits, scripts/make_splits.py)를 Augmentation 팀이 읽는 관문.

K장 실험의 생성 코드는 이 모듈로만 원본을 읽는다.
  - 결함 소스: train_K{k}_seed{s}.csv 의 해당 PCB 불량만
  - 정상 바탕: train_normal.csv 의 해당 PCB 정상만
  - test.csv, val_normal.csv 이미지는 열 수 없다 (경로·해시 둘 다 검사)
그리고 check_manifest()로 AD README 5.2절 4번의 누수 검사를 한다.

    python -m src.data.shared_splits --check ../../Dataset/generated/copypaste/K2_seed0/manifest.csv
"""
import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .prepare import sha

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TYPES = ('bent', 'melt', 'scratch', 'missing')
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}
PCBS = ('pcb1', 'pcb2', 'pcb3')


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class SharedSplits:
    def __init__(self, splits_dir=None, data_root=None, path_prefix='Dataset/visa/raw'):
        """data_root: 원본 pcb 폴더들의 부모. 기본값은 <프로젝트>/Dataset/visa/raw.
        csv 경로의 path_prefix 부분을 data_root로 바꿔 읽는다(Colab·로컬 위치가 달라도 됨)."""
        self.splits = Path(splits_dir) if splits_dir else PROJECT_ROOT / 'Dataset/visa/splits'
        self.data_root = Path(data_root) if data_root else PROJECT_ROOT / path_prefix
        self.prefix = path_prefix.rstrip('/') + '/'
        forbidden = read_rows(self.splits / 'test.csv') + read_rows(self.splits / 'val_normal.csv')
        self.forbidden_paths = {r['image'] for r in forbidden}
        self.forbidden_sha = {r['image_sha256'] for r in forbidden}
        self.pool = {r['image'] for r in read_rows(self.splits / 'pool.csv')}
        self.train_normal = read_rows(self.splits / 'train_normal.csv')

    def _guard(self, rows):
        for r in rows:
            assert r['image'] not in self.forbidden_paths, f'test/val 이미지 사용 시도: {r["image"]}'
            assert r['image_sha256'] not in self.forbidden_sha, f'test/val과 같은 내용의 이미지: {r["image"]}'
        return rows

    def defect_rows(self, k, seed, pcb):
        """train_K{k}_seed{seed}.csv 에서 pcb의 불량 행. 종류별 K장인지 확인한다."""
        assert pcb in PCBS
        rows = [r for r in read_rows(self.splits / f'train_K{k}_seed{seed}.csv') if r['category'] == pcb]
        assert all(r['image'] in self.pool and r['label'] == '1' for r in rows)
        for t in TYPES:
            n = sum(r['defect_type'] == t for r in rows)
            assert n == k, f'{pcb} {t}: {n}장 ≠ K={k}'
        return self._guard(rows)

    def normal_rows(self, pcb):
        assert pcb in PCBS
        return self._guard([r for r in self.train_normal if r['category'] == pcb])

    def path(self, rel):
        assert rel.startswith(self.prefix), f'예상 밖 경로: {rel}'
        p = (self.data_root / rel[len(self.prefix):]).resolve()
        assert p.is_relative_to(self.data_root.resolve()) and p.is_file(), f'파일 없음: {p}'
        return p

    @lru_cache(maxsize=128)
    def _open(self, image, mask, image_sha):
        assert image not in self.forbidden_paths
        p = self.path(image)
        assert sha(p) == image_sha, f'원본 해시 불일치: {p}'
        with Image.open(p) as im:
            rgb = im.convert('RGB').copy()
        ids = np.asarray(Image.open(self.path(mask))) if mask else np.zeros(rgb.size[::-1], np.uint8)
        return rgb, ids

    def open(self, row):
        """(RGB 이미지, 종류 ID 마스크 배열)을 돌려준다. 정상이면 마스크는 전부 0."""
        self._guard([row])
        return self._open(row['image'], row['mask'], row['image_sha256'])


def check_manifest(manifest_path, splits_dir=None):
    """AD README 5.2절 4번: source_image는 같은 PCB의 train_K 불량, base_normal_image는 train_normal에만 있어야 한다."""
    manifest_path = Path(manifest_path)
    name = manifest_path.parent.name                     # K{k}_seed{s}
    k, seed = name[1:].split('_seed')
    splits = Path(splits_dir) if splits_dir else PROJECT_ROOT / 'Dataset/visa/splits'
    forbidden = {r['image'] for r in read_rows(splits / 'test.csv') + read_rows(splits / 'val_normal.csv')}
    k_rows = {(r['category'], r['image']) for r in read_rows(splits / f'train_K{k}_seed{seed}.csv')}
    normals = {(r['category'], r['image']) for r in read_rows(splits / 'train_normal.csv')}
    rows = read_rows(manifest_path)
    assert rows, 'manifest가 비어 있습니다'
    for r in rows:
        assert r['source_image'] not in forbidden and r['base_normal_image'] not in forbidden, f'test/val 누수: {r["image"]}'
        assert (r['category'], r['source_image']) in k_rows, f'K장 밖의 결함 소스: {r["source_image"]}'
        assert (r['category'], r['base_normal_image']) in normals, f'train_normal 밖의 바탕: {r["base_normal_image"]}'
    return {'status': 'PASS', 'manifest': str(manifest_path), 'rows': len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', nargs='+', required=True, help='검사할 manifest.csv 경로들')
    parser.add_argument('--splits')
    args = parser.parse_args()
    for m in args.check:
        print(check_manifest(m, args.splits))
