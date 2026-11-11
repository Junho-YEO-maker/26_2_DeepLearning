"""분할표(Dataset/visa/splits)와 생성 manifest(Dataset/generated)를 읽는 AnomalyDetection 팀의 데이터 관문.

- csv 안의 경로는 프로젝트 루트 기준 상대경로(`Dataset/visa/raw/...`)다.
  원본이 다른 곳(Colab /content 등)에 있으면 --data-root로 `Dataset/visa/raw` 부분만 바꿔 읽는다.
- 이미지는 정사각형(res×res)으로 줄여 uint8 배열로 디스크에 캐시한다(Dataset/cache, git 제외).
- test.csv 행은 평가에서만 쓰고, 학습 집합을 만드는 함수는 test/val 경로가 섞이면 바로 멈춘다.
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / 'Dataset/visa/splits'
GENERATED = ROOT / 'Dataset/generated'
RAW_PREFIX = 'Dataset/visa/raw/'
PCBS = ('pcb1', 'pcb2', 'pcb3')
TYPES = ('bent', 'melt', 'scratch', 'missing')


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class Item:
    image: str          # 프로젝트 루트 기준 상대경로
    label: int          # 0 정상, 1 불량
    defect_type: str    # 정상은 '', 섞인 이미지는 'melt|scratch'
    source: str         # 'real' 또는 생성 방법 이름


class Data:
    def __init__(self, data_root=None, splits=SPLITS, generated=GENERATED, cache=ROOT / 'Dataset/cache'):
        self.data_root = Path(data_root) if data_root else ROOT / RAW_PREFIX
        self.splits = Path(splits)
        self.generated = Path(generated)
        self.cache = Path(cache)
        self._test = read_csv(self.splits / 'test.csv')
        self._val = read_csv(self.splits / 'val_normal.csv')
        self.forbidden = {r['image'] for r in self._test + self._val}
        self.forbidden_sha = {r['image_sha256'] for r in self._test + self._val}

    # ---------- 행 → Item ----------
    @staticmethod
    def _items(rows, pcb):
        return [Item(r['image'], int(r['label']), r.get('defect_type', '') or '', 'real')
                for r in rows if r['category'] == pcb]

    def test(self, pcb):
        return self._items(self._test, pcb)

    def val_normal(self, pcb):
        return self._items(self._val, pcb)

    def train_normal(self, pcb):
        return self._guard(self._items(read_csv(self.splits / 'train_normal.csv'), pcb))

    def train_defects(self, pcb, k, seed):
        if k == 0:
            return []
        rows = [r for r in read_csv(self.splits / f'train_K{k}_seed{seed}.csv') if r['category'] == pcb]
        for t in TYPES:
            assert sum(r['defect_type'] == t for r in rows) == min(k, 8 if (pcb, t) == ('pcb2', 'bent') else 10)
        assert not any(r['image_sha256'] in self.forbidden_sha for r in rows)
        return self._guard(self._items(rows, pcb))

    def generated_defects(self, method, pcb, k, seed):
        """Dataset/generated/{method}/K{k}_seed{s}/manifest.csv 의 해당 PCB 생성 불량. 누수 검사를 통과해야 한다."""
        manifest = self.generated / method / f'K{k}_seed{seed}' / 'manifest.csv'
        assert manifest.exists(), f'생성 결과가 없습니다: {manifest}'
        check_leak(manifest, self.splits)
        rows = [r for r in read_csv(manifest) if r['category'] == pcb]
        assert rows, f'{manifest}에 {pcb} 행이 없습니다'
        return [Item(r['image'], 1, r['defect_type'], method) for r in rows]

    def _guard(self, items):
        bad = [it.image for it in items if it.image in self.forbidden]
        assert not bad, f'학습 집합에 test/val 이미지: {bad[:3]}'
        return items

    # ---------- 이미지 읽기 ----------
    def path(self, rel):
        if rel.startswith(RAW_PREFIX):
            return self.data_root / rel[len(RAW_PREFIX):]
        return ROOT / rel

    def load(self, rel, res):
        """res×res RGB uint8 배열. 원본(raw)만 디스크에 캐시한다(생성 이미지는 수만 장이라 임베딩만 캐시)."""
        cacheable = rel.startswith(RAW_PREFIX)
        cp = self.cache / str(res) / (hashlib.sha1(rel.encode()).hexdigest()[:16] + '.npy')
        if cacheable and cp.exists():
            return np.load(cp)
        with Image.open(self.path(rel)) as im:
            arr = np.asarray(im.convert('RGB').resize((res, res), Image.BILINEAR))
        if cacheable:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(cp, arr)
        return arr


def check_leak(manifest, splits=SPLITS):
    """README 5.2절 4번: manifest의 source_image·base_normal_image가 test/val에 없고,
    source_image는 같은 K·seed의 학습 불량, base_normal_image는 train_normal 안에 있어야 한다."""
    manifest = Path(manifest)
    k, seed = manifest.parent.name[1:].split('_seed')
    splits = Path(splits)
    forbidden = {r['image'] for r in read_csv(splits / 'test.csv') + read_csv(splits / 'val_normal.csv')}
    k_rows = {(r['category'], r['image']) for r in read_csv(splits / f'train_K{k}_seed{seed}.csv')}
    normals = {(r['category'], r['image']) for r in read_csv(splits / 'train_normal.csv')}
    rows = read_csv(manifest)
    for r in rows:
        assert r['source_image'] not in forbidden, f'누수(source): {r["image"]}'
        assert (r['category'], r['source_image']) in k_rows, f'K장 밖 소스: {r["source_image"]}'
        base = r.get('base_normal_image', '')
        if base:
            assert base not in forbidden, f'누수(base): {r["image"]}'
            assert (r['category'], base) in normals, f'train_normal 밖 바탕: {base}'
    return len(rows)
