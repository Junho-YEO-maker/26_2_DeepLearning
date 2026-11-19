"""AIHub SMT 분할표(Dataset/aihub_smt/splits)를 읽는 데이터 관문. ad.data.Data와 같은 인터페이스.

VisA용 ad/data.py는 건드리지 않고 따로 둔다(다른 작업과 충돌 방지). 차이:
  - category = 'pre' | 'solder', 결함 종류는 분할표(pool.csv)에서 읽는다.
  - 마스크가 없다(박스는 label_json). 이미지는 이미 512×512 JPEG라 배열 캐시는 하지 않는다.
  - prep: 'none' 원본, 'photonorm' 이미지마다 채널 평균을 그 category train_normal 평균으로 옮긴다(밝기·색 지름길 제거).
    임베딩 캐시는 prep마다 따로(Dataset/cache/smt/{prep}/emb{res}).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..ad.data import ROOT, Item, read_csv

SPLITS = ROOT / 'Dataset/aihub_smt/splits'
GENERATED = ROOT / 'Dataset/aihub_smt/generated'     # AIHub 파생물이라 git에 올리지 않는다(.gitignore)
CACHE = ROOT / 'Dataset/cache/smt'
CATEGORIES = ('pre', 'solder')
PREPS = ('none', 'photonorm')


class SmtData:
    def __init__(self, prep='none', splits=SPLITS, cache_root=CACHE):
        assert prep in PREPS, prep
        self.prep, self.splits = prep, Path(splits)
        self.cache = Path(cache_root) / prep          # Embedder가 cache/emb{res}에 임베딩을 둔다
        self._test = read_csv(self.splits / 'test.csv')
        self._val = read_csv(self.splits / 'val_normal.csv')
        self.forbidden = {r['image'] for r in self._test + self._val}
        self.forbidden_sha = {r['image_sha256'] for r in self._test + self._val}
        pool = read_csv(self.splits / 'pool.csv')
        self.types = {c: sorted({r['defect_type'] for r in pool if r['category'] == c}) for c in CATEGORIES}
        self._cat_of = {}
        for name in ('test', 'val_normal', 'train_normal', 'extra_normal', 'pool'):
            for r in read_csv(self.splits / f'{name}.csv'):
                self._cat_of[r['image']] = r['category']
        self._mu = {}

    @staticmethod
    def _items(rows, cat):
        return [Item(r['image'], int(r['label']), r.get('defect_type', '') or '', 'real')
                for r in rows if r['category'] == cat]

    def test(self, cat):
        return self._items(self._test, cat)

    def val_normal(self, cat):
        return self._items(self._val, cat)

    def train_normal(self, cat):
        return self._guard(self._items(read_csv(self.splits / 'train_normal.csv'), cat))

    def extra_normal(self, cat):
        return self._guard(self._items(read_csv(self.splits / 'extra_normal.csv'), cat))

    def train_defects(self, cat, k, seed):
        if k == 0:
            return []
        rows = [r for r in read_csv(self.splits / f'train_K{k}_seed{seed}.csv') if r['category'] == cat]
        for t in self.types[cat]:
            assert sum(r['defect_type'] == t for r in rows) == k, (cat, t, k)
        assert not any(r['image_sha256'] in self.forbidden_sha for r in rows)
        return self._guard(self._items(rows, cat))

    def generated_defects(self, method, cat, k, seed):
        """Dataset/aihub_smt/generated/{method}/K{k}_seed{s}/manifest.csv 의 해당 category 생성 불량.
        source_image는 그 조건의 train_K 안, base_normal_image는 train_normal 안이어야 한다(test/val 차단)."""
        return self._generated(method, cat, k, seed, 1)

    def generated_normals(self, method, cat, k, seed):
        """같은 manifest의 label 0 행(smt_sd_real의 '붙이기 정상'). 없으면 빈 목록. source·바탕 모두 train_normal 안이어야 한다."""
        return self._generated(method, cat, k, seed, 0)

    def _generated(self, method, cat, k, seed, label):
        manifest = GENERATED / method / f'K{k}_seed{seed}' / 'manifest.csv'
        assert manifest.exists(), f'생성 결과가 없습니다: {manifest}'
        rows = [r for r in read_csv(manifest) if r['category'] == cat]
        assert rows, f'{manifest}에 {cat} 행이 없습니다'
        rows = [r for r in rows if int(r['label']) == label]
        base = {r['image'] for r in read_csv(self.splits / 'train_normal.csv') if r['category'] == cat}
        src = base if label == 0 else \
            {r['image'] for r in read_csv(self.splits / f'train_K{k}_seed{seed}.csv') if r['category'] == cat}
        bad_src = [r['source_image'] for r in rows if r['source_image'] not in src]
        bad_base = [r['base_normal_image'] for r in rows if r['base_normal_image'] not in base]
        assert not bad_src and not bad_base, f'누수: source {bad_src[:2]} base {bad_base[:2]}'
        return self._guard([Item(r['image'], label, r['defect_type'] if label else '', method) for r in rows])

    def _guard(self, items):
        bad = [it.image for it in items if it.image in self.forbidden]
        assert not bad, f'학습 집합에 test/val 이미지: {bad[:3]}'
        return items

    # ---------- 이미지 읽기 ----------
    def path(self, rel):
        return ROOT / rel

    def _raw(self, rel, res):
        with Image.open(self.path(rel)) as im:
            im = im.convert('RGB')
            if im.size != (res, res):
                im = im.resize((res, res), Image.BILINEAR)
            return np.asarray(im)

    def _normal_mean(self, cat, res):
        """category train_normal의 이미지별 채널 평균의 평균 (photonorm 기준). 한 번 계산해 json으로 둔다."""
        key = (cat, res)
        if key not in self._mu:
            p = self.cache / f'normal_mean_{cat}_{res}.json'
            if p.exists():
                self._mu[key] = np.array(json.loads(p.read_text()))
            else:
                ms = [self._raw(it.image, res).reshape(-1, 3).mean(0) for it in self.train_normal(cat)]
                self._mu[key] = np.mean(ms, 0)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(self._mu[key].tolist()))
        return self._mu[key]

    def load(self, rel, res):
        arr = self._raw(rel, res)
        if self.prep == 'photonorm':
            mu = self._normal_mean(self._cat_of[rel], res)
            arr = np.clip(arr.astype(np.float32) - arr.reshape(-1, 3).mean(0) + mu, 0, 255).astype(np.uint8)
        return arr
