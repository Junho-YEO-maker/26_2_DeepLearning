"""생성 세트에서 PCB·종류마다 n장만 고른 manifest. AD에 넣는 합성 장수 실험용 (11/18).

이미지는 복사하지 않고 원래 세트의 경로를 그대로 가리킨다. 고르는 순서는 (aug, k, seed)마다 고정한 무작위 순서라
n이 커져도 앞에서 고른 장은 그대로다(nested: n=1 ⊂ n=5 ⊂ n=10 ...).

    cd Augmentation
    python -m gen_subset --aug sd_lora_v3 --k 1 --seed 0 1 2 --n 1 5 10 20
결과: Dataset/generated/{aug}_n{n}/K{k}_seed{s}/manifest.csv (끝에 누수 검사)
"""
import argparse
import csv
import zlib
from collections import defaultdict

import numpy as np

from common.kshot import GENERATED, MANIFEST_FIELDS, check_manifest, read_rows


def subset(aug, k, seed, ns, force=False):
    manifest = GENERATED / aug / f'K{k}_seed{seed}' / 'manifest.csv'
    if not manifest.exists():
        print(f'없음, 건너뜀: {manifest}')
        return
    rows = read_rows(manifest)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['category'], r['defect_type'])].append(r)
    rng = np.random.default_rng([zlib.crc32(aug.encode()), k, seed])
    order = {g: [rs[i] for i in rng.permutation(len(rs))] for g, rs in sorted(groups.items())}
    for n in ns:
        out = GENERATED / f'{aug}_n{n}' / f'K{k}_seed{seed}' / 'manifest.csv'
        if out.exists() and not force:
            print(f'이미 있음: {out}')
            continue
        short = {g: len(rs) for g, rs in order.items() if len(rs) < n}
        assert not short, f'{aug} K{k}_seed{seed}: 종류당 {n}장이 안 되는 곳 {short}'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for g in sorted(order):
                w.writerows(order[g][:n])
        print(check_manifest(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aug', required=True)
    p.add_argument('--k', type=int, nargs='+', required=True)
    p.add_argument('--seed', type=int, nargs='+', default=[0])
    p.add_argument('--n', type=int, nargs='+', required=True, help='PCB·종류마다 고를 장수')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    for k in a.k:
        for seed in a.seed:
            subset(a.aug, k, seed, a.n, a.force)


if __name__ == '__main__':
    main()
