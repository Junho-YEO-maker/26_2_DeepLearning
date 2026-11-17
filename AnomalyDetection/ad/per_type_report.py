"""불량 종류별 정리: (1) 결함 제거 진단(D/DR/ND) 판정 비율, (2) 원래 시험셋 FNR@FPR1%를 종류별로.

(1)은 defect_removal_check가 저장한 임베딩 캐시(Dataset/cache/defect_removal/*.npz)와 행별 CSV를 다시 쓴다.
(2)는 prep 전처리로 test·val·train_normal·실제 K장·생성을 임베딩해 none vs +생성(전부)을 헤드 시드 5개로 평균한다.

    python -m AnomalyDetection.ad.per_type_report --aug sd_lora_v3 --prep canon --diag-tag _canon_dil40 [--data-root ...]
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, TYPES, Data
from .eval import evaluate, threshold_at_fpr
from .features import PatchFeatures, device
from .models import EmbeddingHead
from .prep import embedder

MODELS = ('devnet', 'mlp')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prep', default='canon')
    ap.add_argument('--diag-tag', default='_canon_dil40', help='defect_removal_check 결과 파일 접미사')
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(8)

    # (1) 진단: 종류별 D / DR / ND 판정 비율
    rows = list(csv.DictReader(open(ROOT / 'AnomalyDetection/results' / f'defect_removal_K{a.k}_seed{a.seed}{a.diag_tag}.csv')))
    diag = {}      # (model, train, type, key) -> list of flags
    for pcb in PCBS:
        z = np.load(ROOT / 'Dataset/cache/defect_removal' / f'{pcb}_{a.aug}_K{a.k}_seed{a.seed}{a.diag_tag}.npz')
        types = np.array([r['defect_type'] for r in rows if r['pcb'] == pcb])
        assert len(types) == len(z['D'])
        for m in MODELS:
            for train, Xa in (('none', z['Xr']), (f'+{a.aug}', np.concatenate([z['Xr'], z['Xg']]))):
                for hs in range(a.head_seeds):
                    h = EmbeddingHead(m, seed=hs).fit(z['Xn'], Xa)
                    thr = threshold_at_fpr(h.score(z['Xv']), 0.01)
                    for key in ('D', 'DR', 'ND'):
                        flag = h.score(z[key]) > thr
                        for t in TYPES:
                            diag.setdefault((m, train, t, key), []).extend(flag[types == t].tolist())
    print(f'(1) 결함 제거 진단, 종류별 불량 판정 비율 (pool 불량, {a.diag_tag[1:]}, PCB 합침, 헤드 시드 {a.head_seeds}개)')
    print('    D = 실제 불량 사진, DR = 결함 지운 불량 사진(지름길), ND = 정상 사진 + 진짜 결함')
    for m in MODELS:
        for train in ('none', f'+{a.aug}'):
            print(f'  {m:6s} {train:14s} ' + '  '.join(
                f"{t}: D {np.mean(diag[(m, train, t, 'D')]):.2f} DR {np.mean(diag[(m, train, t, 'DR')]):.2f} "
                f"ND {np.mean(diag[(m, train, t, 'ND')]):.2f}" for t in TYPES))

    # (2) 원래 시험셋 종류별 FNR
    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    fnr = {}       # (model, train, type) -> list over pcb×seed
    n_type = {t: 0 for t in TYPES}
    for pcb in PCBS:
        emb_items, _ = embedder(data, ex, dev, pcb, 512, a.prep)
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        Xt, Xv, Xn, Xr, Xg = (emb_items(x) for x in (test, val, nl, real, gen))
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]
        for t in TYPES:
            n_type[t] += sum(1 for x in test if x.label == 1 and t in x.defect_type.split('|'))
        for m in MODELS:
            for train, Xa in (('none', Xr), (f'+{a.aug}', np.concatenate([Xr, Xg]))):
                for hs in range(a.head_seeds):
                    h = EmbeddingHead(m, seed=hs).fit(Xn, Xa)
                    r = evaluate(h.score(Xt), labels, dtypes, h.score(Xv))
                    for t, (n, v) in r['per_type'].items():
                        if n:
                            fnr.setdefault((m, train, t), []).append((n, v))
    print(f'\n(2) 원래 시험셋 FNR@FPR1%, 종류별 (prep={a.prep}, PCB 합침 = 불량 수 가중, 헤드 시드 {a.head_seeds}개). 낮을수록 좋음')
    print('    시험 불량 수: ' + ', '.join(f'{t} {n_type[t]}' for t in TYPES))
    out = []
    for m in MODELS:
        for train in ('none', f'+{a.aug}'):
            vals = {}
            for t in TYPES:
                ps = fnr.get((m, train, t), [])
                vals[t] = sum(n * v for n, v in ps) / max(sum(n for n, _ in ps), 1)
            out.append(dict(model=m, train=train, **vals))
            print(f'  {m:6s} {train:14s} ' + '  '.join(f'{t} {vals[t]:.2f}' for t in TYPES))
    path = ROOT / 'AnomalyDetection/results' / f'per_type_{a.aug}_K{a.k}_seed{a.seed}_{a.prep}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'저장: {path}')


if __name__ == '__main__':
    main()
