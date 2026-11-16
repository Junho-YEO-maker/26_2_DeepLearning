"""생성 불량 빠른 점검: 전체 run(PCB당 수십 분)을 돌리기 전에 생성 결과가 쓸 만한지 몇 분 안에 본다.

    python -m AnomalyDetection.ad.gen_check --aug sd_lora --k 1 --seed 0 [--data-root ...]

임베딩은 run.py와 같은 캐시(Dataset/cache/emb{res})를 쓰고, 없으면 계산해서 채운다. results.csv는 건드리지 않는다.

지표 (PCB × 종류)
  edit     생성 이미지 ↔ 자기 바탕 정상의 거리 / 정상끼리 최근접 거리(중앙값).
           1보다 훨씬 작으면 결함이 특징 공간에서 거의 안 보인다(정상에 불량 라벨을 붙인 셈).
  novelty  가장 가까운 train_normal까지 거리 / 정상끼리 최근접 거리. 생성은 자기 바탕을 빼고 잰다.
           실제 시험 불량의 값(real_novelty)과 비슷해야 한다.
  div      같은 종류 생성끼리 평균 코사인 거리 / 정상끼리 최근접 거리. 클수록 다양.
PCB별
  sep      실제 시험 불량 vs 생성 불량을 로지스틱 회귀로 구분한 5-fold AUROC. 0.5에 가까울수록 좋다
           (copypaste 0.95–0.99, 13.3절).
  FNR      DevNet·MLP를 실제 K장 + 생성으로 학습한 FNR@FPR1%. none(실제만), concat(run.py와 같음),
           balanced(실제:생성 = 1:1이 되게 실제를 복제).
"""
import argparse
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .data import PCBS, TYPES, Data
from .eval import evaluate
from .features import PatchFeatures, device
from .models import EmbeddingHead
from .run import Embedder


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', required=True)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--data-root')
    ap.add_argument('--no-fnr', action='store_true', help='FNR 학습을 건너뛰고 거리 지표만')
    a = ap.parse_args()

    data, dev = Data(a.data_root), device()
    E = Embedder(data, PatchFeatures().to(dev), dev, a.res)
    manifest = data.generated / a.aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv'
    with open(manifest, newline='', encoding='utf-8-sig') as f:
        base_of = {r['image']: r['base_normal_image'] for r in csv.DictReader(f)}
    summary = []
    for pcb in a.pcbs:
        test, val, normals = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        Xt, Xv, Xn, Xr, Xg = (E.items(x) for x in (test, val, normals, real, gen))
        N, G = unit(Xn), unit(Xg)
        idx = {it.image: i for i, it in enumerate(normals)}
        sim_nn = N @ N.T
        np.fill_diagonal(sim_nn, -1)
        scale = float(np.median(1 - sim_nn.max(1)))          # 정상끼리 최근접 거리
        print(f'\n{pcb}  (정상끼리 최근접 거리 {scale:.4f})')
        print(f'  {"type":8s} {"edit":>6s} {"novelty":>8s} {"real_novelty":>12s} {"div":>6s}')
        for t in TYPES:
            gi = [i for i, it in enumerate(gen) if it.defect_type == t]
            ri = [i for i, it in enumerate(test) if it.label == 1 and t in it.defect_type.split('|')]
            if not gi:
                continue
            s = G[gi] @ N.T
            b = np.array([idx[base_of[gen[i].image]] for i in gi])
            edit = 1 - s[np.arange(len(gi)), b]
            s[np.arange(len(gi)), b] = -1
            nov = 1 - s.max(1)
            real_nov = 1 - (unit(Xt[ri]) @ N.T).max(1) if ri else np.array([np.nan])
            gg = G[gi] @ G[gi].T
            div = 1 - gg[np.triu_indices(len(gi), 1)].mean()
            print(f'  {t:8s} {edit.mean() / scale:6.2f} {nov.mean() / scale:8.2f} {real_nov.mean() / scale:12.2f} '
                  f'{div / scale:6.2f}')
        R = Xt[[i for i, it in enumerate(test) if it.label == 1]]

        def separability(B):
            X, y = np.concatenate([R, B]), np.r_[np.zeros(len(R)), np.ones(len(B))]
            return cross_val_score(LogisticRegression(max_iter=2000, class_weight='balanced'), unit(X), y,
                                   cv=5, scoring='roc_auc').mean()
        sep = separability(Xg)
        sep_n = separability(Xn[np.random.default_rng(0).choice(len(Xn), min(len(Xg), len(Xn)), replace=False)])
        row = dict(pcb=pcb, sep=sep, sep_normal=sep_n)
        line = (f'  sep AUROC 실제 시험 불량 vs 생성 = {sep:.3f}  (대조: vs train_normal = {sep_n:.3f}. '
                f'두 값이 비슷하면 생성이 정상과 다를 바 없음)')
        if not a.no_fnr:
            rep = max(len(Xg) // max(len(Xr), 1), 1)
            mixes = {'none': Xr, 'concat': np.concatenate([Xr, Xg]),
                     'balanced': np.concatenate([np.repeat(Xr, rep, 0), Xg])}
            for kind in ('devnet', 'mlp'):
                for name, Xa in mixes.items():
                    h = EmbeddingHead(kind, seed=a.seed).fit(Xn, Xa)
                    m = evaluate(h.score(Xt), [it.label for it in test], [it.defect_type for it in test], h.score(Xv))
                    row[f'{kind}_{name}'] = m['fnr_at_fpr1']
            line += '\n  FNR@1% ' + '  '.join(f'{k}={v:.3f}' for k, v in row.items()
                                              if k not in ('pcb', 'sep', 'sep_normal'))
        print(line)
        summary.append(row)
    print('\n평균')
    for key in summary[0]:
        if key != 'pcb':
            print(f'  {key:16s} {np.mean([r[key] for r in summary]):.3f}')


if __name__ == '__main__':
    main()
