"""results.csv → K 곡선 요약, PatchCore 대비 짝지은 비교, 교차 K / 수확체감 K (README 6.2–6.3절).

  python -m AnomalyDetection.ad.analyze --res 512 [--target 0.10]

- 짝: 같은 (PCB, seed). PatchCore·kNN(K=0)은 seed와 무관하므로 모든 seed에 같은 값을 쓴다.
- 유의미: 짝지은 부트스트랩 95% CI가 0을 포함하지 않고(개선 방향), 평균 개선이 2%p 이상.
- 교차 K: PatchCore보다 유의미하게 좋은 가장 작은 K.
- 수확체감 K: 그 다음 K(2배)로 가도 유의미한 개선이 없는 첫 K.
- 목표 K*: (--target 지정 시) 평균 FNR@1%가 목표 이하인 가장 작은 K.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .eval import RESULTS

MIN_GAIN = 0.02


def paired(a, b, n_boot=10000, seed=0):
    """a − b 의 평균과 부트스트랩 95% CI. a, b는 같은 순서의 짝."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def significant_gain(fnr_new, fnr_old):
    """fnr_new 가 fnr_old 보다 유의미하게 낮은가."""
    m, lo, hi = paired(fnr_old, fnr_new)        # 양수 = 개선
    return (lo > 0 and m >= MIN_GAIN), m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, required=True)
    ap.add_argument('--metric', default='fnr_at_fpr1')
    ap.add_argument('--target', type=float)
    ap.add_argument('--results', default=str(RESULTS / 'results.csv'))
    a = ap.parse_args()

    df = pd.read_csv(a.results)
    df = df[(df.res == a.res) & (df.regime == 'main')]
    df = df.drop_duplicates(['model', 'category', 'K', 'seed', 'aug', 'res'], keep='last')
    pc = df[df.model == 'patchcore'].set_index('category')[a.metric]
    seeds = sorted(df[df.K > 0].seed.unique()) or [0]
    if pc.empty:
        print('PatchCore 결과가 없습니다.')
    else:
        print(f'PatchCore K=0 ({a.metric}): ' + ', '.join(f'{c} {v:.3f}' for c, v in pc.items()) + f'  평균 {pc.mean():.3f}')

    print('\n== K 곡선 (평균 ± 표준편차, 3 PCB × seed) ==')
    summary = (df.groupby(['model', 'aug', 'K'])[a.metric].agg(['mean', 'std', 'count']).round(4))
    print(summary.to_string())

    rows = []
    for (model, aug), g in df[df.model != 'patchcore'].groupby(['model', 'aug']):
        ks = sorted(k for k in g.K.unique() if k > 0)
        vec = {}
        for k in ks:
            gk = g[g.K == k].set_index(['category', 'seed'])[a.metric]
            vec[k] = gk
        cross = dim = target = None
        for i, k in enumerate(ks):
            gk = vec[k]
            if not pc.empty:
                base = np.array([pc[c] for c, _ in gk.index])
                sig, m, lo, hi = significant_gain(gk.values, base)
                rows.append(dict(model=model, aug=aug, K=k, vs='patchcore', gain=round(m, 4),
                                 ci=f'[{lo:.3f}, {hi:.3f}]', significant=sig, n=len(gk)))
                if sig and cross is None:
                    cross = k
            if i + 1 < len(ks):
                nxt = vec[ks[i + 1]]
                common = gk.index.intersection(nxt.index)
                if len(common) >= 3:
                    sig2, m2, lo2, hi2 = significant_gain(nxt[common].values, gk[common].values)
                    rows.append(dict(model=model, aug=aug, K=k, vs=f'K={ks[i + 1]}', gain=round(m2, 4),
                                     ci=f'[{lo2:.3f}, {hi2:.3f}]', significant=sig2, n=len(common)))
                    if not sig2 and dim is None and cross is not None:
                        dim = k
            if a.target is not None and target is None and gk.mean() <= a.target:
                target = k
        print(f'\n[{model} / {aug}] 교차 K={cross}  수확체감 K={dim}' + (f'  목표 K*={target}' if a.target else ''))
    if rows:
        print('\n== 짝지은 비교 (gain = 기준 FNR − 새 FNR, 양수가 개선) ==')
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
