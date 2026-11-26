"""SMT: 실제 K장 + 합성 N장 → '실제만으로 몇 장 쓴 것과 같은가' (등가 실제 장수).
지표: 시험셋 FNR @ 과검 1% (fnr, 낮을수록 좋음)와 AUROC. 모델: 위치 모델 seg_ad (2000 step).
실제만 곡선: K = 1, 2, 4, 8, 16, 32 (있는 만큼). 나누기마다 앞 K가 뒤 K에 포함된다.
곡선은 단조 보정(PAVA, log2 K 위) 후 선형 보간. SMT는 PCB 구분이 없어 단위 = 나누기.
생성 모델 이름: SD+LoRA = sd_lora_fix (사전 공정은 sd_lora와 같은 행이라 둘 다 SD+LoRA로 본다), cTransGAN = ctransgan,
  N 80·160 = *_n160 세트.

    python AnomalyDetection/results/logs_jh/equiv_smt.py
"""
import re
import numpy as np
import pandas as pd

R = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'smt')  # AnomalyDetection/results/smt
KR_ALL = [1, 2, 4, 8, 16, 32]


def pava_dec(y):
    out = []
    for v in y:
        out.append([v, 1])
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            v2, w2 = out.pop(); v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([[v] * w for v, w in out])


def keq(x, Ks, curve, higher_better=False):
    """x를 실제만 곡선에 놓아 등가 K. 0.0 = 실제 최소 K보다 나쁨, inf = 최대 K보다 좋음."""
    s = -1 if higher_better else 1
    c = pava_dec(s * np.asarray(curve)); x = s * x
    lk = np.log2(Ks)
    if x > c[0]:
        return 0.0
    if x <= c[-1]:
        return np.inf
    for i in range(len(Ks) - 1):
        a, b = c[i], c[i + 1]
        if b <= x <= a and a > b:
            return float(2 ** (lk[i] + (a - x) / (a - b) * (lk[i + 1] - lk[i])))
    return float(Ks[0])


def fmt(v, kmax):
    return '<1' if v == 0 else (f'>{kmax}' if np.isinf(v) else f'{v:.1f}')


def gen_of(cond):
    m = re.match(r'(sd_lora_fix|sd_lora|ctransgan|sd_lora_n160|ctransgan_n160):(\d+):0\.5$', cond)
    if not m:
        return None
    g = 'SD+LoRA' if m.group(1).startswith('sd') else 'cTransGAN'
    return g, m.group(1), int(m.group(2))


pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
for cat in ('pre', 'solder'):
    d = pd.read_csv(f'{R}/seg_ad_{cat}.csv')
    d = d[d.steps == 2000]
    real = d[d.cond == 'none'].groupby(['K', 'seed'])[['fnr', 'auroc']].mean()
    print(f'\n######## {cat}  실제만 (나누기별 FNR %)')
    print((real.fnr.unstack('K') * 100).round(1).to_string())
    print('평균:', ' '.join(f'K{k}={v*100:.1f}' for k, v in real.fnr.groupby('K').mean().items()),
          '| AUROC:', ' '.join(f'K{k}={v:.3f}' for k, v in real.auroc.groupby('K').mean().items()))
    rows = []
    for cond in d.cond.unique():
        gm = gen_of(cond)
        if not gm:
            continue
        g, method, n = gm
        if method.endswith('n160') and n <= 40:
            continue
        if cat == 'solder' and method == 'sd_lora':
            continue                                  # 납땜 sd_lora는 고치기 전 세트 → 뺀다
        for (k, s), x in d[d.cond == cond].groupby(['K', 'seed']):
            rows.append(dict(gen=g, method=method, K=k, N=n, seed=s, fnr=x.fnr.mean(), auroc=x.auroc.mean()))
    a = pd.DataFrame(rows)
    # 사전 공정: sd_lora와 sd_lora_fix가 같은 (K, N, 나누기)면 sd_lora_fix 하나만
    a = a.sort_values('method', key=lambda m: m.map({'sd_lora_fix': 0}).fillna(1)).drop_duplicates(['gen', 'K', 'N', 'seed'])
    out = []
    for (g, k, n), x in a.groupby(['gen', 'K', 'N']):
        seeds = sorted(x.seed)
        Ks = [kk for kk in KR_ALL if all((kk, s) in real.index for s in seeds)]
        if k not in Ks:
            continue
        cur = lambda m, ss: np.array([real.loc[[(kk, s) for s in ss], m].mean() for kk in Ks])
        xs = x.set_index('seed')
        res = dict(gen=g, K=k, N=n, 나누기=len(seeds),
                   실제만=real.loc[[(k, s) for s in seeds], 'fnr'].mean() * 100, 합성추가=xs.fnr.mean() * 100)
        res['차이'] = res['합성추가'] - res['실제만']
        res['좋아진나누기'] = f"{int((xs.fnr.values < real.loc[[(k, s) for s in seeds], 'fnr'].values).sum())}/{len(seeds)}"
        res['등가_FNR'] = fmt(keq(xs.fnr.mean(), Ks, cur('fnr', seeds)), Ks[-1])
        res['나누기별'] = ' '.join(fmt(keq(xs.fnr[s], Ks, cur('fnr', [s])), Ks[-1]) for s in seeds)
        res['AUROC'] = f"{real.loc[[(k, s) for s in seeds], 'auroc'].mean():.3f}→{xs.auroc.mean():.3f}"
        res['등가_AUROC'] = fmt(keq(xs.auroc.mean(), Ks, cur('auroc', seeds), higher_better=True), Ks[-1])
        res['곡선K'] = ','.join(map(str, Ks))
        out.append(res)
    o = pd.DataFrame(out).sort_values(['gen', 'K', 'N'])
    print(o.round(1).to_string(index=False))
