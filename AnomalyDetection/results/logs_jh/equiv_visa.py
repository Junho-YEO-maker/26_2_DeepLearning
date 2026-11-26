"""VisA: 실제 K장 + 합성 N장 → '실제만으로 몇 장 쓴 것과 같은가' (등가 실제 장수).
지표: 배경 섞은 시험셋, 과검 5%에서 놓친 불량 비율 (fair_fnr5), 위치 모델 + 촬영 조건 섞기 (seg_aug).
실제만 곡선: K = 1,2,3,4,5,6,8,10 (나누기마다 앞 K가 뒤 K에 포함됨).
곡선은 '실제를 더 넣으면 나빠지지 않는다'로 단조 보정(PAVA, log2 K 위)한 뒤 선형 보간.
불확실성: (나누기, PCB) 15단위를 복원 추출 2000번."""
import sys, glob
import numpy as np, pandas as pd

R = str(__import__('pathlib').Path(__file__).resolve().parents[1])  # AnomalyDetection/results
GEN = sys.argv[1] if len(sys.argv) > 1 else 'ctg'
BASE = {'ctg': '+ctransgan_diverse', 'v3': '+sd_lora_v3'}[GEN]
KR = [1, 2, 3, 4, 5, 6, 8, 10]
NS = [1, 5, 10, 20, 40]
rng = np.random.default_rng(0)

def unit_means(f, train):
    d = pd.read_csv(f)
    d = d[(d.model == 'seg_aug') & (d.train == train)]
    return d.groupby('pcb').fair_fnr5.mean()          # 머리 시드 2개 평균

# 실제만: real[(seed, pcb)][K]
real = {}
for s in range(5):
    for k in KR:
        for p, v in unit_means(f'{R}/fair_ad_K{k}_seed{s}.csv', 'none').items():
            real.setdefault((s, p), {})[k] = v
# 합성: aug[(K, N)][(seed, pcb)]
aug = {}
for f in sorted(glob.glob(f'{R}/fair_ad_K*_seed*_grid_{GEN}.csv')):
    k = int(f.split('_K')[1].split('_')[0]); s = int(f.split('_seed')[1].split('_')[0])
    for n in NS:
        tr = BASE if n == 40 else f'{BASE}_n{n}'
        for p, v in unit_means(f, tr).items():
            aug.setdefault((k, n), {})[(s, p)] = v

def pava_dec(y):
    """단조 비증가 회귀 (같은 가중치)."""
    blocks = [[v, 1] for v in y]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            v2, w2 = out.pop(); v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in out:
        res += [v] * w
    return np.array(res)

LK = np.log2(KR)
def keq(miss, curve):
    c = pava_dec(curve)
    if miss >= c[0]:
        return 1.0 if miss <= curve[0] + 1e-12 else 0.0   # 0.0 = 실제 1장보다도 나쁨
    if miss <= c[-1]:
        return np.inf
    for i in range(len(KR) - 1):
        a, b = c[i], c[i + 1]
        if b <= miss <= a and a > b:
            return float(2 ** (LK[i] + (a - miss) / (a - b) * (LK[i + 1] - LK[i])))
    return np.nan

def fmt(x):
    if x == 0.0: return '<1'
    if np.isinf(x): return '>10'
    return f'{x:.1f}'

units_all = sorted(real)
curve_all = np.array([np.mean([real[u][k] for u in units_all]) for k in KR])
print('실제만 곡선 (15단위 평균, 놓친 불량 %):', ' '.join(f'K{k}={v*100:.1f}' for k, v in zip(KR, curve_all)))
print('  단조 보정 후               :', ' '.join(f'K{k}={v*100:.1f}' for k, v in zip(KR, pava_dec(curve_all))))
per_split = {s: np.array([np.mean([real[(s, p)][k] for p in ('pcb1', 'pcb2', 'pcb3')]) for k in KR]) for s in range(5)}
for s in range(5):
    print(f'  나누기 {s}:', ' '.join(f'{v*100:5.1f}' for v in per_split[s]))
print()
rows = []
for (k, n), d in sorted(aug.items()):
    units = sorted(d)
    seeds = sorted({u[0] for u in units})
    m_aug = np.mean([d[u] for u in units]); m_real = np.mean([real[u][k] for u in units])
    curve = np.array([np.mean([real[u][kk] for u in units]) for kk in KR])
    point = keq(m_aug, curve)
    # 나누기별 등가 장수
    ps = []
    for s in seeds:
        us = [u for u in units if u[0] == s]
        ps.append(keq(np.mean([d[u] for u in us]), np.array([np.mean([real[u][kk] for u in us]) for kk in KR])))
    better = sum(np.mean([d[u] for u in units if u[0] == s]) < np.mean([real[u][k] for u in units if u[0] == s]) for s in seeds)
    # 부트스트랩
    bs, diff = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(units), len(units)); uu = [units[i] for i in idx]
        ma = np.mean([d[u] for u in uu]); cr = np.array([np.mean([real[u][kk] for u in uu]) for kk in KR])
        bs.append(keq(ma, cr)); diff.append(ma - np.mean([real[u][k] for u in uu]))
    bs = np.array(bs); fin = np.where(np.isinf(bs), 1e9, bs)
    lo, hi = np.percentile(fin, [2.5, 97.5]); dlo, dhi = np.percentile(diff, [2.5, 97.5])
    rows.append(dict(K=k, N=n, 나누기=len(seeds), 실제만=f'{m_real*100:.1f}', 합성추가=f'{m_aug*100:.1f}',
                     차이=f'{(m_aug-m_real)*100:+.1f} [{dlo*100:+.1f},{dhi*100:+.1f}]', 좋아진나누기=f'{better}/{len(seeds)}',
                     등가=fmt(point), 등가95=f'[{fmt(lo) if lo < 1e8 else ">10"}, {fmt(hi) if hi < 1e8 else ">10"}]',
                     나누기별=' '.join(fmt(x) for x in ps)))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
