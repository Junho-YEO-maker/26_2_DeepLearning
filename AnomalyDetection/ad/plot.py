"""K 곡선 그림 (README 8절 1·2번).

  python -m AnomalyDetection.ad.plot --res 512 --aug none           # 모델별 K 곡선 (실제 불량만)
  python -m AnomalyDetection.ad.plot --res 512 --model devnet       # 한 모델의 증강별 K 곡선

가로축 K(로그, K=0은 왼쪽 끝), 세로축 FNR@FPR1%. 선 = 3 PCB × seed 평균, 띠 = 부트스트랩 95% CI.
PatchCore K=0은 수평 점선.
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .eval import RESULTS  # noqa: E402

from matplotlib import font_manager  # noqa: E402
_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = [f for f in ('AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans') if f in _have]
plt.rcParams['axes.unicode_minus'] = False
X0 = 0.5          # K=0을 놓을 가로 위치(로그 축)


def ci(v, n=5000, seed=0):
    v = np.asarray(v)
    b = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--aug', default='none')
    ap.add_argument('--model')
    ap.add_argument('--metric', default='fnr_at_fpr1')
    ap.add_argument('--out')
    a = ap.parse_args()
    df = pd.read_csv(RESULTS / 'results.csv')
    df = df[df.res == a.res].drop_duplicates(['model', 'category', 'K', 'seed', 'aug', 'res'], keep='last')
    pc = df[df.model == 'patchcore'][a.metric]

    if a.model:
        groups = [(aug, g) for aug, g in df[df.model == a.model].groupby('aug')]
        title, out = f'{a.model}: 증강별 K 곡선 (해상도 {a.res})', a.out or RESULTS / f'kcurve_{a.model}_{a.res}.png'
    else:
        sub = df[(df.model != 'patchcore') & (df.aug == a.aug)]
        groups = list(sub.groupby('model'))
        title, out = f'모델별 K 곡선 (증강: {a.aug}, 해상도 {a.res})', a.out or RESULTS / f'kcurve_{a.aug}_{a.res}.png'
    # K=0 kNN 결과는 증강과 무관하므로 모든 증강 곡선의 시작점으로 쓴다
    k0 = df[(df.model == 'knn_mlp') & (df.K == 0)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, g in groups:
        if a.model == 'knn_mlp' or (not a.model and name == 'knn_mlp'):
            g = pd.concat([g[g.K > 0], k0])
        ks = sorted(g.K.unique())
        xs = [k if k > 0 else X0 for k in ks]
        mean = [g[g.K == k][a.metric].mean() for k in ks]
        lo, hi = zip(*[ci(g[g.K == k][a.metric]) for k in ks])
        ax.plot(xs, mean, marker='o', label=name)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    if len(pc):
        ax.axhline(pc.mean(), ls='--', color='gray', label=f'PatchCore K=0 ({pc.mean():.3f})')
    ax.set_xscale('log')
    ax.set_xticks([X0, 1, 2, 4, 8])
    ax.set_xticklabels(['0', '1', '2', '4', '8'])
    ax.set_xlabel('K (PCB당 결함 종류별 실제 불량 장수)')
    ax.set_ylabel('미검률 @ 과검률 1%')
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    main()
