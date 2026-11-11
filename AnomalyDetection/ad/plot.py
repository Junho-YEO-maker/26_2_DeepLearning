"""K 곡선 그림 (README 8절 1·2번).

  python -m AnomalyDetection.ad.plot --res 512 --aug none           # 모델별 K 곡선 (실제 불량만)
  python -m AnomalyDetection.ad.plot --res 512 --model devnet       # 한 모델의 증강별 K 곡선
  python -m AnomalyDetection.ad.plot --res 512 --per-type           # 결함 종류별 K 곡선 (2×2, 증강 none)

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
XLABEL = 'K (PCB당 결함 종류별 실제 불량 장수)'
YLABEL = '미검률 @ 과검률 1%'
TYPE_KO = {'bent': 'Bent (휜 핀)', 'melt': 'Melt (녹음)', 'scratch': 'Scratch (긁힘)', 'missing': 'Missing (누락)'}


def ci(v, n=5000, seed=0):
    v = np.asarray(v)
    b = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def draw(ax, groups, k0, pc_value, metric):
    """groups: [(이름, df)]. k0: kNN K=0 행(knn_mlp 곡선 시작점, 없으면 None). pc_value: PatchCore 수평선 값."""
    allk = set()
    for name, g in groups:
        if name == 'knn_mlp' and k0 is not None:
            g = pd.concat([g[g.K > 0], k0])
        ks = sorted(g.K.unique())
        allk |= set(ks)
        xs = [k if k > 0 else X0 for k in ks]
        mean = [g[g.K == k][metric].mean() for k in ks]
        lo, hi = zip(*[ci(g[g.K == k][metric]) for k in ks])
        ax.plot(xs, mean, marker='o', label=name)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    if pc_value is not None:
        ax.axhline(pc_value, ls='--', color='gray', label=f'PatchCore K=0 ({pc_value:.3f})')
    ax.set_xscale('log')
    ticks = sorted(allk | {0})
    ax.set_xticks([k if k > 0 else X0 for k in ticks])
    ax.set_xticklabels([str(k) for k in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)


def per_type(a):
    df = pd.read_csv(RESULTS / 'results_per_type.csv')
    df = df[(df.res == a.res) & (df.aug == 'none')]
    df = df.drop_duplicates(['model', 'category', 'K', 'seed', 'aug', 'res', 'defect_type'], keep='last')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, t in zip(axes.flat, ('bent', 'missing', 'melt', 'scratch')):
        d = df[df.defect_type == t]
        pc = d[d.model == 'patchcore']
        k0 = d[(d.model == 'knn_mlp') & (d.K == 0)]
        draw(ax, list(d[d.model != 'patchcore'].groupby('model')), k0,
             pc[a.metric].mean() if len(pc) else None, a.metric)
        ax.set_title(f'{TYPE_KO[t]} · 시험 불량 {int(pc.n_test.sum())}장 (3 PCB 합)')
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(XLABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL)
    fig.suptitle(f'결함 종류별 K 곡선 (증강 없음, 해상도 {a.res}, 3 PCB × 5 seed, 띠 = 95% CI)')
    fig.tight_layout()
    out = a.out or RESULTS / f'kcurve_per_type_{a.res}.png'
    fig.savefig(out, dpi=150)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--aug', default='none')
    ap.add_argument('--model')
    ap.add_argument('--metric', default='fnr_at_fpr1')
    ap.add_argument('--out')
    ap.add_argument('--per-type', action='store_true')
    a = ap.parse_args()
    if a.per_type:
        return per_type(a)

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
    # K=0 kNN 결과는 증강과 무관하므로 knn_mlp 곡선(모든 증강)의 시작점으로 쓴다
    k0 = df[(df.model == 'knn_mlp') & (df.K == 0)]
    if a.model == 'knn_mlp':
        groups = [(name, pd.concat([g[g.K > 0], k0])) for name, g in groups]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw(ax, groups, None if a.model else k0, pc.mean() if len(pc) else None, a.metric)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    main()
