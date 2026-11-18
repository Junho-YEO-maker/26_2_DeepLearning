"""발표용 표와 그림을 결과 파일에서 한 번에 만든다 (11/18, ppt_preview.md 채울 실험 3). 숫자를 손으로 옮기지 않기 위해서.

입력 (AnomalyDetection/results/)
  fair_ad_K{k}_seed{s}[_v3].csv     공정 시험셋 (ad/fair_ad.py). 헤드 시드 평균 → PCB 평균 → 분할 seed 평균 ± 표준편차
  gradcam_K1_seed0.csv               Grad-CAM / seg 결함 지도 (ad/gradcam.py --summarize)
  local_ad_{img,seg}_nogen_K1_seed*.csv   표준 시험셋, 이미지 vs 위치 (ad/local_ad.py)
  results.csv                        표준 시험셋 K 곡선 (512, 실제만)
출력 (AnomalyDetection/results/ppt/)
  summary.md          슬라이드별 표
  fair_kcurve.png     공정 시험셋 K 곡선 | 원래 시험셋 K 곡선 (둘 다 과검률 5%)
  fair_kcurve_pcb.png PCB별 공정 K 곡선

    python -m AnomalyDetection.ad.ppt_summary
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import ROOT  # noqa: E402

RES = ROOT / 'AnomalyDetection/results'
OUT = RES / 'ppt'
KS = [1, 2, 4, 8, 10]
TYPES = ['bent', 'melt', 'scratch', 'missing']
NAME = {'patchcore': 'PatchCore', 'seg': '위치 seg', 'seg_aug': '위치 seg + 촬영 무작위화',
        'devnet': '이미지 DevNet', 'mlp': 'MLP'}
TRAIN = {'none': '실제만', '+sd_lora_v3': '+ SD+LoRA v3', '+sd_lora': '+ SD+LoRA v2',
         '+copypaste': '+ copy-paste', '+anomalydiffusion': '+ AnomalyDiffusion'}
MAIN = [('patchcore', 'none'), ('seg', 'none'), ('seg', '+sd_lora_v3'), ('seg_aug', 'none'),
        ('seg_aug', '+sd_lora_v3'), ('seg_aug', '+copypaste'), ('devnet', 'none'), ('devnet', '+sd_lora_v3')]


def label(m, t):
    return f'{NAME.get(m, m)}, {"정상만" if m == "patchcore" else TRAIN.get(t, t)}'


def load_fair():
    """분할 seed마다 한 줄: (k, seed, model, train) → PCB 평균 지표. pcb열은 PCB별 값."""
    rows = []
    for f in sorted(RES.glob('fair_ad_K*_seed*.csv')):
        m = re.fullmatch(r'fair_ad_K(\d+)_seed(\d+)(_v3)?\.csv', f.name)
        if m:
            rows.append(pd.read_csv(f))
    d = pd.concat(rows)
    num = [c for c in d.columns if c not in ('pcb', 'k', 'seed', 'model', 'train', 'head_seed')]
    per_pcb = d.groupby(['k', 'seed', 'model', 'train', 'pcb'])[num].mean().reset_index()
    return per_pcb


def stats(per_pcb, col, pcb=None):
    """(model, train, k) → (평균, 표준편차, seed 수). pcb=None이면 PCB 평균을 먼저 낸다."""
    x = per_pcb if pcb is None else per_pcb[per_pcb.pcb == pcb]
    s = x.groupby(['k', 'seed', 'model', 'train'])[col].mean()   # seed마다 PCB 평균
    if pcb is None:
        n_pcb = x.groupby(['k', 'seed', 'model', 'train']).pcb.nunique()
        s = s[n_pcb == 3]
    g = s.groupby(['model', 'train', 'k'])
    return pd.DataFrame({'mean': g.mean(), 'sd': g.std(ddof=1), 'n': g.size()})


def fmt(mu, sd=np.nan, n=1, d=3):
    if pd.isna(mu):
        return '—'
    n = int(n)
    s = f'{mu:.{d}f}'.lstrip('0') if mu < 1 else f'{mu:.{d}f}'
    if n > 1 and not pd.isna(sd):
        s += f' ± {sd:.{d}f}'.replace('0.', '.')
    return s + (f' ({n})' if n > 1 else '')


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ---------------------------------------------------------------- 그림
COL = {'seg': '#0072B2', 'seg_aug': '#009E73', 'devnet': '#D55E00'}   # Okabe-Ito (색각 이상 구분용)
GRAY, INK, MUTED, GRID = '#6b6b6b', '#1a1a1a', '#5a5a5a', '#e4e3df'
LINES = [('seg', 'none', 'Seg (real only)'), ('seg', '+sd_lora_v3', 'Seg + SD-LoRA'),
         ('seg_aug', 'none', 'Seg + capture aug'), ('seg_aug', '+sd_lora_v3', 'Seg + capture aug + SD-LoRA'),
         ('devnet', 'none', 'Image DevNet (real only)'), ('devnet', '+sd_lora_v3', 'Image DevNet + SD-LoRA')]


def draw(ax, st, pc, title, ylabel):
    for m, t, lab in LINES:
        if (m, t) not in st.index.droplevel('k'):
            continue
        s = st.loc[(m, t)].reindex(KS).dropna(subset=['mean'])
        ls = '-' if t == 'none' else '--'
        ax.plot(s.index, s['mean'], ls, color=COL[m], lw=2, label=lab, zorder=3)
        multi = s['n'] > 1
        ax.scatter(s.index[multi], s['mean'][multi], s=40, color=COL[m], edgecolor='white', lw=1.5, zorder=4)
        ax.scatter(s.index[~multi], s['mean'][~multi], s=40, facecolor='white', edgecolor=COL[m], lw=1.5, zorder=4)
        band = s[multi & s['sd'].notna()]
        if len(band) > 1:
            ax.fill_between(band.index, band['mean'] - band['sd'], band['mean'] + band['sd'],
                            color=COL[m], alpha=0.10, lw=0, zorder=1)
    if pc is not None:
        ax.axhline(pc, color=GRAY, ls=':', lw=2, zorder=2, label='PatchCore (normal only)')
    ax.set_xscale('log')
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlim(0.9, 11)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_xlabel('Real defects per type in training (K)', fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def figures(per_pcb):
    f5, s5 = stats(per_pcb, 'fair_fnr5'), stats(per_pcb, 'fnr5')
    pc = lambda st: st.loc[('patchcore', 'none', 1), 'mean'] if ('patchcore', 'none', 1) in st.index else None  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    draw(ax[0], f5, pc(f5), 'A. Capture-mixed (fair) test', 'Missed defects (FNR) at 5% false alarms')
    draw(ax[1], s5, pc(s5), 'B. Original VisA test (defect photos captured differently)', '')
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.01, 0.97, 'VisA PCB1-3 mean. Filled dot = mean of split seeds (band = ±1 sd), hollow = seed 0 only.',
             fontsize=9, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(OUT / 'fair_kcurve.png', dpi=150, facecolor='white')
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for i, p in enumerate(['pcb1', 'pcb2', 'pcb3']):
        st = stats(per_pcb, 'fair_fnr5', pcb=p)
        draw(ax[i], st, pc(st), p.upper(), 'Fair test FNR at 5% false alarms' if i == 0 else '')
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / 'fair_kcurve_pcb.png', dpi=150, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- 표
def kcurve_table(per_pcb, col):
    st = stats(per_pcb, col)
    rows = []
    for m, t in MAIN:
        if (m, t) not in st.index.droplevel('k'):
            continue
        s = st.loc[(m, t)]
        rows.append([label(m, t)] + [fmt(*s.loc[k]) if k in s.index else '—' for k in KS])
    return md_table(['모델, 학습'] + [f'K={k}' for k in KS], rows)


def section_fair(per_pcb):
    L = ['## 공정 시험셋 (슬라이드 14)', '',
         '값 = 평균 ± 표준편차 (분할 seed 수). seed가 하나면 값만. 헤드 시드 평균 → PCB 평균 → 분할 seed 평균.',
         'PatchCore는 정상만 학습해 K와 무관하므로 K=1 seed 0 값 하나다.', '',
         '### 공정 K 곡선: 미검률 @ 과검률 5% (공정 시험셋, 촬영 조건 반반)', '', kcurve_table(per_pcb, 'fair_fnr5'), '',
         '### 같은 모델의 원래 시험셋 미검률 @ 과검률 1% (지름길이 있는 시험셋)', '', kcurve_table(per_pcb, 'fnr'), '',
         '### 공정 시험셋 AUROC', '', kcurve_table(per_pcb, 'fair_auroc'), '']

    # K=1 생성기 비교 (seed 0, 모든 생성기)
    k1 = per_pcb[(per_pcb.k == 1) & (per_pcb.seed == 0)]
    g = k1.groupby(['model', 'train'])[['fair_fnr5', 'fair_auroc', 'fnr', 'v_D', 'v_ND', 'v_DR']].mean()
    order = ['none', '+sd_lora_v3', '+copypaste', '+sd_lora', '+anomalydiffusion']
    rows = []
    for m in ['patchcore', 'seg', 'seg_aug', 'devnet', 'mlp']:
        for t in order:
            if (m, t) in g.index:
                r = g.loc[(m, t)]
                rows.append([label(m, t)] + [fmt(r[c]) for c in g.columns])
    L += ['### K=1 seed 0, 생성기별', '',
          'v_D / v_ND / v_DR = 정상 검증 사진으로 정한 과검률 1% 임계값에서 불량 판정 비율. '
          'D 시험 불량, ND 정상 + 결함(높을수록 좋음), DR 결함 지운 불량(낮을수록 좋음).', '',
          md_table(['모델, 학습', '공정@5', '공정 AUROC', '원래 시험셋 @1', 'v_D', 'v_ND', 'v_DR'], rows), '']

    # PCB별, 불량 종류별, 결함을 보는가: K=1과 가장 큰 K
    kmax = int(per_pcb.k.max())
    for k in sorted({1, kmax}):
        rows_p, rows_t, rows_v = [], [], []
        for m, t in MAIN:
            x = per_pcb[(per_pcb.k == k) & (per_pcb.model == m) & (per_pcb.train == t)]
            if m == 'patchcore':
                x = per_pcb[(per_pcb.k == 1) & (per_pcb.model == m)]
            if x.empty:
                continue
            n = x.seed.nunique()
            pp = x.groupby('pcb').fair_fnr5.mean()
            rows_p.append([label(m, t), n] + [fmt(pp.get(p, np.nan)) for p in ['pcb1', 'pcb2', 'pcb3']])
            seed_mean = x.groupby('seed')[[f'fair_fnr_{ty}' for ty in TYPES] + ['v_D', 'v_ND', 'v_DR', 'v_NR']].mean()
            rows_t.append([label(m, t), n] + [fmt(seed_mean[f'fair_fnr_{ty}'].mean()) for ty in TYPES])
            rows_v.append([label(m, t), n] + [fmt(seed_mean[c].mean(), d=2) for c in ['v_D', 'v_ND', 'v_DR', 'v_NR']])
        L += [f'### K={k}: PCB별 공정@5', '', md_table(['모델, 학습', 'seed 수', 'PCB1', 'PCB2', 'PCB3'], rows_p), '',
              f'### K={k}: 불량 종류별 공정 미검률 @ 과검률 1% (fair_ad가 종류별로는 1%만 잰다. @5 평균보다 훨씬 높게 나온다)', '',
              md_table(['모델, 학습', 'seed 수'] + TYPES, rows_t), '',
              f'### K={k}: 결함을 보는가 (정상 검증 과검률 1% 임계값에서 불량 판정 비율)', '',
              'D·ND는 높을수록, DR·NR은 낮을수록 좋다. DR이 높으면 결함이 아니라 촬영 조건을 보고 있다는 뜻이다.', '',
              md_table(['모델, 학습', 'seed 수', 'D', 'ND', 'DR', 'NR'], rows_v), '']

    # 정상만을 넘는 첫 K
    f5 = stats(per_pcb, 'fair_fnr5')
    if ('patchcore', 'none', 1) in f5.index:
        pcv = f5.loc[('patchcore', 'none', 1), 'mean']
        L += ['### PatchCore(정상만)를 처음 넘는 K (공정@5 평균 기준)', '']
        for m, t in MAIN[1:]:
            if (m, t) not in f5.index.droplevel('k'):
                continue
            s = f5.loc[(m, t)]['mean']
            win = [k for k in KS if k in s.index and s[k] < pcv]
            L.append(f'- {label(m, t)}: ' + (f'K={win[0]}' if win else '넘지 못함') +
                     f' (계산된 K: {", ".join(map(str, s.index))})')
        L.append('')
    return L


def section_gradcam():
    f = RES / 'gradcam_K1_seed0.csv'
    if not f.exists():
        return ['## Grad-CAM (슬라이드 12, 14)', '', '결과 없음 (ad/gradcam.py --summarize 필요)', '']
    g = pd.read_csv(f)
    g = g[g['map'].isin(['gradcam', 'seg'])]   # LayerCAM은 빼고 Grad-CAM만
    rows = []
    for (m, t), x in g.groupby(['model', 'train'], sort=False):
        x = x.set_index('part')
        rows.append([label(m, t)] + [fmt(x.loc[p, 'mass_in'], d=2) for p in ['D', 'DR', 'ND']] +
                    [fmt(x.loc[p, 'top_in'], d=2) for p in ['D', 'ND']] + [fmt(x.loc['D', 'area'], d=3)])
    return ['## 모델이 결함을 보는가: Grad-CAM vs seg 결함 지도 (K=1 seed 0, 슬라이드 12·14)', '',
            'mass_in = 히트맵 질량 중 결함 영역(40px 넓힘) 안 비율. 지도가 사진 전체에 고르게 퍼지면 mass_in = 영역 면적. '
            'top_in = 가장 뜨거운 픽셀을 영역 크기만큼 골랐을 때 그중 영역 안 비율. 이미지 모델은 Grad-CAM(layer3), seg는 출력 지도.',
            '그림: `AnomalyDetection/results/gradcam/K1_seed0_all.jpg`', '',
            md_table(['모델, 학습', 'mass_in D', 'mass_in DR', 'mass_in ND', 'top_in D', 'top_in ND', '영역 면적'], rows), '']


def section_standard():
    L = ['## 원래 시험셋 (슬라이드 11, 14)', '']
    fs = sorted(RES.glob('local_ad_*_nogen_K1_seed*.csv'))
    if fs:
        d = pd.concat([pd.read_csv(f).assign(split=int(re.search(r'seed(\d+)', f.name).group(1))) for f in fs])
        d = d[(d.train == 'none') & d['agg'].isin(['image', 'top10'])]   # seg 점수는 fair_ad와 같은 top10
        s = d.groupby(['model', 'split', 'pcb'])[['fnr', 'loc_hit']].mean().groupby(['model', 'split']).mean()
        rows = []
        for m, x in s.groupby('model'):
            rows.append([NAME.get(m, m), x.index.get_level_values('split').nunique(), fmt(x.fnr.mean(), x.fnr.std(ddof=1)),
                         fmt(x.loc_hit.mean(), d=2) if x.loc_hit.notna().any() else '—'])
        L += ['### 실제 불량만, K=1, 같은 분할 seed에서 이미지 vs 위치 (`local_ad_{img,seg}_nogen_K1_seed*.csv`)', '',
              md_table(['모델', 'seed 수', '미검률 @ 과검률 1%', '위치 적중률'], rows), '']
    f = RES / 'results.csv'
    if f.exists():
        r = pd.read_csv(f)
        r = r[(r.regime == 'main') & (r.aug == 'none') & (r.res == 512)]
        g = r.groupby(['model', 'K', 'seed', 'category']).fnr_at_fpr1.mean().groupby(['model', 'K', 'seed']).mean()
        g = g.groupby(['model', 'K']).agg(['mean', 'std', 'size'])
        ks = sorted(g.index.get_level_values('K').unique())
        rows = [[m] + [fmt(*g.loc[(m, k)]) if (m, k) in g.index else '—' for k in ks]
                for m in g.index.get_level_values('model').unique()]
        L += ['### 표준 K 곡선, 실제만, 512 (`results.csv`, regime main)', '',
              md_table(['모델'] + [f'K={k}' for k in ks], rows), '']
    return L


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_pcb = load_fair()
    figures(per_pcb)
    done = per_pcb.groupby(['k', 'seed']).pcb.nunique()
    have = ', '.join(f'K{k}: seed {",".join(str(s) for s in sorted(x.index.get_level_values("seed")))}'
                     for k, x in done[done == 3].groupby(level='k'))
    L = ['# 발표용 표 (자동 생성)', '',
         f'`python -m AnomalyDetection.ad.ppt_summary`로 만든다. 직접 고치지 않는다. 생성 {pd.Timestamp.now():%Y-%m-%d %H:%M}.', '',
         f'포함된 공정 시험셋 결과: {have}', '',
         '그림: `fair_kcurve.png` (공정 vs 원래 시험셋 K 곡선), `fair_kcurve_pcb.png` (PCB별)', '']
    L += section_fair(per_pcb) + section_gradcam() + section_standard()
    (OUT / 'summary.md').write_text('\n'.join(L))
    print('저장:', OUT / 'summary.md', OUT / 'fair_kcurve.png', OUT / 'fair_kcurve_pcb.png')


if __name__ == '__main__':
    main()
