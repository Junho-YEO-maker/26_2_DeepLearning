"""생성기 × K × 합성 장수 N 반복 결과를 발표용 그림·표로 모은다 (11/18, ppt_preview.md 0번·0-2번).

입력
  VisA (AnomalyDetection/results/)
    fair_ad_K{k}_seed{s}_grid_{v3|ctg}.csv   위치 seg(촬영 조건 무작위화) + 합성 N장 (train = +{aug}_n{N}, N=40은 +{aug})
    fair_ad_K{k}_seed{s}.csv                 N=0(실제만) 줄, PatchCore(정상만) 기준선
  SMT (AnomalyDetection/results/smt/)
    seg_ad_{pre,solder}.csv                  cond = none | sd_lora_fix:{N}:0.5 | sd_lora_fix_msreal:{N}:0.5 (ms 따로 실험)
평균 순서: 헤드 시드 → PCB → 분할 seed. 칸마다 seed 수를 같이 적는다.
출력 (AnomalyDetection/results/ppt/)
  grid_visa.png   생성기마다 x = N, 선 = K, y = 공정 시험셋 미검률 @ 과검 5%
  grid_smt.png    공정마다 x = N, 선 = K, y = 미검률 @ 과검 1%
  grid_summary.md 위 숫자 표, VisA bent 미검률, SMT 유형 × N 표(K=1), ms 따로 실험 표

    python -m AnomalyDetection.ad.grid_summary
"""
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import ROOT  # noqa: E402

RES = ROOT / 'AnomalyDetection/results'
OUT = RES / 'ppt'
NS = [0, 1, 5, 10, 20, 40]
GENS = {'v3': ('sd_lora_v3', 'SD+LoRA'), 'ctg': ('ctransgan_diverse', 'cTransGAN')}
# 순서형 램프 (dataviz 참고 팔레트 blue 250→700, 밝은 면에서 순서형 하한 250). 선마다 마커·끝 라벨을 같이 둔다.
RAMP = {4: ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'], 3: ['#86b6ef', '#2a78d6', '#104281']}
MARK = ['o', 's', '^', 'D']
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'


def n_of(train, aug):
    if train == 'none':
        return 0
    if train == f'+{aug}':
        return 40
    m = re.fullmatch(rf'\+{aug}_n(\d+)', train)
    return int(m.group(1)) if m else None


def visa_cells(metric='fair_fnr5'):
    rows = []
    for f in sorted(RES.glob('fair_ad_K*_seed*_grid_*.csv')):
        k, s, g = re.fullmatch(r'fair_ad_K(\d+)_seed(\d+)_grid_(v3|ctg)\.csv', f.name).groups()
        d = pd.read_csv(f)
        d = d[d.model == 'seg_aug']
        d['N'] = [n_of(t, GENS[g][0]) for t in d.train]
        d = d.dropna(subset=['N'])
        for n, x in d.groupby('N'):
            rows.append(dict(gen=g, K=int(k), seed=int(s), N=int(n), value=x.groupby('pcb')[metric].mean().mean()))
    base = []
    for f in sorted(RES.glob('fair_ad_K*_seed*.csv')):
        m = re.fullmatch(r'fair_ad_K(\d+)_seed(\d+)\.csv', f.name)
        if not m:
            continue
        d = pd.read_csv(f)
        x = d[(d.model == 'seg_aug') & (d.train == 'none')]
        if len(x):
            base.append(dict(K=int(m.group(1)), seed=int(m.group(2)), N=0, value=x.groupby('pcb')[metric].mean().mean()))
    cells = pd.DataFrame(rows)
    base = pd.DataFrame(base)
    out = []
    for g in GENS:
        have = cells[cells.gen == g] if len(cells) else cells
        for k in sorted(set(have.K) if len(have) else []):
            seeds = set(have[have.K == k].seed)
            b = base[(base.K == k) & base.seed.isin(seeds)] if len(base) else base
            out += [dict(gen=g, K=k, seed=r.seed, N=0, value=r.value) for r in b.itertuples()]
    allc = pd.concat([cells, pd.DataFrame(out)]) if len(cells) else pd.DataFrame(out)
    return allc


def patchcore_line():
    f = RES / 'fair_ad_K1_seed0.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f)
    x = d[d.model == 'patchcore']
    return float(x.groupby('pcb').fair_fnr5.mean().mean()) if len(x) else None


def smt_cells(cond_prefix='sd_lora_fix'):
    rows = []
    for cat in ('pre', 'solder'):
        f = RES / 'smt' / f'seg_ad_{cat}.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for r in d.itertuples():
            if r.cond == 'none':
                n = 0
            else:
                m = re.fullmatch(rf'{cond_prefix}:(\d+):0\.5', r.cond)
                if not m:
                    continue
                n = int(m.group(1))
            rows.append(dict(cat=cat, K=int(r.K), seed=int(r.seed), N=n, value=r.fnr, per_type=r.per_type))
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    # N=0(실제만)은 그 (공정, K)에서 합성 칸이 있는 seed만 쓴다. 실제만 실험(run_smt_equiv.sh)이 seed 3–4를 더해
    # 그대로 평균하면 N=0과 N>0이 서로 다른 분할의 평균이 된다. 합성 칸이 없는 K(2 · 8 · 32)는 실제만 seed를 다 쓴다.
    syn = d[d.N > 0].groupby(['cat', 'K']).seed.agg(set)
    keep = [r.N > 0 or (r.cat, r.K) not in syn.index or r.seed in syn[(r.cat, r.K)] for r in d.itertuples()]
    return d[keep].reset_index(drop=True)


def summarize(c, keys):
    """seed 평균·개수. N=0은 그 (K)에서 합성 칸이 있는 seed만."""
    if not len(c):
        return c
    return c.groupby(keys + ['N']).value.agg(['mean', 'count']).reset_index()


def panel(ax, s, title, ks, ylabel, hline=None):
    ax.set_facecolor(SURF)
    xs = {n: i for i, n in enumerate(NS)}
    cols = RAMP[4] if len(ks) > 3 else RAMP[3]
    ends = []
    for i, k in enumerate(ks if len(s) else []):
        x = s[s.K == k].sort_values('N')
        if not len(x):
            continue
        px = [xs[n] for n in x.N]
        ax.plot(px, x['mean'], color=cols[i], lw=2, marker=MARK[i], ms=7, mec=SURF, mew=1.5, label=f'K={k}', zorder=3)
        ends.append((x['mean'].iloc[-1], px[-1], k))
    # 끝 이름표가 겹치지 않게: 아래부터 쌓으며 세로 범위의 5%씩 띄운다
    lo, hi = ax.get_ylim()
    ends.sort()
    ys = [e[0] for e in ends]
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + (hi - lo) * 0.05)
    for (_, px_end, k), y in zip(ends, ys):
        ax.annotate(f'K={k}', (px_end, y), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK2)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1.2, ls='--', zorder=1)
        ax.annotate('PatchCore (normal only)', (len(NS) - 1, hline), xytext=(0, 4), textcoords='offset points',
                    ha='right', va='bottom', fontsize=8, color=MUTED)
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel('synthetic defects per type (N)', color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlim(-0.3, len(NS) - 0.4)


def table(s, keys, fmt='{:.3f}'):
    if not len(s):
        return ['(결과 없음)', '']
    lines = []
    for key, x in s.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        vals = {int(r.N): fmt.format(r['mean']) + ' (' + str(int(r['count'])) + ')' for _, r in x.iterrows()}
        lines.append('| ' + ' | '.join(map(str, key)) + ' | ' + ' | '.join(vals.get(n, '') for n in NS) + ' |')
    head = '| ' + ' | '.join(keys) + ' | ' + ' | '.join(f'N={n}' for n in NS) + ' |'
    return [head, '|' + '---|' * (len(keys) + len(NS))] + lines + ['']


def per_type_table(c, cat, k=1):
    x = c[(c.cat == cat) & (c.K == k)]
    if not len(x):
        return ['(결과 없음)', '']
    rows = []
    for r in x.itertuples():
        for t, v in json.loads(r.per_type).items():
            if '|' not in t:                     # 여러 유형이 섞인 사진은 몇 장뿐이라 뺀다
                rows.append((t, r.N, v))
    t = pd.DataFrame(rows, columns=['type', 'N', 'fnr']).groupby(['type', 'N']).fnr.mean().unstack()
    head = '| 유형 | ' + ' | '.join(f'N={n}' for n in NS if n in t.columns) + ' |'
    out = [head, '|' + '---|' * (1 + sum(n in t.columns for n in NS))]
    for typ, r in t.iterrows():
        out.append(f'| {typ} | ' + ' | '.join(f'{r[n]:.2f}' for n in NS if n in t.columns) + ' |')
    return out + ['']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ['# 생성기 × K × 합성 장수 N (grid_summary.py가 만든 표)', '',
          '값 = 평균 (seed 수). 헤드 시드 → PCB → 분할 seed 순으로 평균. N=0은 실제만.', '']
    # VisA
    v = visa_cells()
    vb = visa_cells('fair_fnr_bent')
    pc = patchcore_line()
    sv = summarize(v, ['gen', 'K'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURF)
    for ax, g in zip(axes, GENS):
        panel(ax, sv[sv.gen == g] if len(sv) else sv, f'VisA {GENS[g][1]}', [1, 2, 4, 8],
              'miss rate @ 5% false alarm (fair test)', pc)
    fig.tight_layout()
    fig.savefig(OUT / 'grid_visa.png', dpi=200, facecolor=SURF)
    plt.close(fig)
    md += ['## VisA: 위치 seg(촬영 조건 무작위화), 공정 시험셋 미검률 @ 과검 5%', '',
           f'PatchCore(정상만) 기준선: {pc:.3f}' if pc is not None else 'PatchCore 기준선 없음', '']
    md += table(sv.assign(gen=sv.gen.map(lambda g: GENS[g][1])) if len(sv) else sv, ['gen', 'K'])
    md += ['### VisA bent 미검률 (잔상 제거 효과를 볼 곳)', '']
    sb = summarize(vb, ['gen', 'K'])
    md += table(sb.assign(gen=sb.gen.map(lambda g: GENS[g][1])) if len(sb) else sb, ['gen', 'K'])
    # SMT
    # 행 = 생성기 (SD+LoRA sd_lora_fix, cTransGAN = 팀원 smt_ctransgan, cond ctransgan:N:0.5), 열 = 공정. 열마다 같은 y축
    smt_gens = [('sd_lora_fix', 'SD+LoRA'), ('ctransgan', 'cTransGAN')]
    cs = {p: smt_cells(p) for p, _ in smt_gens}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey='col', facecolor=SURF)
    for row, (p, name) in zip(axes, smt_gens):
        s = summarize(cs[p], ['cat', 'K'])
        for ax, cat in zip(row, ('pre', 'solder')):
            panel(ax, s[s.cat == cat] if len(s) else s, f'SMT {cat}, {name}', [1, 4, 16], 'miss rate @ 1% false alarm')
    fig.tight_layout()
    fig.savefig(OUT / 'grid_smt.png', dpi=200, facecolor=SURF)
    plt.close(fig)
    c = cs['sd_lora_fix']
    for p, name in smt_gens:
        md += [f'## SMT: 위치 seg, {name} (같은 자리 규칙), 미검률 @ 과검 1%', '']
        md += table(summarize(cs[p], ['cat', 'K']), ['cat', 'K'])
        for cat in ('pre', 'solder'):
            md += [f'### SMT {cat} 유형별 미검률, {name}, K=1 (seed 평균)', ''] + per_type_table(cs[p], cat)
    # ms 따로 실험
    m = smt_cells('sd_lora_fix_msreal')
    m = m[(m.cat == 'solder') & (m.N > 0)] if len(m) else m
    md += ['## SMT 납땜 ms(미삽) 따로 실험: ms만 실제 빈 패드 기반으로 바꾼 세트', '',
           '| K | N | ms 미검률: SD (sd_lora_fix) | ms 미검률: 실제 빈 패드 (msreal) | 전체 미검률: SD | 전체: msreal | seed |', '|---|---|---|---|---|---|---|']
    if len(m):
        for (k, n), x in m.groupby(['K', 'N']):
            y = c[(c.cat == 'solder') & (c.K == k) & (c.N == n) & c.seed.isin(x.seed)]
            ms = lambda d: np.mean([json.loads(p).get('ms', np.nan) for p in d.per_type]) if len(d) else np.nan  # noqa: E731
            md.append(f'| {k} | {n} | {ms(y):.2f} | {ms(x):.2f} | {y.value.mean():.3f} | {x.value.mean():.3f} | {len(x)} |')
    md.append('')
    (OUT / 'grid_summary.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))
    print('저장:', OUT / 'grid_visa.png', OUT / 'grid_smt.png', OUT / 'grid_summary.md')


if __name__ == '__main__':
    main()
