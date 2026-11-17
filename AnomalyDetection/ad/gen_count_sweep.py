"""생성 장수 sweep: 실제 K장 + 생성 n장(n = 0…160)으로 DevNet·MLP를 학습해 FNR@FPR1%가 n에 따라 어떻게 변하는지 본다.

"생성을 갑자기 많이 넣어서 나빠지는가?"를 확인한다. 임베딩은 photonorm(gen_select와 같음).
n장은 종류를 번갈아(round-robin) 고른 무작위 순서의 앞 n장이라 n이 커져도 앞 샘플은 그대로다(nested).
헤드 시드마다 순서를 새로 섞으므로, 시드 평균은 샘플 선택과 헤드 학습의 우연을 같이 평균한다.

    python -m AnomalyDetection.ad.gen_count_sweep --aug sd_lora_v3 --k 1 --seed 0 [--data-root ...]
결과: AnomalyDetection/results/gen_count_sweep_{aug}_K{k}_seed{s}.csv, Augmentation/figures/gen_count_sweep_{aug}_K{k}_seed{s}.png
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, TYPES, Data
from .eval import evaluate
from .features import PatchFeatures, device
from .gen_select import photonorm_embed
from .models import EmbeddingHead

NS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160]
MODELS = ('devnet', 'mlp')
COLOR = {'devnet': '#2a78d6', 'mlp': '#eb6834'}     # dataviz 기본 팔레트 1·2번(검증된 쌍)
NAME = {'devnet': 'DevNet', 'mlp': 'MLP'}


def order(types, rng):
    """종류별로 섞은 뒤 종류 순서도 섞어 번갈아 이어 붙인다 → 앞 n장이 종류를 고르게 덮는다."""
    per = {t: list(rng.permutation(np.where(types == t)[0])) for t in TYPES}
    seq, ts = [], list(rng.permutation(TYPES))
    while any(per.values()):
        for t in ts:
            if per[t]:
                seq.append(per[t].pop(0))
    return np.array(seq)


def plot(rows, aug, k, seed, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    pcbs = sorted({r['pcb'] for r in rows})
    fig, axes = plt.subplots(1, 1 + len(pcbs), figsize=(4.2 * (1 + len(pcbs)), 3.6), sharey=True, facecolor=surface)
    for ax, sel in zip(axes, ['mean'] + pcbs):
        ax.set_facecolor(surface)
        for m in MODELS:
            ys = []
            for n in NS:
                v = [r['fnr_at_fpr1'] for r in rows if r['model'] == m and r['n_gen'] == n and (sel == 'mean' or r['pcb'] == sel)]
                ys.append(np.mean(v))
            xs = [max(n, 0.5) for n in NS]                         # n=0(실제만)은 로그축 왼쪽 끝에
            ax.axhline(ys[0], color=COLOR[m], lw=1, ls=(0, (3, 3)), alpha=0.8)
            ax.plot(xs, ys, color=COLOR[m], lw=2, marker='o', ms=4, label=NAME[m])
            ax.annotate(f'{NAME[m]} {ys[-1]:.2f}', (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=8, color=ink)
        ax.set_xscale('log', base=2)
        ax.set_xticks([0.5, 1, 4, 16, 64, 160])
        ax.set_xticklabels(['0', '1', '4', '16', '64', '160'], color=muted, fontsize=8)
        ax.tick_params(axis='y', colors=muted, labelsize=8)
        ax.grid(axis='y', color=grid, lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(grid)
        ax.set_title('Mean of PCB1-3' if sel == 'mean' else sel.upper(), color=ink, fontsize=10, loc='left')
        ax.set_xlabel(f'# generated per PCB (added to {4 * k} real)', color=muted, fontsize=8)
    axes[0].set_ylabel('FNR @ FPR 1% (lower is better)', color=muted, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle(f'{aug}: FNR vs number of generated samples (K={k}, seed {seed}, photonorm, mean of head seeds)'
                 f'  -  dashed = real only (n=0)', color=ink, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    rows = []
    for pcb in a.pcbs:
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        ref = np.mean([data.load(x.image, a.res).astype(np.float32).mean((0, 1)) for x in nl], 0)
        Xt, Xv, Xn, Xr, Xg = (photonorm_embed(data, ex, dev, x, ref, a.res) for x in (test, val, nl, real, gen))
        types = np.array([g.defect_type for g in gen])
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]
        for hs in range(a.head_seeds):
            seq = order(types, np.random.default_rng(1000 + hs))
            for n in NS:
                if n > len(seq):
                    continue
                Xa = np.concatenate([Xr, Xg[seq[:n]]])
                for m in MODELS:
                    h = EmbeddingHead(m, seed=hs).fit(Xn, Xa)
                    r = evaluate(h.score(Xt), labels, dtypes, h.score(Xv))
                    rows.append(dict(pcb=pcb, aug=a.aug, n_gen=n, model=m, head_seed=hs,
                                     fnr_at_fpr1=r['fnr_at_fpr1'], auroc=r['auroc']))
        print(f'{pcb}: ' + '  '.join(
            f"n={n}:" + '/'.join(f"{np.mean([r['fnr_at_fpr1'] for r in rows if r['pcb'] == pcb and r['n_gen'] == n and r['model'] == m]):.2f}"
                                 for m in MODELS) for n in NS), flush=True)

    tag = f'{a.aug}_K{a.k}_seed{a.seed}'
    out = ROOT / 'AnomalyDetection/results' / f'gen_count_sweep_{tag}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nFNR@FPR1% PCB {len(a.pcbs)}개·헤드 시드 {a.head_seeds}개 평균 (DevNet / MLP)')
    for n in NS:
        d, m = (np.mean([r['fnr_at_fpr1'] for r in rows if r['n_gen'] == n and r['model'] == x]) for x in MODELS)
        print(f'  n={n:3d}  DevNet {d:.3f}  MLP {m:.3f}')
    fig = ROOT / 'Augmentation/figures' / f'gen_count_sweep_{tag}.png'
    plot(rows, a.aug, a.k, a.seed, fig)
    print(f'저장: {out}\n그림: {fig}')


if __name__ == '__main__':
    main()
