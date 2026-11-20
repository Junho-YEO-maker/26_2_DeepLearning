"""SMT판 생성 점수 (ad/gen_quality.py와 같은 점수, 11/19). VisA는 PCB·종류별 시험 결함이 4–10장이라 이득이 너무 흔들려서,
종류당 시험 결함이 약 40장인 SMT에서 다시 본다. SMT에는 SD가 ms(미삽) 자리에 부품을 다시 그리는 "라벨이 틀린" 사례도 있다.

세트: sd_lora_fix(SD+LoRA, 자리 고침), ctransgan(팀원) × K 1·4·16 × seed 0–2. 마스크 = 박스.
실제 기준 = 시험 불량(test.csv의 불량, 한 종류짜리만). 정상 은행 = train_normal 200장의 패치 20000개, T = val 정상 99% 분위.
이득 = seg_ad 표준 시험셋 종류별 미검률(과검 1%): 실제만 − 생성 40장.

    python -m AnomalyDetection.smt.gen_quality_smt            # 점수
    python -m AnomalyDetection.smt.gen_quality_smt --analyze  # 상관
결과: AnomalyDetection/results/smt/gen_quality_smt.csv, gen_quality_smt_images.csv, gen_quality_smt_gain.csv,
      AnomalyDetection/results/ppt/gen_quality_smt_summary.md
"""
import argparse
import csv
import json

import numpy as np
import torch
from PIL import Image

from ..ad.data import ROOT, Item, read_csv
from ..ad.features import PatchFeatures, device
from ..ad.gen_quality import outside_mask, pair_stats
from ..ad.gen_realism import nn_dist
from ..ad.patch_ad import GRID, RES, cell_labels, patch_feats
from .data import CATEGORIES, GENERATED, SmtData
from .seg_ad import defect_mask

RESULTS = ROOT / 'AnomalyDetection/results/smt'
SETS = [('sd_lora_fix', 'SD+LoRA'), ('ctransgan', 'cTransGAN')]
KS = [(k, s) for k in (1, 4, 16) for s in (0, 1, 2)]
SCORES = ['in_frac', 'in_anom', 'out_excess', 'out_change', 'div', 'nn_ratio', 'precision', 'recall']


def compute():
    torch.set_num_threads(4)
    D, dev = SmtData('none'), device()
    ex = PatchFeatures().to(dev)
    rows, img_rows = [], []
    test_rows = read_csv(D.splits / 'test.csv')
    for cat in CATEGORIES:
        types = D.types[cat]
        nl = D.train_normal(cat)
        rng = np.random.default_rng(0)
        nl_s = [nl[i] for i in rng.choice(len(nl), min(200, len(nl)), replace=False)]
        ref = np.mean([D.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl_s], 0)
        feats = lambda items: patch_feats(D, ex, dev, items, ref).astype(np.float32)        # noqa: E731
        Pn = feats(nl_s).reshape(-1, 1024)
        Pn = Pn[rng.choice(len(Pn), 20000, replace=False)]
        mu, sd = Pn.mean(0), Pn.std(0) + 1e-6
        z = lambda X: (X - mu) / sd                                                         # noqa: E731
        Bn = z(Pn)
        vl = D.val_normal(cat)
        vl = [vl[i] for i in rng.choice(len(vl), min(200, len(vl)), replace=False)]
        dv = nn_dist(z(feats(vl).reshape(-1, 1024)), Bn, dev)
        T, med = float(np.quantile(dv, 0.99)), float(np.median(dv))
        rr = [r for r in test_rows if r['category'] == cat and r['label'] == '1' and r['defect_type'] in types]
        real = [Item(r['image'], 1, r['defect_type'], 'real') for r in rr]
        RF = z(feats(real))
        rlab = [cell_labels(defect_mask(r, cat)).reshape(-1) for r in rr]
        ok = [i for i in range(len(real)) if (rlab[i] == 1).any()]
        rdesc = {i: RF[i][rlab[i] == 1].mean(0) for i in ok}
        r_in_t = {t: [float(np.mean(nn_dist(RF[i][rlab[i] == 1], Bn, dev) > T)) for i in ok if real[i].defect_type == t]
                  for t in types}
        print(f'{cat}: T={T:.2f}, 시험 불량 {len(real)}장, 실제 in_frac ' +
              ' '.join(f'{t}:{np.mean(v):.2f}' for t, v in r_in_t.items() if v), flush=True)
        base_cache = {}
        for aug, name in SETS:
            for k, s in KS:
                man = GENERATED / aug / f'K{k}_seed{s}' / 'manifest.csv'
                if not man.exists():
                    continue
                mrows = [r for r in read_csv(man) if r['category'] == cat and r['label'] == '1']
                gen = [Item(r['image'], 1, r['defect_type'], aug) for r in mrows]
                GF = z(feats(gen))
                need = sorted({r['base_normal_image'] for r in mrows} - set(base_cache))
                if need:
                    BF = z(feats([Item(p, 0, '', 'real') for p in need]))
                    base_cache.update({p: BF[i] for i, p in enumerate(need)})
                glab = []
                for r in mrows:
                    with Image.open(ROOT / r['mask']) as im:
                        glab.append(cell_labels(np.asarray(im.convert('L').resize((RES, RES), Image.NEAREST)) > 0).reshape(-1))
                Dg = nn_dist(GF.reshape(-1, 1024), Bn, dev).reshape(len(gen), -1)
                per = []
                for i, (it, r) in enumerate(zip(gen, mrows)):
                    inside = glab[i] == 1
                    outside = outside_mask(glab[i].reshape(GRID, GRID)).reshape(-1)
                    bfeat = base_cache[r['base_normal_image']]
                    if outside.sum() >= 5:     # 박스가 사진을 거의 다 덮으면 마스크 밖이 없다
                        db = nn_dist(bfeat[outside], Bn, dev)
                        chg = np.linalg.norm(GF[i][outside] - bfeat[outside], axis=1)
                        oe, oc = float(np.mean(Dg[i][outside] > T) - np.mean(db > T)), float(np.sort(chg)[-5:].mean() / med)
                    else:
                        oe = oc = np.nan
                    rec = dict(image=it.image, aug=aug, k=k, seed=s, cat=cat, type=it.defect_type,
                               in_frac=float(np.mean(Dg[i][inside] > T)) if inside.any() else np.nan,
                               in_anom=float(np.mean(Dg[i][inside] / T)) if inside.any() else np.nan,
                               out_excess=oe, out_change=oc)
                    per.append(rec)
                    img_rows.append(rec)
                for t in types:
                    idx = [i for i, it in enumerate(gen) if it.defect_type == t and (glab[i] == 1).any()]
                    ridx = [i for i in ok if real[i].defect_type == t]
                    if len(idx) < 2 or len(ridx) < 2:
                        continue
                    gpair, gnn = pair_stats(np.stack([GF[i][glab[i] == 1].mean(0) for i in idx]))
                    rpair, rnn = pair_stats(np.stack([rdesc[i] for i in ridx]))
                    G = np.concatenate([GF[i][glab[i] == 1] for i in idx])
                    R = np.concatenate([RF[i][rlab[i] == 1] for i in ridx])
                    dn_G, dn_R = nn_dist(G, Bn, dev), nn_dist(R, Bn, dev)
                    rows.append(dict(aug=aug, k=k, seed=s, cat=cat, type=t, n=len(idx),
                                     **{m: float(np.nanmean([per[i][m] for i in idx]))
                                        for m in ('in_frac', 'in_anom', 'out_excess', 'out_change')},
                                     div=gpair / rpair, nn_ratio=gnn / rnn,
                                     precision=float(np.mean(nn_dist(G, R, dev) < dn_G)),
                                     recall=float(np.mean(nn_dist(R, G, dev) < dn_R)),
                                     real_in_frac=float(np.mean(r_in_t[t])) if r_in_t[t] else np.nan, T=T))
                print(f'  {cat} {name} K{k} seed{s}: ' + ' '.join(
                    f"{r['type']}:{r['in_frac']:.2f}" for r in rows if (r['aug'], r['k'], r['seed'], r['cat']) == (aug, k, s, cat)),
                    flush=True)
    for path, rs in ((RESULTS / 'gen_quality_smt.csv', rows), (RESULTS / 'gen_quality_smt_images.csv', img_rows)):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        print('저장:', path)


def gains():
    import pandas as pd
    out = []
    for cat in CATEGORIES:
        d = pd.read_csv(RESULTS / f'seg_ad_{cat}.csv')
        for aug, _ in SETS:
            for k, s in KS:
                b = d[(d.K == k) & (d.seed == s) & (d.cond == 'none')]
                g = d[(d.K == k) & (d.seed == s) & (d.cond == f'{aug}:40:0.5')]
                if not len(b) or not len(g):
                    continue
                pb, pg = json.loads(b.iloc[-1].per_type), json.loads(g.iloc[-1].per_type)
                for t in pb:
                    if '|' not in t and t in pg:
                        out.append(dict(aug=aug, k=k, seed=s, cat=cat, type=t, base_fnr=pb[t], gen_fnr=pg[t],
                                        gain=pb[t] - pg[t]))
    return pd.DataFrame(out)


def analyze():
    import pandas as pd
    from scipy.stats import spearmanr
    q = pd.read_csv(RESULTS / 'gen_quality_smt.csv')
    m = q.merge(gains(), on=['aug', 'k', 'seed', 'cat', 'type'])
    m.to_csv(RESULTS / 'gen_quality_smt_gain.csv', index=False)
    cols = SCORES + ['base_fnr']
    lines = ['# SMT: 어떤 생성이 도움이 되나 (gen_quality_smt.py)', '',
             '이득 = 위치 seg 표준 시험셋 종류별 미검률(실제만 − 생성 40장, 과검 1%). 양수면 생성이 도움.', '']

    def corr(d, title, within=False):
        x = d.copy()
        if within:
            for c in cols + ['gain']:
                x[c] = x.groupby(['cat', 'type'])[c].rank()
        out = [f'## {title} (칸 {len(x)}개)', '', '| 점수 | 순위상관 ρ | p |', '|---|---|---|']
        for c in cols:
            y = x[[c, 'gain']].dropna()
            rho, p = spearmanr(y[c], y.gain)
            out.append(f'| {c} | {rho:+.2f} | {p:.3f} |')
        return out + ['']

    cell = m.groupby(['aug', 'cat', 'type', 'k'])[cols + ['gain']].mean().reset_index()
    lines += corr(cell, '생성기 × 공정 × 종류 × K, seed 3개 평균')
    lines += corr(m, '종류 안에서만 비교 (종류별 순위, 모든 seed)', within=True)
    w = m.groupby(['aug', 'cat', 'type', 'k']).gain.std().mean()
    b = cell.gain.std()
    lines += [f'잡음: 같은 칸 seed 사이 이득 표준편차 평균 {w:.3f}, 칸 평균끼리 표준편차 {b:.3f}', '']
    t = m[m.k == 1].groupby(['aug', 'cat', 'type'])[['in_frac', 'real_in_frac', 'out_change', 'div', 'precision', 'gain']].mean()
    lines += ['## K=1 생성기 × 종류 (seed 평균)', '', '| 생성기 | 공정 | 종류 | in_frac | 실제 in_frac | out_change | div | precision | 이득 |',
              '|---|---|---|---|---|---|---|---|---|']
    for (aug, cat, typ), r in t.iterrows():
        lines.append(f'| {aug} | {cat} | {typ} | {r.in_frac:.2f} | {r.real_in_frac:.2f} | {r.out_change:.2f} | {r["div"]:.2f} | '
                     f'{r.precision:.2f} | {r.gain:+.3f} |')
    lines.append('')
    (ROOT / 'AnomalyDetection/results/ppt/gen_quality_smt_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    analyze() if a.analyze else compute()
