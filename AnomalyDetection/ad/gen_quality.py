"""어떤 생성이 AD에 도움이 되나 (11/19): 생성 세트마다 점수를 재고, 이미 잰 유형별 이득과 맞춰 본다.

AD를 다시 학습하지 않는다. AD와 같은 패치 특징(WRN50, 32×32 격자, photonorm, 정상 패치 평균·표준편차로 표준화)에서 잰다.
정상 은행 = train_normal[::5]의 패치 20000개, T = val 정상 패치의 은행 최근접 거리 99% 분위(정상 패치의 1%만 T를 넘는다).

점수 (PCB·종류별 평균. 장마다 값은 gen_quality_images.csv)
  ① 라벨이 맞나 (마스크 안에 결함이 있나)
     in_frac     마스크 안 패치 중 정상 은행에서 T보다 먼 비율. 낮으면 "결함을 안 그렸다"(정상처럼 보이는데 불량 라벨)
     in_anom     마스크 안 패치의 (최근접 거리 / T) 평균
  ② 흔적이 없나 (마스크 밖이 정상인가)
     out_excess  마스크 밖(마스크 주변 2칸 넓힌 바깥) 패치 중 T보다 먼 비율 − 같은 바탕 정상 사진의 그 비율
     out_change  마스크 밖 패치의 바탕 대비 특징 변화 상위 5개 평균 / 정상 최근접 거리 중앙값
  ③ 다양한가
     div         같은 종류 생성끼리 결함 서술자(마스크 안 패치 평균) 평균 쌍거리 / 실제 pool 결함끼리의 평균 쌍거리
     nn_ratio    생성끼리 최근접 거리 중앙값 / 실제끼리 최근접 거리 중앙값 (작으면 거의 같은 장이 많다)
  ④ 실제를 닮았나·덮나 (gen_realism과 같은 정의, 시험 결함 = pool에서 K장을 뺀 나머지를 쓰므로 현장에서는 못 쓰는 참고값)
     precision, recall
  참고: real_in_frac = 실제 pool 결함의 in_frac

    python -m AnomalyDetection.ad.gen_quality --data-root $DR              # 점수 계산 (VisA 격자 세트 16개)
    python -m AnomalyDetection.ad.gen_quality --analyze                    # 유형별 이득과 상관
결과: AnomalyDetection/results/gen_quality.csv, gen_quality_images.csv, gen_quality_gain.csv, ppt/gen_quality_summary.md
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F

from .data import PCBS, ROOT, TYPES, Data, Item, read_csv
from .features import PatchFeatures, device
from .gen_realism import nn_dist
from .patch_ad import GRID, RES, cell_labels, load_mask, patch_feats

RESULTS = ROOT / 'AnomalyDetection/results'
SETS = [('sd_lora_v3', 'v3'), ('ctransgan_diverse', 'ctg')]
KS = [(1, s) for s in range(5)] + [(2, 0), (4, 0), (8, 0)]
SCORES = ['in_frac', 'in_anom', 'out_excess', 'out_change', 'div', 'nn_ratio', 'precision', 'recall']


def outside_mask(lab, grow=2):
    """lab(1 양성, -1 경계)을 grow칸 넓힌 것의 바깥."""
    t = torch.from_numpy((lab != 0).astype(np.float32))[None, None]
    near = F.max_pool2d(t, 2 * grow + 1, 1, grow)[0, 0] > 0
    return ~near.numpy()


def pair_stats(D):
    """서술자들의 평균 쌍거리, 최근접 거리 중앙값."""
    if len(D) < 2:
        return np.nan, np.nan
    d = torch.cdist(torch.as_tensor(D), torch.as_tensor(D)).numpy()
    iu = np.triu_indices(len(D), 1)
    np.fill_diagonal(d, np.inf)
    return float(d[iu].mean()), float(np.median(d.min(1)))


def compute(a):
    torch.set_num_threads(4)
    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    pool = read_csv(data.splits / 'pool.csv')
    rows, img_rows = [], []
    for pcb in PCBS:
        nl = data.train_normal(pcb)
        ref = np.mean([data.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl], 0)
        feats = lambda items: patch_feats(data, ex, dev, items, ref).astype(np.float32)   # noqa: E731
        Pn = feats(nl[::5]).reshape(-1, 1024)
        rng = np.random.default_rng(0)
        Pn = Pn[rng.choice(len(Pn), 20000, replace=False)]
        mu, sd = Pn.mean(0), Pn.std(0) + 1e-6
        z = lambda X: (X - mu) / sd                                                      # noqa: E731
        Bn = z(Pn)
        dv = nn_dist(z(feats(data.val_normal(pcb)).reshape(-1, 1024)), Bn, dev)
        T, med = float(np.quantile(dv, 0.99)), float(np.median(dv))
        # 실제 pool 결함: 패치, 라벨, 서술자
        real = [Item(r['image'], 1, r['defect_type'], 'real') for r in pool if r['category'] == pcb]
        RF = z(feats(real))                                                              # (n, 1024, 1024)
        rlab = [cell_labels(load_mask(data, it, {})).reshape(-1) for it in real]
        rdesc = np.stack([RF[i][rlab[i] == 1].mean(0) if (rlab[i] == 1).any() else np.zeros(1024, np.float32)
                          for i in range(len(real))])
        r_in = [float(np.mean(nn_dist(RF[i][rlab[i] == 1], Bn, dev) > T)) for i in range(len(real)) if (rlab[i] == 1).any()]
        r_in_t = {t: [] for t in TYPES}
        for i, it in enumerate(real):
            if it.defect_type in TYPES and (rlab[i] == 1).any():
                r_in_t[it.defect_type].append(float(np.mean(nn_dist(RF[i][rlab[i] == 1], Bn, dev) > T)))
        print(f'{pcb}: T={T:.2f}, 실제 pool {len(real)}장, 실제 in_frac 평균 {np.mean(r_in):.2f}', flush=True)
        base_cache = {}
        for aug, tag in SETS:
            for k, s in KS:
                man = data.generated / aug / f'K{k}_seed{s}' / 'manifest.csv'
                if not man.exists():
                    continue
                mrows = [r for r in read_csv(man) if r['category'] == pcb]
                gen_masks = {r['image']: r['mask'] for r in mrows}
                gen = [Item(r['image'], 1, r['defect_type'], aug) for r in mrows]
                kimgs = {x.image for x in data.train_defects(pcb, k, s)}
                GF = z(feats(gen))
                need = sorted({r['base_normal_image'] for r in mrows} - set(base_cache))
                if need:
                    BF = z(feats([Item(p, 0, '', 'real') for p in need]))
                    base_cache.update({p: BF[i] for i, p in enumerate(need)})
                glab = [cell_labels(load_mask(data, it, gen_masks)).reshape(-1) for it in gen]
                Dg = nn_dist(GF.reshape(-1, 1024), Bn, dev).reshape(len(gen), -1)
                per = []
                for i, (it, r) in enumerate(zip(gen, mrows)):
                    lab = glab[i]
                    inside = lab == 1
                    outside = outside_mask(lab.reshape(GRID, GRID)).reshape(-1)
                    bfeat = base_cache[r['base_normal_image']]
                    db = nn_dist(bfeat[outside], Bn, dev)
                    chg = np.linalg.norm(GF[i][outside] - bfeat[outside], axis=1)
                    rec = dict(image=it.image, aug=aug, k=k, seed=s, pcb=pcb, type=it.defect_type,
                               in_frac=float(np.mean(Dg[i][inside] > T)) if inside.any() else np.nan,
                               in_anom=float(np.mean(Dg[i][inside] / T)) if inside.any() else np.nan,
                               out_excess=float(np.mean(Dg[i][outside] > T) - np.mean(db > T)),
                               out_change=float(np.sort(chg)[-5:].mean() / med))
                    per.append(rec)
                    img_rows.append(rec)
                for t in TYPES:
                    idx = [i for i, it in enumerate(gen) if it.defect_type == t]
                    ridx = [i for i, it in enumerate(real) if it.defect_type == t and it.image not in kimgs]
                    if not idx or not ridx:
                        continue
                    gdesc = np.stack([GF[i][glab[i] == 1].mean(0) for i in idx if (glab[i] == 1).any()])
                    gpair, gnn = pair_stats(gdesc)
                    rpair, rnn = pair_stats(rdesc[ridx])
                    G = np.concatenate([GF[i][glab[i] == 1] for i in idx])
                    R = np.concatenate([RF[i][rlab[i] == 1] for i in ridx])
                    dn_G, dn_R = nn_dist(G, Bn, dev), nn_dist(R, Bn, dev)
                    row = dict(aug=aug, k=k, seed=s, pcb=pcb, type=t, n=len(idx),
                               **{m: float(np.nanmean([per[i][m] for i in idx])) for m in ('in_frac', 'in_anom', 'out_excess', 'out_change')},
                               div=gpair / rpair, nn_ratio=gnn / rnn,
                               precision=float(np.mean(nn_dist(G, R, dev) < dn_G)),
                               recall=float(np.mean(nn_dist(R, G, dev) < dn_R)),
                               real_in_frac=float(np.mean(r_in_t[t])) if r_in_t[t] else np.nan, T=T)
                    rows.append(row)
                print(f'  {pcb} {tag} K{k} seed{s}: ' + ' '.join(
                    f"{t}:in{np.mean([r['in_frac'] for r in rows[-4:] if r['type'] == t]):.2f}" for t in TYPES), flush=True)
    for path, rs in ((RESULTS / 'gen_quality.csv', rows), (RESULTS / 'gen_quality_images.csv', img_rows)):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        print('저장:', path)


def gains():
    """seg + 촬영 조건 무작위화, 공정 시험셋, 종류별 미검률: 실제만(N=0) − 생성 40장. (aug, k, seed, pcb, type)마다."""
    import pandas as pd
    out = []
    for aug, tag in SETS:
        for k, s in KS:
            gf = RESULTS / f'fair_ad_K{k}_seed{s}_grid_{tag}.csv'
            bf = RESULTS / f'fair_ad_K{k}_seed{s}.csv'
            if not gf.exists() or not bf.exists():
                continue
            g, b = pd.read_csv(gf), pd.read_csv(bf)
            g = g[(g.model == 'seg_aug') & (g.train == '+' + aug)]
            b = b[(b.model == 'seg_aug') & (b.train == 'none')]
            for pcb in PCBS:
                for t in TYPES:
                    c = f'fair_fnr_{t}'
                    bv, gv = b[b.pcb == pcb][c].mean(), g[g.pcb == pcb][c].mean()
                    out.append(dict(aug=aug, k=k, seed=s, pcb=pcb, type=t, base_fnr=bv, gen_fnr=gv, gain=bv - gv))
    return pd.DataFrame(out)


def analyze():
    import pandas as pd
    from scipy.stats import spearmanr
    q = pd.read_csv(RESULTS / 'gen_quality.csv')
    m = q.merge(gains(), on=['aug', 'k', 'seed', 'pcb', 'type'])
    m.to_csv(RESULTS / 'gen_quality_gain.csv', index=False)
    cols = SCORES + ['base_fnr']
    lines = ['# 어떤 생성이 도움이 되나: 점수와 유형별 이득의 상관 (gen_quality.py)', '',
             '이득 = seg + 촬영 조건 무작위화, 공정 시험셋 종류별 미검률(실제만 − 생성 40장). 양수면 생성이 도움.',
             'base_fnr = 실제만의 종류별 미검률 (나아질 여지, 비교용).', '']

    def corr_table(d, title):
        out = [f'## {title} (칸 {len(d)}개)', '', '| 점수 | 순위상관 ρ | p |', '|---|---|---|']
        for c in cols:
            x = d[[c, 'gain']].dropna()
            rho, p = spearmanr(x[c], x.gain)
            out.append(f'| {c} | {rho:+.2f} | {p:.3f} |')
        return out + ['']

    k1 = m[m.k == 1].groupby(['aug', 'pcb', 'type'])[cols + ['gain']].mean().reset_index()
    lines += corr_table(k1, 'K=1, 분할 5개 평균 (생성기 2 × PCB 3 × 종류 4)')
    lines += corr_table(m, '모든 세트 (K 1·2·4·8, 칸마다 분할 1개)')
    t = m[m.k == 1].groupby(['aug', 'type'])[SCORES + ['gain']].mean().round(3)
    lines += ['## K=1 생성기 × 종류 평균', '', '| 생성기 | 종류 | ' + ' | '.join(SCORES + ['gain']) + ' |',
              '|' + '---|' * (2 + len(SCORES) + 1)]
    for (aug, typ), r in t.iterrows():
        lines.append(f'| {aug} | {typ} | ' + ' | '.join(f'{r[c]:.3f}' for c in SCORES + ['gain']) + ' |')
    lines.append('')
    (RESULTS / 'ppt').mkdir(exist_ok=True)
    (RESULTS / 'ppt' / 'gen_quality_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root')
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    analyze() if a.analyze else compute(a)
