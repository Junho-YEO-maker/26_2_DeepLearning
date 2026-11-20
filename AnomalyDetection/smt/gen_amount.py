"""생성량 실험 (11/20): SMT 사전공정 SD+LoRA 합성을 종류당 20 / 40 / 80 / 160장 넣으면 위치 seg가 얼마나 나아지나,
그리고 장수가 늘 때 합성이 실제로 더 다양해지나(새 결함 모양을 더 덮나)를 같이 본다.

세트: sd_lora_n160 (Augmentation/smt_sd_lora/generate.py --per-type 160 --tag n160, 320장 중 변화 큰 160장)
     ctransgan_n160 (Augmentation/smt_ctransgan/generate_n.py, 팀원이 학습한 체크포인트로 160장, 선별 없음)
     N장 = 종류마다 manifest 앞 N장 (seg_ad의 `{세트}:N:0.5`와 같은 자르기, 20 ⊂ 40 ⊂ 80 ⊂ 160).
다양성 점수 (gen_quality_smt와 같은 패치 특징: WRN50, 32×32, 정상 패치 통계로 z, 정상 은행 20000개):
  spread   합성 결함 서술자(박스 안 패치 평균)끼리 평균 거리 / 실제 시험 결함끼리 평균 거리. 장수와 무관한 퍼짐 정도.
  recall   실제 시험 결함 패치 중 가장 가까운 합성 패치가 가장 가까운 정상 패치보다 가까운 비율. 실제 결함 모양을 얼마나 덮나.
  precision 합성 결함 패치 중 가장 가까운 실제 결함 패치가 가장 가까운 정상 패치보다 가까운 비율. 진짜 결함처럼 보이나.
  dup      합성 서술자의 최근접 합성 거리가 실제 결함끼리 최근접 거리 중앙값의 절반보다 가까운 비율. 거의 같은 장의 비율.
  bases    서로 다른 바탕 정상 이미지 수 / N.
시험 결함은 점수 계산에만 쓰고 학습에는 안 쓴다.

    python -m AnomalyDetection.smt.gen_amount --diversity   # 다양성 (생성이 끝난 칸만)
    python -m AnomalyDetection.smt.gen_amount --compare     # seg 결과 + 다양성 표
결과: AnomalyDetection/results/smt/gen_amount_diversity.csv, gen_amount.csv, AnomalyDetection/results/ppt/gen_amount_summary.md
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from ..ad.data import ROOT, Item, read_csv
from ..ad.features import PatchFeatures, device
from ..ad.gen_quality import pair_stats
from ..ad.gen_realism import nn_dist
from ..ad.patch_ad import RES, cell_labels, patch_feats
from .data import GENERATED, SmtData
from .seg_ad import defect_mask

RESULTS = ROOT / 'AnomalyDetection/results/smt'
CAT = 'pre'
METHODS = [('sd_lora_n160', 'SD+LoRA', 'sd_lora_fix'), ('ctransgan_n160', 'cTransGAN', 'ctransgan')]   # (세트, 이름, 기존 40장 세트)
KS = [(k, s) for k in (1, 4) for s in (0, 1, 2)]
NS = (20, 40, 80, 160)


def diversity():
    torch.set_num_threads(4)
    D, dev = SmtData('none'), device()
    ex = PatchFeatures().to(dev)
    types = D.types[CAT]
    rng = np.random.default_rng(0)
    nl = D.train_normal(CAT)
    nl_s = [nl[i] for i in rng.choice(len(nl), min(200, len(nl)), replace=False)]
    ref = np.mean([D.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl_s], 0)
    feats = lambda items: patch_feats(D, ex, dev, items, ref).astype(np.float32)        # noqa: E731
    Pn = feats(nl_s).reshape(-1, 1024)
    Pn = Pn[rng.choice(len(Pn), 20000, replace=False)]
    mu, sd = Pn.mean(0), Pn.std(0) + 1e-6
    z = lambda X: (X - mu) / sd                                                         # noqa: E731
    Bn = z(Pn)
    rr = [r for r in read_csv(D.splits / 'test.csv') if r['category'] == CAT and r['label'] == '1' and r['defect_type'] in types]
    RF = z(feats([Item(r['image'], 1, r['defect_type'], 'real') for r in rr]))
    rlab = [cell_labels(defect_mask(r, CAT)).reshape(-1) == 1 for r in rr]
    real = {}
    for t in types:
        idx = [i for i, r in enumerate(rr) if r['defect_type'] == t and rlab[i].any()]
        R = np.concatenate([RF[i][rlab[i]] for i in idx])
        rpair, rnn = pair_stats(np.stack([RF[i][rlab[i]].mean(0) for i in idx]))
        real[t] = dict(R=R, dn=nn_dist(R, Bn, dev), pair=rpair, nn=rnn, n=len(idx))
    del RF
    out = []
    for (method, gname, _), (k, s) in [(m, ks) for m in METHODS for ks in KS]:
        man = GENERATED / method / f'K{k}_seed{s}' / 'manifest.csv'
        if not man.exists():
            print(f'{gname} K{k} seed{s}: 아직 없음')
            continue
        rows = [r for r in read_csv(man) if r['category'] == CAT and r['label'] == '1']
        for t in types:
            rt = [r for r in rows if r['defect_type'] == t]
            GF = z(feats([Item(r['image'], 1, t, method) for r in rt]))
            glab = []
            for r in rt:
                with Image.open(ROOT / r['mask']) as im:
                    glab.append(cell_labels(np.asarray(im.convert('L').resize((RES, RES), Image.NEAREST)) > 0).reshape(-1) == 1)
            ok = [i for i in range(len(rt)) if glab[i].any()]
            desc = np.stack([GF[i][glab[i]].mean(0) if glab[i].any() else np.zeros(1024, np.float32) for i in range(len(rt))])
            Rt = real[t]
            for n in NS:
                sel = [i for i in ok if i < n]
                if len(sel) < 2:
                    continue
                G = np.concatenate([GF[i][glab[i]] for i in sel])
                d = torch.cdist(torch.as_tensor(desc[sel]), torch.as_tensor(desc[sel])).numpy()
                np.fill_diagonal(d, np.inf)
                gpair, _ = pair_stats(desc[sel])
                out.append(dict(gen=gname, K=k, seed=s, type=t, N=n, n_ok=len(sel), spread=gpair / Rt['pair'],
                                recall=float(np.mean(nn_dist(Rt['R'], G, dev) < Rt['dn'])),
                                precision=float(np.mean(nn_dist(G, Rt['R'], dev) < nn_dist(G, Bn, dev))),
                                dup=float(np.mean(d.min(1) < 0.5 * Rt['nn'])),
                                bases=len({rt[i]['base_normal_image'] for i in range(n)}) / n))
            print(f'{gname} K{k} seed{s} {t}: ' + ' '.join(f"N{r['N']} rec {r['recall']:.2f} spr {r['spread']:.2f}"
                  for r in out if (r['gen'], r['K'], r['seed'], r['type']) == (gname, k, s, t)), flush=True)
    import pandas as pd
    pd.DataFrame(out).to_csv(RESULTS / 'gen_amount_diversity.csv', index=False)
    print('저장:', RESULTS / 'gen_amount_diversity.csv')


def compare():
    import pandas as pd
    d = pd.read_csv(RESULTS / f'seg_ad_{CAT}.csv')
    rows = []
    for method, gname, ref in METHODS:
        conds = [('none', '실제만'), (f'{ref}:20:0.5', '기존 20'), (f'{ref}:40:0.5', '기존 40')] + \
                [(f'{method}:{n}:0.5', f'{n}장') for n in NS]
        for k, s in KS:
            x = d[(d.K == k) & (d.seed == s)]
            for c, name in conds:
                y = x[x.cond == c]
                if len(y):
                    r = y.iloc[-1]
                    rows.append(dict(gen=gname, K=k, seed=s, cond=c, name=name, fnr=r.fnr, auroc=r.auroc, DR=r.DR, ND=r.ND,
                                     **{f'fnr_{t}': v for t, v in json.loads(r.per_type).items() if '|' not in t}))
    a = pd.DataFrame(rows)
    a.to_csv(RESULTS / 'gen_amount.csv', index=False)
    tcols = sorted(c for c in a.columns if c.startswith('fnr_'))
    names = ['실제만', '기존 20', '기존 40'] + [f'{n}장' for n in NS]
    lines = ['# 생성량 실험: SMT 사전공정 합성 20 / 40 / 80 / 160장 (gen_amount.py)', '',
             '위치 seg, 표준 시험셋. 미검률 = 과검 1%에서 놓친 불량 비율(낮을수록 좋음). seed 0–2 평균 (± seed 표준편차).',
             '"기존 20·40" = 이미 있던 종류당 40장 세트(SD+LoRA는 sd_lora_fix, cTransGAN은 ctransgan). '
             'N장 = 새로 만든 160장 세트의 앞 N장.', '',
             '## 한눈에 (미검률, seed 평균)', '', '| 생성기 | K | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for (g, k), b in a.groupby(['gen', 'K'], sort=False):
        m = b.groupby('name').fnr.mean()
        lines.append(f'| {g} | {k} | ' + ' | '.join(f'{m[n]:.3f}' if n in m else '' for n in names) + ' |')
    lines.append('')
    for (g, k), b in a.groupby(['gen', 'K'], sort=False):
        ns = b.groupby('name').seed.nunique()
        lines += [f'## {g}, K={k}', '', '| 조건 | seed 수 | 미검률 | AUROC | DR | ND | ' + ' | '.join(c[4:] for c in tcols) + ' |',
                  '|---|---|---|---|---|---|' + '---|' * len(tcols)]
        for name in names:
            c = b[b.name == name]
            if not len(c):
                continue
            lines.append(f'| {name} | {ns[name]} | {c.fnr.mean():.3f} ± {c.fnr.std(ddof=0):.3f} | {c.auroc.mean():.3f} | '
                         f'{c.DR.mean():.2f} | {c.ND.mean():.2f} | ' + ' | '.join(f'{c[t].mean():.2f}' for t in tcols) + ' |')
        lines.append('')
    p = RESULTS / 'gen_amount_diversity.csv'
    if p.exists():
        v = pd.read_csv(p)
        if 'gen' not in v:                       # 11/20 SD+LoRA만 있던 때의 파일
            v.insert(0, 'gen', 'SD+LoRA')
        lines += ['## 장수가 늘 때 합성이 더 다양해지나 (종류·seed 평균)', '',
                  'recall = 실제 시험 결함 패치를 합성이 덮는 비율(높을수록 실제 결함 모양을 더 많이 그림). '
                  'precision = 합성 패치가 실제 결함 쪽에 가까운 비율. spread = 합성끼리 퍼짐 / 실제끼리 퍼짐. '
                  'dup = 거의 같은 장 비율. bases = 서로 다른 바탕 비율.', '',
                  '| 생성기 | K | N | recall | precision | spread | dup | bases |', '|---|---|---|---|---|---|---|---|']
        t = v.groupby(['gen', 'K', 'N'], sort=False)[['recall', 'precision', 'spread', 'dup', 'bases']].mean()
        lines += [f'| {g} | {k} | {n} | {r.recall:.3f} | {r.precision:.3f} | {r.spread:.2f} | {r.dup:.2f} | {r.bases:.2f} |'
                  for (g, k, n), r in t.iterrows()]
        tt = v.groupby(['gen', 'K', 'type', 'N'], sort=False).recall.mean().unstack('N')
        lines += ['', '### 종류별 recall (seed 평균)', '', '| 생성기 | K | 종류 | ' + ' | '.join(f'N{n}' for n in tt.columns) + ' |',
                  '|---|---|---|' + '---|' * len(tt.columns)]
        lines += [f'| {g} | {k} | {t} | ' + ' | '.join(f'{x:.3f}' for x in r) + ' |' for (g, k, t), r in tt.iterrows()]
        lines.append('')
    (ROOT / 'AnomalyDetection/results/ppt/gen_amount_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diversity', action='store_true')
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    diversity() if a.diversity else compare()
