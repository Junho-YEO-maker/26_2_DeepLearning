"""임계값만 바꾸면 생성 증강이 이기는가? (11/16)

주 지표 FNR@FPR1%는 시험셋 정상 점수로 정한 임계값(정상 중 1%까지만 넘게 하는 가장 낮은 값)을 쓴다.
즉 FPR 1%에서는 이미 가장 유리한 임계값이다. 임계값을 더 낮추면 과검이 늘어난다.
여기서는 과검 허용치를 1·2·5·10·20%로 바꿔 가며 미검률을 비교한다(ROC 곡선의 여러 지점).
또 FPR 1% 임계값을 정하는 상위 정상 점수(시험 정상 약 300장 중 상위 3~4장)가 얼마나 튀는지도 본다.

  이미지 단위: canon 전처리 임베딩 + EmbeddingHead (DevNet, MLP)
  패치 단위:   patch_ad와 같은 설정 (DevNet, MLP, kNN)
  학습: 실제 K장만(none) vs 실제 + 생성(--aug). 헤드 시드 5개.

    python -m AnomalyDetection.ad.threshold_check --aug sd_lora_v3 [--data-root ...]
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, Data, read_csv
from .eval import threshold_at_fpr
from .features import PatchFeatures, device
from .models import EmbeddingHead
from .patch_ad import RES, KnnHead, PatchHead, cell_labels, load_mask, patch_feats, sample_cells
from .prep import embedder

FPRS = (0.01, 0.02, 0.05, 0.10, 0.20)


def curve(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    out = {}
    for f in FPRS:
        thr = threshold_at_fpr(s[y == 0], f)
        out[f'fnr@{int(f * 100)}'] = float(np.mean(s[y == 1] <= thr))
    sn = np.sort(s[y == 0])[::-1]
    # 정상 점수 분포에서 상위 꼬리가 얼마나 튀는지: (상위 4번째 - 상위 20번째) / (상위 20번째 - 중앙값)
    out['tail'] = float((sn[3] - sn[19]) / max(sn[19] - np.median(sn), 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    rows, scores = [], {}
    for pcb in a.pcbs:
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        y = np.array([x.label for x in test])

        # 이미지 단위 (canon)
        emb_items, _ = embedder(data, ex, dev, pcb, 512, 'canon')
        Xt, Xn, Xr, Xg = (emb_items(x) for x in (test, nl, real, gen))
        for m in ('devnet', 'mlp'):
            for name, XA in (('none', Xr), (f'+{a.aug}', np.concatenate([Xr, Xg]))):
                for hs in range(a.head_seeds):
                    s = EmbeddingHead(m, seed=hs).fit(Xn, XA).score(Xt)
                    scores[f'{pcb}/image/{m}/{name}/{hs}'] = s
                    rows.append(dict(pcb=pcb, level='image', model=m, train=name, head_seed=hs, **curve(s, y)))
        print(f'{pcb}: 이미지 단위 끝', flush=True)

        # 패치 단위 (patch_ad와 같은 설정)
        ref = np.mean([data.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl], 0)
        gen_masks = {r['image']: r['mask'] for r in
                     read_csv(data.generated / a.aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv')}
        Pt, Pn, Pr, Pg = (patch_feats(data, ex, dev, x, ref) for x in (test, nl, real, gen))
        Lr = np.stack([cell_labels(load_mask(data, x, gen_masks)).reshape(-1) for x in real])
        Lg = np.stack([cell_labels(load_mask(data, x, gen_masks)).reshape(-1) for x in gen])
        rng = np.random.default_rng(0)
        bank = sample_cells(Pn, np.zeros(Pn.shape[:2], np.int8), 0, 128, rng)
        s = KnnHead().fit(bank).score(Pt).max(1)
        scores[f'{pcb}/patch/knn/none/0'] = s
        rows.append(dict(pcb=pcb, level='patch', model='knn', train='none', head_seed=0, **curve(s, y)))
        for name, (Pd, Ld) in (('none', (Pr, Lr)), (f'+{a.aug}', (np.concatenate([Pr, Pg]), np.concatenate([Lr, Lg])))):
            pos = sample_cells(Pd, Ld, 1, None, rng)
            Pnorm = np.concatenate([bank, sample_cells(Pd, Ld, 0, 128, rng)])
            for m in ('devnet', 'mlp'):
                for hs in range(a.head_seeds):
                    s = PatchHead(m, seed=hs, dev=dev).fit(Pnorm, pos).score(Pt).max(1)
                    scores[f'{pcb}/patch/{m}/{name}/{hs}'] = s
                    rows.append(dict(pcb=pcb, level='patch', model=m, train=name, head_seed=hs, **curve(s, y)))
        scores[f'{pcb}/labels'] = y
        print(f'{pcb}: 패치 단위 끝', flush=True)

    res = ROOT / 'AnomalyDetection/results'
    with open(res / f'threshold_check_{a.aug}_K{a.k}_seed{a.seed}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(ROOT / 'Dataset/cache' / f'threshold_check_scores_{a.aug}_K{a.k}_seed{a.seed}.npz', **scores)

    keys = [f'fnr@{int(f * 100)}' for f in FPRS]
    print('\n과검 허용치(FPR)별 미검률 FNR (PCB 평균, 헤드 시드 평균). 낮을수록 좋음. tail = 정상 점수 상위 꼬리 튐 정도')
    print(f'  {"단위":6s} {"모델":6s} {"학습":14s} ' + ' '.join(f'{k:>7s}' for k in keys) + '   tail')
    for lvl in ('image', 'patch'):
        for m in (('devnet', 'mlp') if lvl == 'image' else ('knn', 'devnet', 'mlp')):
            for name in (('none',) if m == 'knn' else ('none', f'+{a.aug}')):
                rs = [r for r in rows if r['level'] == lvl and r['model'] == m and r['train'] == name]
                v = {k: np.mean([np.mean([r[k] for r in rs if r['pcb'] == p]) for p in a.pcbs]) for k in keys + ['tail']}
                print(f'  {lvl:6s} {m:6s} {name:14s} ' + ' '.join(f'{v[k]:7.3f}' for k in keys) + f'  {v["tail"]:5.2f}',
                      flush=True)


if __name__ == '__main__':
    main()
