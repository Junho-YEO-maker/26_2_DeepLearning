"""촬영 조건 지름길 점검 (13.5절).

1) 결함과 무관한 전역 통계 8개(밝기, RGB 평균, 대비, 기판 면적·위치)로 train_normal vs test 정상 / test 불량을
   로지스틱 회귀로 구분한 5-fold AUROC. 정상끼리 ≈ 0.5인데 불량만 높으면 불량 사진의 촬영 조건이 다르다는 뜻.
2) --photonorm: 이미지마다 채널 평균을 그 PCB train_normal 평균으로 맞춘 뒤(밝기·색 지름길 제거) 임베딩을 새로 계산해
   none / 생성 증강(concat, balanced)의 FNR@FPR1%를 다시 잰다. 캐시하지 않는다(GPU 서버용, CPU는 PCB당 20분+).

    python -m AnomalyDetection.ad.shortcut_check [--data-root ...]
    python -m AnomalyDetection.ad.shortcut_check --photonorm --aug sd_lora --k 1 --seed 0 [--data-root ...]
"""
import argparse

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .data import PCBS, Data
from .eval import evaluate
from .features import PatchFeatures, device, embed, to_tensor
from .models import EmbeddingHead

STATS = ['gray', 'R', 'G', 'B', 'std', 'board_frac', 'board_y', 'board_x']


def global_stats(data, items):
    out = []
    for x in items:
        with Image.open(data.path(x.image)) as im:
            a = np.asarray(im.convert('RGB').resize((256, 196)), np.float32)
        bm = (a[..., 2] - a[..., 0]) > 30
        ys, xs = np.where(bm)
        out.append([a.mean(), *a.mean((0, 1)), a.std(), bm.mean(),
                    ys.mean() if len(ys) else 0, xs.mean() if len(xs) else 0])
    return np.array(out)


def separability(A, B):
    X, y = np.concatenate([A, B]), np.r_[np.zeros(len(A)), np.ones(len(B))]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return cross_val_score(LogisticRegression(max_iter=3000, class_weight='balanced'), X, y,
                           cv=5, scoring='roc_auc').mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--data-root')
    ap.add_argument('--photonorm', action='store_true')
    ap.add_argument('--aug', default='sd_lora')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', type=int, default=512)
    a = ap.parse_args()
    data = Data(a.data_root)
    rng = np.random.default_rng(0)

    for pcb in a.pcbs:
        tn, test = data.train_normal(pcb), data.test(pcb)
        tn = [tn[i] for i in rng.choice(len(tn), 200, replace=False)]
        tnor, tdef = [x for x in test if x.label == 0][:200], [x for x in test if x.label == 1]
        A, N2, D = global_stats(data, tn), global_stats(data, tnor), global_stats(data, tdef)
        eff = (D.mean(0) - A.mean(0)) / np.sqrt((A.var(0) + D.var(0)) / 2)
        print(f'{pcb}: 전역 통계 AUROC  정상 train vs test 정상 {separability(A, N2):.3f} | vs test 불량 {separability(A, D):.3f}')
        print('   효과크기 d (불량 − 정상): ' + '  '.join(f'{n}={e:+.2f}' for n, e in zip(STATS, eff)))

    if not a.photonorm:
        return
    dev = device()
    ex = PatchFeatures().to(dev)

    def emb(items, ref):
        out = []
        for i in range(0, len(items), 16):
            arr = [np.clip(data.load(it.image, a.res).astype(np.float32)
                           - data.load(it.image, a.res).astype(np.float32).mean((0, 1)) + ref, 0, 255).astype(np.uint8)
                   for it in items[i:i + 16]]
            out.append(embed(ex(to_tensor(np.stack(arr)).to(dev))).float().cpu().numpy())
        return np.concatenate(out)

    for pcb in a.pcbs:
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        ref = np.mean([data.load(x.image, a.res).astype(np.float32).mean((0, 1)) for x in nl], 0)
        Xt, Xv, Xn, Xr, Xg = (emb(x, ref) for x in (test, val, nl, real, gen))
        rep = max(len(Xg) // len(Xr), 1)
        for kind in ('devnet', 'mlp'):
            for name, Xa in {'none': Xr, f'{a.aug} concat': np.concatenate([Xr, Xg]),
                             f'{a.aug} balanced': np.concatenate([np.repeat(Xr, rep, 0), Xg])}.items():
                h = EmbeddingHead(kind, seed=a.seed).fit(Xn, Xa)
                m = evaluate(h.score(Xt), [x.label for x in test], [x.defect_type for x in test], h.score(Xv))
                print(f'{pcb} photonorm {kind:6s} {name:22s} FNR@1%={m["fnr_at_fpr1"]:.3f} AUROC={m["auroc"]:.4f}',
                      flush=True)


if __name__ == '__main__':
    main()
