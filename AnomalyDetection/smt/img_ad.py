"""SMT 사진 단위 모델 (DevNet · MLP): 위치 모델(seg_ad.py)과 같은 데이터 · 같은 확인으로 비교한다 (11/27).

질문: SMT는 결함이 사진의 20–50%로 커서(VisA는 0.2–0.5%) 사진 단위 라벨만으로도 되지 않을까?
      원래 시험셋 점수가 비슷해도 촬영 조건(지름길)을 보고 맞히는 것일 수 있으니, 결함을 지운 사진으로 확인한다.
모델  고정 WRN50 패치 특징(ad/features.PatchFeatures) → 이미지 임베딩 [공간 평균, 공간 최대] 2048차원
      → ad/models.EmbeddingHead ('devnet' = deviation loss, 'mlp' = BCE). VisA ad/run.py와 같은 머리, 같은 설정(600 step, 배치 64).
      머리는 시드 2개로 학습해 지표를 평균한다(VisA와 같음).
데이터 실제 불량만(합성 없음). 정상 학습 500장, 해상도 512 (SMT 원본 그대로), 전처리 없음 — seg_ad.py의 'none'과 같다.
평가  seg_ad.py와 같다.
      시험셋: FNR@FPR1%(임계값 = 시험 정상), AUROC, 종류별 FNR.
      지름길 확인(seg_ad.diag_images, pool 불량 중 학습 K장을 뺀 나머지, 과검률 1%):
        D  = 원래 불량 사진을 불량이라 함 (임계값: 바꾸지 않은 extra_normal)
        DR = 결함을 지운 불량 사진을 불량이라 함 (낮을수록 결함을 본다. 임계값: 붙이기 대조 NR)
        ND = 정상 사진에 실제 결함을 붙인 것을 불량이라 함 (높을수록 결함을 본다)

    python -m AnomalyDetection.smt.img_ad --category pre --ks 1 2 4 8 16 32 --seeds 0 1 2 3 4
결과: AnomalyDetection/results/smt/img_ad_{category}.csv (이미 있는 (K, seed, model)은 건너뜀)
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..ad.data import read_csv
from ..ad.eval import threshold_at_fpr
from ..ad.features import PatchFeatures, device, embed, to_tensor
from ..ad.models import EmbeddingHead
from .data import SPLITS, SmtData
from .seg_ad import OUT_DIR, RES, diag_images

FIELDS = ['category', 'K', 'seed', 'model', 'fnr', 'auroc', 'D', 'DR', 'ND', 'per_type', 'seconds']
HEAD_SEEDS = (0, 1)


def embeddings(ex, dev, arrays, bs=16):
    out = []
    for i in range(0, len(arrays), bs):
        with torch.no_grad():
            out.append(embed(ex(to_tensor(np.stack(arrays[i:i + bs])).to(dev))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--category', required=True, choices=['pre', 'solder'])
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 4, 16])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--models', nargs='+', default=['devnet', 'mlp'], choices=['devnet', 'mlp'])
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev, cat = device(), a.category
    D = SmtData('none')
    load = lambda rel: D.load(rel, RES)  # noqa: E731
    t0 = time.time()
    ex = PatchFeatures().to(dev)
    Xn = embeddings(ex, dev, [load(it.image) for it in D.train_normal(cat)])
    test_rows = [r for r in read_csv(SPLITS / 'test.csv') if r['category'] == cat]
    assert [r['image'] for r in test_rows] == [it.image for it in D.test(cat)]
    Xt = embeddings(ex, dev, [load(r['image']) for r in test_rows])
    yt = np.array([int(r['label']) for r in test_rows])
    tt = [r['defect_type'] for r in test_rows]
    pool, diag = diag_images(D, cat, load)
    Xd = {c: embeddings(ex, dev, list(v)) for c, v in diag.items()}
    print(f'{cat}: 임베딩 준비 {time.time() - t0:.0f}s (정상 {len(Xn)}, 시험 {len(Xt)}, 진단 pool {len(pool)})', flush=True)

    OUT = OUT_DIR / f'img_ad_{cat}.csv'
    done = {(r['K'], r['seed'], r['model']) for r in read_csv(OUT)} if OUT.exists() else set()
    new = not OUT.exists()
    f = open(OUT, 'a', newline='')
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for k in a.ks:
        for seed in a.seeds:
            kshot = [r for r in read_csv(SPLITS / f'train_K{k}_seed{seed}.csv') if r['category'] == cat]
            assert not ({r['image'] for r in kshot} & D.forbidden)
            Xa = embeddings(ex, dev, [load(r['image']) for r in kshot])
            held = np.array([p not in {r['image'] for r in kshot} for p in pool])
            for m in a.models:
                if (str(k), str(seed), m) in done:
                    continue
                t1 = time.time()
                rows = []
                for hs in HEAD_SEEDS:
                    head = EmbeddingHead(m, seed=seed * 100 + hs).fit(Xn, Xa)
                    st = head.score(Xt)
                    miss = st <= threshold_at_fpr(st[yt == 0], 0.01)
                    sd = {c: head.score(v) for c, v in Xd.items()}
                    t_orig, t_swap = threshold_at_fpr(sd['R'], 0.01), threshold_at_fpr(sd['NR'], 0.01)
                    rows.append(dict(fnr=float(np.mean(miss[yt == 1])), auroc=float(roc_auc_score(yt, st)),
                                     D=float(np.mean(sd['D'][held] > t_orig)), DR=float(np.mean(sd['DR'][held] > t_swap)),
                                     ND=float(np.mean(sd['ND'][held] > t_swap)),
                                     per_type={t: float(np.mean(miss[[yt[i] == 1 and tt[i] == t for i in range(len(yt))]]))
                                               for t in sorted(set(tt) - {''})}))
                avg = {key: float(np.mean([r[key] for r in rows])) for key in ('fnr', 'auroc', 'D', 'DR', 'ND')}
                per_type = {t: float(np.mean([r['per_type'][t] for r in rows])) for t in rows[0]['per_type']}
                row = dict(category=cat, K=k, seed=seed, model=m, **avg, per_type=json.dumps(per_type), seconds=round(time.time() - t1))
                w.writerow(row)
                f.flush()
                print(f'{cat} K={k} seed={seed} {m:6s} FNR={avg["fnr"]:.3f} AUROC={avg["auroc"]:.3f} '
                      f'D={avg["D"]:.2f} DR={avg["DR"]:.2f} ND={avg["ND"]:.2f} ({row["seconds"]}s)', flush=True)
    f.close()


if __name__ == '__main__':
    main()
