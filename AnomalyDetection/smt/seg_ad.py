"""SMT 분할 AD: 결함 박스로 학습하는 위치별 판정 위에서 합성 결함의 값어치를 다시 본다 (11/18).

이미지 임베딩 하나로 판정하면 결함(사진의 약 5%)이 묻혀 합성 불량이 "불량 라벨 붙은 정상"이 된다.
여기서는 라벨이 결함 박스 안에만 걸린다. 불량 사진의 박스 밖과 붙이기 정상은 정상으로 학습한다.
모델  VisA ad/local_ad.py의 SegModel(고정 WRN50 layer1–3 + 작은 FPN 디코더, focal + dice, 결함 테두리 근처 무시)을 그대로 쓰고
      불량 배치 안 합성 비율 r과 붙이기 정상(정상 쪽에 더함)만 보탰다. 이미지 점수 = 128×128 지도 상위 10칸 평균.
마스크 실제 = label_json의 불량 박스, 합성 = manifest condition의 붙인 박스.
조건  'none' = 실제 K장만.
      '{method}:{n}:{r}' — method = Dataset/aihub_smt/generated/ 아래 폴더, n = 종류당 앞에서 n장(n이 커져도 앞의 것을 포함),
      r = 불량 배치 안 합성 비율(0.5 = VisA SegModel 기본, 0.25, nat = 실제와 한 풀로 섞기). 짝 붙이기 정상이 있으면 같은 n장을 정상에 더한다.
평가  시험셋: FNR@FPR1%(임계값 = 시험 정상), AUROC, 종류별 FNR, 위치 적중률(시험 불량에서 최고점 칸이 불량 박스 ±8px 안).
      지름길 진단(defect_removal_check와 같은 방식, pool 불량 중 학습 K장을 뺀 나머지, 과검률 1%):
        D(원래 불량 사진, 임계값 = R 바꾸지 않은 extra_normal) / DR(결함 지움)·ND(정상 + 실제 결함) (임계값 = NR 붙이기 대조).

    python -m AnomalyDetection.smt.seg_ad --category pre --k 1 --seeds 0 1 2 --conds none transplant_n40:5:0.5
결과: AnomalyDetection/results/smt/seg_ad_{category}.csv (이미 있는 (category, K, seed, cond, steps)는 건너뜀)
"""
import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from ..ad.data import ROOT, read_csv
from ..ad.eval import threshold_at_fpr
from ..ad.features import PatchFeatures, device
from ..ad.local_ad import Backbone, SegHead, SegModel, aggregate, photonorm, to_gpu, to_input
from .data import GENERATED, SPLITS, SmtData

sys.path.insert(0, str(ROOT / 'Augmentation'))
from smt_sd_lora.smt import box_mask, boxes, paste  # noqa: E402

RES = 512
OUT_DIR = ROOT / 'AnomalyDetection/results/smt'
N_REF = 400
FIELDS = ['category', 'K', 'seed', 'cond', 'steps', 'n_gen', 'n_neg', 'fnr', 'auroc', 'loc_hit', 'D', 'DR', 'ND',
          'per_type', 'seconds']


class MixSeg(SegModel):
    """SegModel과 같은 학습. 불량 절반 배치 안 합성 수만 r로 정한다(r=0.5면 SegModel과 같음)."""

    def fit(self, normals, real, real_masks, gen=None, gen_masks=None, r=0.5):
        dev = self.dev
        torch.manual_seed(self.seed)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        with torch.no_grad():
            f = self.bb(to_input(photonorm(to_gpu(normals[rng.choice(len(normals), 32, replace=False)], dev), self.ref)))
            self.stats = [(a.mean((0, 2, 3), keepdim=True), a.std((0, 2, 3), keepdim=True) + 1e-4) for a in f]
        self.head = SegHead().to(dev)
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        half = self.batch // 2
        for step in range(self.steps):
            ni = rng.choice(len(normals), half)
            if gen is None or not len(gen):
                ng = 0
            elif r == 'nat':
                ng = int(rng.binomial(half, len(gen) / (len(gen) + len(real))))
            else:
                ng = int(round(float(r) * half))
            src = [(real, real_masks, rng.integers(len(real))) for _ in range(half - ng)] + \
                  [(gen, gen_masks, rng.integers(len(gen))) for _ in range(ng)]
            imgs = np.concatenate([normals[ni], np.stack([im[i] for im, _, i in src])])
            msk = np.concatenate([np.zeros((half, RES, RES), bool), np.stack([mk[i] for _, mk, i in src])])
            x = to_gpu(imgs, dev)
            m = torch.as_tensor(msk, device=dev).float()[:, None]
            flip = torch.rand(2, generator=g, device=dev)
            if flip[0] < 0.5:
                x, m = x.flip(3), m.flip(3)
            if flip[1] < 0.5:
                x, m = x.flip(2), m.flip(2)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = self.head(*self.feats(x)).float()
            frac = F.avg_pool2d(m, RES // 128)[:, 0]
            pos = (frac > 0.25).float()
            near = F.max_pool2d(F.max_pool2d(m, RES // 128), 5, 1, 2)[:, 0] > 0
            w = (~near | (pos > 0)).float()
            p = torch.sigmoid(logit)
            pt = torch.where(pos > 0, p, 1 - p)
            focal = (F.binary_cross_entropy_with_logits(logit, pos, reduction='none') * (1 - pt) ** 2 * w).sum() / w.sum()
            inter = (p * pos * w).sum()
            dice = 1 - (2 * inter + 1) / ((p * w).sum() + pos.sum() + 1)
            loss = focal * 10 + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        self.head.eval()
        return self


def defect_mask(row, cat, dilate=0):
    return box_mask([b for lab, _, b in boxes(row, cat) if lab == '불량'], dilate=dilate)


def swap(dst, src, region):
    return np.asarray(paste(Image.fromarray(dst), Image.fromarray(src), region, feather=3))


def diag_images(D, cat, load):
    """defect_removal_check와 같은 짝 만들기(rng 0). 기준 정상은 N_REF장."""
    pool = [r for r in read_csv(SPLITS / 'pool.csv') if r['category'] == cat]
    rng = np.random.default_rng(0)
    extra, train = D.extra_normal(cat), D.train_normal(cat)
    ref = [extra[i] for i in rng.choice(len(extra), min(N_REF, len(extra)), replace=False)]
    regions = [defect_mask(r, cat, dilate=8) for r in pool]
    Dm, DR, ND, NR = [], [], [], []
    for r, m in zip(pool, regions):
        d = load(r['image'])
        Dm.append(d)
        DR.append(swap(d, load(train[rng.integers(len(train))].image), m))
        ND.append(swap(load(ref[rng.integers(len(ref))].image), d, m))
    R = [load(it.image) for it in ref]
    for x in R:
        NR.append(swap(x, load(train[rng.integers(len(train))].image), regions[rng.integers(len(regions))]))
    return [r['image'] for r in pool], {k: np.stack(v) for k, v in dict(D=Dm, DR=DR, ND=ND, R=R, NR=NR).items()}


def generated(D, method, cat, k, seed, n, load):
    """종류마다 manifest 순서로 앞의 n장 → (이미지, 마스크, 짝 붙이기 정상 이미지)."""
    D.generated_defects(method, cat, k, seed)                 # 누수 검사
    rows = [r for r in read_csv(GENERATED / method / f'K{k}_seed{seed}' / 'manifest.csv') if r['category'] == cat]
    pos = [r for r in rows if r['label'] == '1']
    chosen = []
    for t in sorted({r['defect_type'] for r in pos}):
        rt = [r for r in pos if r['defect_type'] == t]
        assert len(rt) >= n, (method, t, len(rt), n)
        chosen += rt[:n]
    neg = {r['pair']: r for r in rows if r['label'] == '0' and r.get('pair')}
    imgs = np.stack([load(r['image']) for r in chosen])
    masks = np.stack([box_mask([tuple(json.loads(r['condition'])['box'])]) for r in chosen])
    negs = np.stack([load(neg[r['pair']]['image']) for r in chosen if r.get('pair') in neg]) if neg else None
    return imgs, masks, negs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--category', required=True, choices=['pre', 'solder'])
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--conds', nargs='+', default=['none'])
    ap.add_argument('--steps', type=int, default=2000)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev, cat, k = device(), a.category, a.k
    D = SmtData('none')
    load = lambda rel: D.load(rel, RES)
    t0 = time.time()
    An = np.stack([load(it.image) for it in D.train_normal(cat)])
    ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
    test_rows = [r for r in read_csv(SPLITS / 'test.csv') if r['category'] == cat]
    assert [r['image'] for r in test_rows] == [it.image for it in D.test(cat)]
    At = np.stack([load(r['image']) for r in test_rows])
    yt = np.array([int(r['label']) for r in test_rows])
    tt = [r['defect_type'] for r in test_rows]
    hit_mask = {i: F.max_pool2d(torch.as_tensor(defect_mask(r, cat, dilate=8)).float()[None, None], 4)[0, 0].bool().numpy()
                for i, r in enumerate(test_rows) if yt[i] == 1}
    pool, diag = diag_images(D, cat, load)
    print(f'{cat} K={k}: 데이터 준비 {time.time() - t0:.0f}s (정상 {len(An)}, 시험 {len(At)}, 진단 pool {len(pool)})', flush=True)
    bb = Backbone(PatchFeatures().to(dev)).to(dev).eval()

    OUT = OUT_DIR / f'seg_ad_{cat}.csv'
    done = set()
    if OUT.exists():
        done = {(r['category'], r['K'], r['seed'], r['cond'], r['steps']) for r in read_csv(OUT)}
    new = not OUT.exists()
    f = open(OUT, 'a', newline='')
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for seed in a.seeds:
        kshot = [r for r in read_csv(SPLITS / f'train_K{k}_seed{seed}.csv') if r['category'] == cat]
        assert not ({r['image'] for r in kshot} & D.forbidden)
        Ar = np.stack([load(r['image']) for r in kshot])
        Mr = np.stack([defect_mask(r, cat) for r in kshot])
        held = np.array([p not in {r['image'] for r in kshot} for p in pool])
        for cond in a.conds:
            if (cat, str(k), str(seed), cond, str(a.steps)) in done:
                continue
            t1 = time.time()
            gen = gm = None
            normals, n_neg, r = An, 0, 0.5
            if cond != 'none':
                method, n, r = cond.split(':')
                gen, gm, negs = generated(D, method, cat, k, seed, int(n), load)
                if negs is not None:
                    normals, n_neg = np.concatenate([An, negs]), len(negs)
            model = MixSeg(bb, ref, dev, seed=seed, steps=a.steps).fit(normals, Ar, Mr, gen, gm, r)
            St = model.score(At)
            st = aggregate(St, 'top10')
            miss = st <= threshold_at_fpr(st[yt == 0], 0.01)
            per_type = {t: float(np.mean(miss[[yt[i] == 1 and tt[i] == t for i in range(len(yt))]]))
                        for t in sorted(set(tt) - {''})}
            hits = [bool(hit_mask[i][np.unravel_index(St[i].argmax(), St[i].shape)]) for i in hit_mask]
            sd = {c: aggregate(model.score(diag[c]), 'top10') for c in diag}
            t_orig, t_swap = threshold_at_fpr(sd['R'], 0.01), threshold_at_fpr(sd['NR'], 0.01)
            row = dict(category=cat, K=k, seed=seed, cond=cond, steps=a.steps,
                       n_gen=0 if gen is None else len(gen), n_neg=n_neg,
                       fnr=float(np.mean(miss[yt == 1])), auroc=float(roc_auc_score(yt, st)), loc_hit=float(np.mean(hits)),
                       D=float(np.mean(sd['D'][held] > t_orig)), DR=float(np.mean(sd['DR'][held] > t_swap)),
                       ND=float(np.mean(sd['ND'][held] > t_swap)), per_type=json.dumps(per_type),
                       seconds=round(time.time() - t1))
            w.writerow(row)
            f.flush()
            print(f'{cat} K={k} seed={seed} {cond:24s} FNR={row["fnr"]:.3f} AUROC={row["auroc"]:.3f} '
                  f'loc={row["loc_hit"]:.2f} D={row["D"]:.2f} DR={row["DR"]:.2f} ND={row["ND"]:.2f} ({row["seconds"]}s)', flush=True)
            del model
            torch.cuda.empty_cache()
    f.close()


if __name__ == '__main__':
    main()
