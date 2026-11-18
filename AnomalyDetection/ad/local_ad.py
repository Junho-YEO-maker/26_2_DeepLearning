"""지름길을 줄이는 국소 판정 세 가지를 순서대로 비교한다 (11/18). patch_ad와 같은 데이터·라벨·헤드 위에서.

  score  학습은 patch_ad 그대로, 패치 점수 → 이미지 점수 집계만 바꾼다.
         max(기존) / max−med(사진 안 패치 점수 중앙값을 뺀다: 사진 전체에 깔린 촬영 차이를 상쇄)
         / sm3(3×3 평균 뒤 최댓값: 튀는 패치 하나를 누른다) / sm3−med / top10(상위 10개 평균) / top10−med
  aug    + 학습 이미지에 촬영 조건 무작위화(capture_aug): 밝기·대비, 조명 기울기, 채도, 블러↔샤픈, 잡음.
         정상은 원본 + 2벌, 실제 K장은 원본 + 8벌, 생성은 원본 + 1벌. kNN 은행도 무작위화한 정상으로 만든다.
  seg    고정 WRN50 layer1–3 위 작은 FPN 디코더로 128×128 결함 지도(focal + dice). 마스크 = 실제 K장(+ 생성).
         배치 = 정상 4 + 불량 4(생성이 있으면 불량의 절반은 실제). --seg-aug면 capture_aug도 쓴다.
  img    비교 기준: 이미지 임베딩(canon) + DevNet/MLP (label_weight_check r=0과 같은 헤드). 지도가 없어 agg = image.
평가
  시험셋: FNR@FPR1%·5%(임계값 = 시험 정상), AUROC, 픽셀 AUROC(64×64), 위치 적중률, 종류별 FNR
  지름길 진단(시험셋 아님): pool 불량으로 만든 D(실제) / DR(결함 지움) / ND(정상 + 실제 결함) / NR(붙이기 대조)
    defect_removal_check와 같은 짝, 결함 영역 40px. val_normal FPR 1% 임계값으로 불량 판정 비율. DR 낮을수록, ND 높을수록 좋다.
전처리: photonorm(채널 평균 맞춤, 기하 변환 없음 → 마스크가 그대로 맞는다). patch_ad와 같다.

    python -m AnomalyDetection.ad.local_ad --stage score [--data-root ...]
    python -m AnomalyDetection.ad.local_ad --stage aug
    python -m AnomalyDetection.ad.local_ad --stage seg [--seg-aug]
    python -m AnomalyDetection.ad.local_ad --stage seg --seed 1 --no-gen --no-diag   # 생성이 없는 분할, 진단 생략
결과: AnomalyDetection/results/local_ad_{stage}_K{k}_seed{s}.csv
"""
import argparse
import csv
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from .data import PCBS, ROOT, TYPES, Data, read_csv
from .defect_removal_check import region_in
from .eval import evaluate, threshold_at_fpr
from .features import MEAN, STD, PatchFeatures, device
from .models import EmbeddingHead
from .patch_ad import GRID, RES, KnnHead, PatchHead, cell_labels, load_mask, sample_cells
from .prep import embedder

sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import (STRUCTURE_TYPES, TYPE_ID, KShot, aligned_donor, board_transform,  # noqa: E402
                          change_region, dilate, paste_back)

AGGS = ('max', 'max-med', 'sm3', 'sm3-med', 'top10', 'top10-med')
DIAG = ('D', 'DR', 'ND', 'NR')


# ---------------- 지름길 진단 사진 ----------------
def diag_set(data, ks, pcb, k_imgs, px=40):
    """defect_removal_check와 같은 짝(같은 난수)으로 D/DR/ND/NR을 RES×RES uint8로. 만들기가 느려서 캐시한다."""
    cp = ROOT / 'Dataset/cache/local_ad' / f'diag_{pcb}_K{ks.k}_seed{ks.seed}_dil{px}.npz'
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in DIAG}
    normals = ks.normals(pcb)
    rng = np.random.default_rng(7017)
    out = {k: [] for k in DIAG}
    small = lambda im: np.asarray(im.convert('RGB').resize((RES, RES), Image.BILINEAR))  # noqa: E731
    for row in [r for r in read_csv(data.splits / 'pool.csv') if r['category'] == pcb and r['image'] not in k_imgs]:
        t = row['defect_type'].split('|')[0]
        D, ids = ks.load(row)
        m_D = ids == TYPE_ID[t]
        if not m_D.any():
            continue
        cands = [normals[i] for i in rng.choice(len(normals), 5, replace=False)]
        ranked = sorted(cands, key=lambda r: -board_transform(ks.load(r)[0], D)['score'])
        N, _ = ks.load(ranked[0])
        reg_D, N_al, _ = region_in(D, N, m_D, t, px)
        W, H = D.size
        DR = paste_back(D, N_al, (0, 0, W, H), reg_D, feather=3)
        tf = board_transform(D, N)
        info = {'align': {'rot180': bool(tf['rot180'])}, 'applied_shift_xy': [int(tf['dx']), int(tf['dy'])]}
        D_al = aligned_donor(D, N, info)
        m_N = np.asarray(aligned_donor(Image.fromarray(m_D.astype(np.uint8) * 255), N, info)) > 127
        reg_N = dilate(change_region(D, N, m_N, info) if t in STRUCTURE_TYPES else m_N, px)
        Wn, Hn = N.size
        ND = paste_back(N, D_al, (0, 0, Wn, Hn), reg_N, feather=3)
        N2, _ = ks.load(ranked[1])
        tf2 = board_transform(N2, N)
        info2 = {'align': {'rot180': bool(tf2['rot180'])}, 'applied_shift_xy': [int(tf2['dx']), int(tf2['dy'])]}
        NR = paste_back(N, aligned_donor(N2, N, info2), (0, 0, Wn, Hn), reg_N, feather=3)
        for k, im in (('D', D), ('DR', DR), ('ND', ND), ('NR', NR)):
            out[k].append(small(im))
    out = {k: np.stack(v) for k, v in out.items()}
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, **out)
    return out


# ---------------- 입력 처리 ----------------
def to_gpu(arrs, dev):
    """(B, H, W, 3) uint8 → (B, 3, H, W) float 0..255 on dev."""
    return torch.as_tensor(np.ascontiguousarray(arrs), device=dev).permute(0, 3, 1, 2).float()


def photonorm(x, ref):
    return (x - x.mean((2, 3), keepdim=True) + ref.view(1, 3, 1, 1)).clamp(0, 255)


def to_input(x):
    return (x / 255 - MEAN.to(x.device)) / STD.to(x.device)


def _gauss_blur(x, sigma):
    """이미지마다 다른 sigma (B,)로 가우시안 블러 (분리형 9탭)."""
    B, C, H, W = x.shape
    r = torch.arange(-4, 5, device=x.device).float()
    k = torch.exp(-(r[None] ** 2) / (2 * sigma[:, None] ** 2))
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)                  # (B*C, 9)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (4, 4, 0, 0), mode='reflect'), k[:, None, None, :], groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, 4, 4), mode='reflect'), k[:, None, :, None], groups=B * C)
    return y.reshape(B, C, H, W)


def capture_aug(x, g):
    """촬영 조건 무작위화. x (B, 3, H, W) 0..255. capture_probe에서 불량 사진이 다르던 축을 무작위로 흔든다."""
    B, _, H, W = x.shape
    dev = x.device

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g, device=dev)

    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.85, 1.15, B, 1, 1, 1) + m * u(0.85, 1.15, B, 1, 1, 1)          # 대비·밝기
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing='ij')
    x = x * (1 + u(-0.15, 0.15, B, 1, 1, 1) * xx + u(-0.15, 0.15, B, 1, 1, 1) * yy)    # 조명 기울기
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * u(0.7, 1.4, B, 1, 1, 1)                                   # 채도
    amt = u(-1.0, 1.0, B, 1, 1, 1)                                                     # <0 블러, >0 샤픈
    x = x + amt * (x - _gauss_blur(x, u(0.5, 1.5, B)))
    x = x + torch.randn(x.shape, generator=g, device=dev) * u(0, 3, B, 1, 1, 1)         # 잡음
    return x.clamp(0, 255)


@torch.no_grad()
def patch_feats(ex, arrs, ref, dev, g=None, bs=16):
    """(N, GRID*GRID, 1024) float16. g가 있으면 capture_aug를 먼저."""
    out = []
    for i in range(0, len(arrs), bs):
        x = to_gpu(arrs[i:i + bs], dev)
        if g is not None:
            x = capture_aug(x, g)
        f = ex(to_input(photonorm(x, ref)))
        f = F.avg_pool2d(f, f.shape[-1] // GRID)
        out.append(f.flatten(2).transpose(1, 2).half().cpu().numpy())
    return np.concatenate(out)


# ---------------- 점수 집계·지표 ----------------
def aggregate(maps, how):
    """maps (N, h, w) → (N,) 이미지 점수."""
    t = torch.as_tensor(maps).float()
    flat = t.flatten(1)
    med = flat.median(1).values
    if how.startswith('sm3'):
        flat = F.avg_pool2d(t[:, None], 3, 1, 1, count_include_pad=False).flatten(1)
    if how.startswith('top10'):
        top = flat.topk(10, 1).values.mean(1)
    else:
        top = flat.max(1).values
    return (top - med if how.endswith('-med') else top).numpy()


def map_metrics(maps, labels, masks):
    """집계와 무관한 지표: 픽셀 AUROC(64×64), 위치 적중률(최고점이 결함 위·바로 옆 패치인 비율)."""
    t = torch.as_tensor(maps).float()[:, None]
    m64 = F.interpolate(t, size=(64, 64), mode='bilinear', align_corners=False)[:, 0].numpy()
    gt = np.stack([np.asarray(Image.fromarray(m).resize((64, 64), Image.NEAREST)) for m in masks])
    pix = roc_auc_score(gt.reshape(-1), m64.reshape(-1)) if gt.any() else float('nan')
    h, w = maps.shape[1:]
    hits = []
    for s, m, lab in zip(maps, masks, labels):
        if lab == 1:
            r, c = np.unravel_index(s.argmax(), s.shape)
            hits.append(bool((cell_labels(m) != 0)[r * GRID // h, c * GRID // w]))
    return dict(pixel_auroc=pix, loc_hit=float(np.mean(hits)))


def all_metrics(St, Sv, Sd, labels, types, masks):
    """St·Sv (N, h, w), Sd {D/DR/ND/NR: maps} 또는 None(진단 생략). 집계마다 한 행."""
    base = map_metrics(St, labels, masks)
    y = np.asarray(labels)
    out = {}
    for how in AGGS:
        st, sv = aggregate(St, how), aggregate(Sv, how)
        r = evaluate(st, labels, types, sv)
        thr5 = threshold_at_fpr(st[y == 0], 0.05)
        thr_v = threshold_at_fpr(sv, 0.01)
        out[how] = dict(fnr=r['fnr_at_fpr1'], fnr5=float(np.mean(st[y == 1] <= thr5)), auroc=r['auroc'], **base,
                        **{f'fnr_{t}': r['per_type'][t][1] for t in TYPES}, **{f'n_{t}': r['per_type'][t][0] for t in TYPES},
                        **{k: float(np.mean(aggregate(Sd[k], how) > thr_v)) if Sd else float('nan') for k in DIAG})
    return out


def as_maps(S):
    return S.reshape(len(S), GRID, GRID)


# ---------------- seg: FPN 디코더 ----------------
class Backbone(nn.Module):
    def __init__(self, ex):
        super().__init__()
        self.stem, self.layer2, self.layer3 = ex.stem, ex.layer2, ex.layer3

    @torch.no_grad()
    def forward(self, x):
        f1 = self.stem(x)                 # (B, 256, 128, 128)
        f2 = self.layer2(f1)              # (B, 512, 64, 64)
        return f1, f2, self.layer3(f2)    # (B, 1024, 32, 32)


class SegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1, self.r2, self.r3 = nn.Conv2d(256, 64, 1), nn.Conv2d(512, 128, 1), nn.Conv2d(1024, 128, 1)
        self.c2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU())
        self.c1 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.Dropout2d(0.1),
                                nn.Conv2d(64, 1, 1))

    def forward(self, f1, f2, f3):
        p = self.c2(self.r2(f2) + F.interpolate(self.r3(f3), size=f2.shape[-2:], mode='bilinear', align_corners=False))
        p = torch.cat([F.interpolate(p, size=f1.shape[-2:], mode='bilinear', align_corners=False), self.r1(f1)], 1)
        return self.c1(p)[:, 0]          # (B, 128, 128) logit


class SegModel:
    def __init__(self, bb, ref, dev, seed=0, steps=2000, batch=8, lr=1e-3, aug=False):
        self.bb, self.ref, self.dev = bb, ref, dev
        self.seed, self.steps, self.batch, self.lr, self.aug = seed, steps, batch, lr, aug

    def feats(self, x):
        f = self.bb(to_input(photonorm(x, self.ref)))
        return [(a - m) / s for a, (m, s) in zip(f, self.stats)]

    def fit(self, normals, real, real_masks, gen=None, gen_masks=None):
        dev = self.dev
        torch.manual_seed(self.seed)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        with torch.no_grad():   # 층마다 채널 평균·표준편차 (정상 32장)
            f = self.bb(to_input(photonorm(to_gpu(normals[rng.choice(len(normals), 32, replace=False)], dev), self.ref)))
            self.stats = [(a.mean((0, 2, 3), keepdim=True), a.std((0, 2, 3), keepdim=True) + 1e-4) for a in f]
        self.head = SegHead().to(dev)
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        half = self.batch // 2
        for step in range(self.steps):
            ni = rng.choice(len(normals), half)
            if gen is None:
                src = [(real, real_masks, i) for i in rng.choice(len(real), half)]
            else:   # 불량 절반 중 반은 실제 K장
                src = [(real, real_masks, rng.integers(len(real))) if j % 2 == 0 else (gen, gen_masks, rng.integers(len(gen)))
                       for j in range(half)]
            imgs = np.concatenate([normals[ni], np.stack([im[i] for im, _, i in src])])
            msk = np.concatenate([np.zeros((half, RES, RES), bool), np.stack([mk[i] for _, mk, i in src])])
            x = to_gpu(imgs, dev)
            m = torch.as_tensor(msk, device=dev).float()[:, None]
            flip = torch.rand(2, generator=g, device=dev)
            if flip[0] < 0.5:
                x, m = x.flip(3), m.flip(3)
            if flip[1] < 0.5:
                x, m = x.flip(2), m.flip(2)
            if self.aug:
                x = capture_aug(x, g)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = self.head(*self.feats(x)).float()
            frac = F.avg_pool2d(m, RES // 128)[:, 0]
            pos = (frac > 0.25).float()
            near = F.max_pool2d(F.max_pool2d(m, RES // 128), 5, 1, 2)[:, 0] > 0
            w = (~near | (pos > 0)).float()                          # 결함 테두리 근처는 무시
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

    @torch.no_grad()
    def score(self, arrs, bs=16):
        out = []
        for i in range(0, len(arrs), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(self.head(*self.feats(to_gpu(arrs[i:i + bs], self.dev))).float().cpu().numpy())
        return np.concatenate(out)


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['score', 'aug', 'seg', 'img'])
    ap.add_argument('--aug', default='sd_lora_v3', help='생성 불량 (Dataset/generated 아래 이름)')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--head-seeds', type=int, default=None, help='기본: score·aug 5, seg 3')
    ap.add_argument('--seg-steps', type=int, default=2000)
    ap.add_argument('--seg-aug', action='store_true', help='seg 학습에 capture_aug')
    ap.add_argument('--no-gen', action='store_true', help='생성 불량 없이 실제 K장만 (생성이 없는 분할용)')
    ap.add_argument('--no-diag', action='store_true', help='지름길 진단 생략 (진단 사진 만들기가 PCB당 약 20분)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    hseeds = a.head_seeds or (3 if a.stage == 'seg' else 5)
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ks = KShot(a.k, a.seed, a.data_root)
    ex = PatchFeatures().to(dev)
    gen_masks = {} if a.no_gen else {r['image']: r['mask'] for r in
                                     read_csv(data.generated / a.aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv')}
    rows = []
    for pcb in a.pcbs:
        t0 = time.time()
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real = data.train_defects(pcb, a.k, a.seed)
        gen = [] if a.no_gen else data.generated_defects(a.aug, pcb, a.k, a.seed)
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]

        def add(model, train, hs, res):
            for how, m in res.items():
                rows.append(dict(pcb=pcb, stage=a.stage, model=model, train=train, agg=how, head_seed=hs, **m))

        if a.stage == 'img':   # 비교 기준: 이미지 임베딩 + 헤드
            emb, _ = embedder(data, ex, dev, pcb, RES, 'canon')
            Xt, Xv, Xn, Xr = (emb(x) for x in (test, val, nl, real))
            sets = {'none': Xr} if a.no_gen else {'none': Xr, f'+{a.aug}': np.concatenate([Xr, emb(gen)])}
            for name, Xa in sets.items():
                for kind in ('devnet', 'mlp'):
                    for hs in range(hseeds):
                        h = EmbeddingHead(kind, seed=hs).fit(Xn, Xa)
                        st, sv = h.score(Xt), h.score(Xv)
                        r = evaluate(st, labels, dtypes, sv)
                        y = np.asarray(labels)
                        add(kind, name, hs, {'image': dict(
                            fnr=r['fnr_at_fpr1'], fnr5=float(np.mean(st[y == 1] <= threshold_at_fpr(st[y == 0], 0.05))),
                            auroc=r['auroc'], pixel_auroc=float('nan'), loc_hit=float('nan'),
                            **{f'fnr_{t}': r['per_type'][t][1] for t in TYPES}, **{f'n_{t}': r['per_type'][t][0] for t in TYPES},
                            **{k: float('nan') for k in DIAG})})
            print(f'{pcb}: img 끝 ({time.time() - t0:.0f}s)', flush=True)
            continue
        load = lambda items: np.stack([data.load(x.image, RES) for x in items])  # noqa: E731
        At, Av, An, Ar = (load(x) for x in (test, val, nl, real))
        Ag = load(gen) if gen else None
        Mt = [load_mask(data, x, gen_masks) for x in test]
        Mr = np.stack([load_mask(data, x, gen_masks) for x in real])
        Mg = np.stack([load_mask(data, x, gen_masks) for x in gen]) if gen else None
        diag = None if a.no_diag else diag_set(data, ks, pcb, {x.image for x in real})
        ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()   # patch_ad와 같은 정상 평균색
        print(f'{pcb}: 시험 {len(test)}, 정상 {len(nl)}, 실제 {len(real)}, 생성 {len(gen)}, '
              f'진단 {len(diag["D"]) if diag else 0}장 ({time.time() - t0:.0f}s)', flush=True)

        if a.stage in ('score', 'aug'):
            pf = lambda arr, g=None: patch_feats(ex, arr, ref, dev, g)  # noqa: E731
            Pt, Pv = pf(At), pf(Av)
            Pd = {k: pf(v) for k, v in diag.items()} if diag else None
            g = torch.Generator(device=dev).manual_seed(123) if a.stage == 'aug' else None
            reps = (2, 8, 1) if a.stage == 'aug' else (0, 0, 0)
            Pn = np.concatenate([pf(An)] + [pf(An, g) for _ in range(reps[0])])
            Pr = np.concatenate([pf(Ar)] + [pf(Ar, g) for _ in range(reps[1])])
            Pg = np.concatenate([pf(Ag)] + [pf(Ag, g) for _ in range(reps[2])]) if gen else None
            Lr = np.tile(np.stack([cell_labels(m).reshape(-1) for m in Mr]), (reps[1] + 1, 1))
            Lg = np.tile(np.stack([cell_labels(m).reshape(-1) for m in Mg]), (reps[2] + 1, 1)) if gen else None
            rng = np.random.default_rng(0)
            bank = sample_cells(Pn, np.zeros(Pn.shape[:2], np.int8), 0, 128, rng)
            kn = KnnHead().fit(bank)
            add('knn', 'none', 0, all_metrics(as_maps(kn.score(Pt)), as_maps(kn.score(Pv)),
                                              {k: as_maps(kn.score(v)) for k, v in Pd.items()} if Pd else None,
                                              labels, dtypes, Mt))
            sets = {'none': (Pr, Lr)}
            if gen:
                sets[f'+{a.aug}'] = (np.concatenate([Pr, Pg]), np.concatenate([Lr, Lg]))
            for name, (P, L) in sets.items():
                pos = sample_cells(P, L, 1, None, rng)
                Pneg = np.concatenate([bank, sample_cells(P, L, 0, 128, rng)])
                for kind in ('devnet', 'mlp'):
                    for hs in range(hseeds):
                        h = PatchHead(kind, seed=hs, dev=dev).fit(Pneg, pos)
                        add(kind, name, hs, all_metrics(as_maps(h.score(Pt)), as_maps(h.score(Pv)),
                                                        {k: as_maps(h.score(v)) for k, v in Pd.items()} if Pd else None,
                                                        labels, dtypes, Mt))
        else:
            bb = Backbone(ex).to(dev).eval()
            for name, extra in [('none', ())] + ([(f'+{a.aug}', (Ag, Mg))] if gen else []):
                for hs in range(hseeds):
                    t1 = time.time()
                    sm = SegModel(bb, ref, dev, seed=hs, steps=a.seg_steps, aug=a.seg_aug).fit(An, Ar, Mr, *extra)
                    add('seg', name, hs, all_metrics(sm.score(At), sm.score(Av),
                                                     {k: sm.score(v) for k, v in diag.items()} if diag else None,
                                                     labels, dtypes, Mt))
                    print(f'  {pcb} seg {name} 시드 {hs} ({time.time() - t1:.0f}s)', flush=True)
        for model, train in dict.fromkeys((r['model'], r['train']) for r in rows if r['pcb'] == pcb):
            rs = [r for r in rows if r['pcb'] == pcb and r['model'] == model and r['train'] == train]
            line = ' | '.join(f'{how} {np.mean([r["fnr"] for r in rs if r["agg"] == how]):.3f}'
                              for how in dict.fromkeys(r['agg'] for r in rs))
            print(f'  {pcb} {model:6s} {train:16s} FNR@1 {line}', flush=True)

    out = ROOT / 'AnomalyDetection/results' / f'local_ad_{a.stage}{a.tag}_K{a.k}_seed{a.seed}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wmean(rs, t):
        num = sum(r[f'n_{t}'] * r[f'fnr_{t}'] for r in rs if r[f'n_{t}'])
        return num / max(sum(r[f'n_{t}'] for r in rs if r[f'n_{t}']), 1)

    print(f'\nstage={a.stage}{a.tag}  PCB 평균, 헤드 시드 {hseeds}개 평균. FNR 낮을수록, AUROC·픽셀·위치·ND 높을수록, DR 낮을수록 좋음')
    for model, train in dict.fromkeys((r['model'], r['train']) for r in rows):
        for how in dict.fromkeys(r['agg'] for r in rows):
            rs = [r for r in rows if r['model'] == model and r['train'] == train and r['agg'] == how]
            per = lambda key: np.mean([np.mean([r[key] for r in rs if r['pcb'] == p]) for p in a.pcbs])  # noqa: E731
            print(f'  {model:6s} {train:16s} {how:9s} | FNR@1 {per("fnr"):.3f} @5 {per("fnr5"):.3f} AUROC {per("auroc"):.3f} '
                  f'픽셀 {per("pixel_auroc"):.3f} 위치 {per("loc_hit"):.2f} | '
                  + ' '.join(f'{t} {wmean(rs, t):.2f}' for t in TYPES)
                  + f' | D {per("D"):.2f} DR {per("DR"):.2f} ND {per("ND"):.2f} NR {per("NR"):.2f}', flush=True)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
