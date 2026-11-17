"""지름길 해결 B: 패치 단위 AD (11/16).

이미지 임베딩 하나로 판정하면 불량 사진의 촬영 조건(지름길)만으로 점수를 올릴 수 있다(defect_removal_check).
여기서는 패치마다 판정하고, 마스크로 패치 라벨을 준다.
  특징   PatchFeatures(512px 입력, 64×64×1024)를 2×2 평균으로 32×32 격자로 (패치 = 원본의 약 1/32 × 1/32)
  라벨   결함 마스크가 패치 면적의 5% 넘게 덮으면 불량, 결함에 닿거나 바로 옆(1칸)이면 무시, 나머지는 정상.
         불량 사진(실제 K장, 생성)의 결함 없는 패치도 "정상"으로 학습한다 → 사진 전체 분위기로는 점수를 못 올린다.
  헤드   DevNet(deviation loss) / MLP(BCE), 패치 1024차원 → 256 → 1, GPU. 정상 패치 기준으로 표준화.
         배치 = 정상 패치 절반 + 불량 패치 절반(복원 추출). 이미지 점수 = 패치 점수 최댓값.
  기준선 knn: 정상 패치 은행까지의 최근접 거리(불량 없이, PatchCore 방식)
평가 (원래 시험셋 그대로)
  FNR@FPR1%(이미지), AUROC(이미지), 픽셀 AUROC(64×64), 위치 적중률(불량 이미지에서 최고 점수 패치가 결함 위·바로 옆인 비율)
  위치 적중률·픽셀 AUROC는 결함 자리를 찾아야만 오르는 지표라 촬영 조건 지름길로는 올릴 수 없다.
전처리는 photonorm(채널 평균 맞춤, 기하 변환 없음 → 마스크가 그대로 맞는다).

    python -m AnomalyDetection.ad.patch_ad --augs sd_lora_v3 --k 1 --seed 0 [--data-root ...]
결과: AnomalyDetection/results/patch_ad_K{k}_seed{s}.csv
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from .data import PCBS, ROOT, TYPES, Data, read_csv
from .eval import evaluate
from .features import PatchFeatures, device, to_tensor

GRID = 32          # 패치 격자
RES = 512
POS_FRAC = 0.05


def mask_path(data, rel, gen_masks):
    if rel in gen_masks:
        return ROOT / gen_masks[rel]
    return data.path(rel.replace('/Images/', '/Masks/').rsplit('.', 1)[0] + '.png')


def load_mask(data, item, gen_masks):
    """res×res bool (정상이면 전부 False)."""
    if item.label == 0:
        return np.zeros((RES, RES), bool)
    with Image.open(mask_path(data, item.image, gen_masks)) as im:
        return np.asarray(im.convert('L').resize((RES, RES), Image.NEAREST)) > 0


def cell_labels(mask):
    """GRID×GRID: 1 불량, 0 정상, -1 무시."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    frac = F.avg_pool2d(t, RES // GRID)[0, 0]
    touch = F.max_pool2d(t, RES // GRID)
    near = F.max_pool2d(touch, 3, 1, 1)[0, 0] > 0
    lab = torch.zeros(GRID, GRID, dtype=torch.int8)
    lab[near] = -1
    lab[frac > POS_FRAC] = 1
    return lab.numpy()


@torch.no_grad()
def patch_feats(data, ex, dev, items, ref, bs=16):
    """(N, GRID*GRID, 1024) float16, photonorm."""
    out = []
    for i in range(0, len(items), bs):
        arr = []
        for it in items[i:i + bs]:
            a = data.load(it.image, RES).astype(np.float32)
            arr.append(np.clip(a - a.mean((0, 1)) + ref, 0, 255).astype(np.uint8))
        f = ex(to_tensor(np.stack(arr)).to(dev))                     # (B, 1024, 64, 64)
        f = F.avg_pool2d(f, f.shape[-1] // GRID)                     # (B, 1024, 32, 32)
        out.append(f.flatten(2).transpose(1, 2).half().cpu().numpy())
    return np.concatenate(out)


class PatchHead:
    def __init__(self, kind, seed=0, steps=2000, batch=256, lr=1e-3, dev='cuda'):
        self.kind, self.seed, self.steps, self.batch, self.lr, self.dev = kind, seed, steps, batch, lr, dev

    def fit(self, Pn, Pa):
        dev = self.dev
        Pn = torch.as_tensor(Pn, device=dev).float()
        Pa = torch.as_tensor(Pa, device=dev).float()
        self.mu, self.sd = Pn.mean(0), Pn.std(0) + 1e-6
        Zn, Za = (Pn - self.mu) / self.sd, (Pa - self.mu) / self.sd
        g = torch.Generator(device=dev).manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(Zn.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(256, 1)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        half = self.batch // 2
        y = torch.cat([torch.zeros(half, device=dev), torch.ones(half, device=dev)])
        for _ in range(self.steps):
            x = torch.cat([Zn[torch.randint(len(Zn), (half,), generator=g, device=dev)],
                           Za[torch.randint(len(Za), (half,), generator=g, device=dev)]])
            out = self.net(x).squeeze(1)
            if self.kind == 'mlp':
                loss = F.binary_cross_entropy_with_logits(out, y)
            else:
                ref = torch.randn(5000, generator=g, device=dev)
                d = (out - ref.mean()) / ref.std()
                loss = ((1 - y) * d.abs() + y * F.relu(5.0 - d)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, P):
        """P (N, cells, 1024) → (N, cells) 패치 점수."""
        out = []
        for i in range(0, len(P), 64):
            z = (torch.as_tensor(P[i:i + 64], device=self.dev).float() - self.mu) / self.sd
            out.append(self.net(z).squeeze(-1).cpu().numpy())
        return np.concatenate(out)


class KnnHead:
    def __init__(self, bank=20000, seed=0, dev='cuda'):
        self.bank, self.seed, self.dev = bank, seed, dev

    def fit(self, Pn, Pa=None):
        rng = np.random.default_rng(self.seed)
        sel = rng.choice(len(Pn), min(self.bank, len(Pn)), replace=False)
        self.B = torch.as_tensor(Pn[sel], device=self.dev).float()
        return self

    @torch.no_grad()
    def score(self, P):
        out = []
        for i in range(len(P)):
            q = torch.as_tensor(P[i], device=self.dev).float()
            out.append(torch.cdist(q, self.B).min(1).values.cpu().numpy())
        return np.stack(out)


def sample_cells(P, L, want, n_per, rng):
    """P (N, cells, d), L (N, cells) 라벨에서 라벨 == want인 패치. n_per가 있으면 이미지당 최대 n_per개."""
    out = []
    for p, l in zip(P, L):
        idx = np.where(l == want)[0]
        if n_per and len(idx) > n_per:
            idx = rng.choice(idx, n_per, replace=False)
        out.append(p[idx])
    return np.concatenate(out) if out else np.zeros((0, P.shape[-1]), P.dtype)


def metrics(S, labels, types, Sv, masks):
    """S (N, cells) 패치 점수 → 이미지·픽셀·위치 지표."""
    img, img_v = S.max(1), Sv.max(1)
    r = evaluate(img, labels, types, img_v)
    maps = torch.as_tensor(S.reshape(-1, 1, GRID, GRID)).float()
    maps = F.interpolate(maps, size=(64, 64), mode='bilinear', align_corners=False)[:, 0].numpy()
    gt = np.stack([np.asarray(Image.fromarray(m).resize((64, 64), Image.NEAREST)) for m in masks])
    pix = roc_auc_score(gt.reshape(-1), maps.reshape(-1)) if gt.any() else float('nan')
    hits = []
    for s, m, lab in zip(S, masks, labels):
        if lab == 1:
            near = cell_labels(m) != 0
            hits.append(bool(near.reshape(-1)[s.argmax()]))
    return dict(fnr=r['fnr_at_fpr1'], auroc=r['auroc'], pixel_auroc=pix, loc_hit=float(np.mean(hits)),
                **{f'fnr_{t}': r['per_type'][t][1] for t in TYPES}, **{f'n_{t}': r['per_type'][t][0] for t in TYPES})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--augs', nargs='+', default=['sd_lora_v3'])
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--normal-per-image', type=int, default=128, help='정상 이미지당 뽑을 정상 패치 수')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    rows = []
    for pcb in a.pcbs:
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real = data.train_defects(pcb, a.k, a.seed)
        ref = np.mean([data.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl], 0)
        gen_masks, gens = {}, {}
        for aug in a.augs:
            for r in read_csv(data.generated / aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv'):
                gen_masks[r['image']] = r['mask']
            gens[aug] = data.generated_defects(aug, pcb, a.k, a.seed)
        Pt, Pv, Pn, Pr = (patch_feats(data, ex, dev, x, ref) for x in (test, val, nl, real))
        test_masks = [load_mask(data, x, gen_masks) for x in test]
        Lr = np.stack([cell_labels(load_mask(data, x, gen_masks)).reshape(-1) for x in real])
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]
        rng = np.random.default_rng(0)
        normal_bank = sample_cells(Pn, np.zeros(Pn.shape[:2], np.int8), 0, a.normal_per_image, rng)
        print(f'{pcb}: 정상 패치 {len(normal_bank)}, 실제 K장 불량 패치 {(Lr == 1).sum()}', flush=True)

        # knn 기준선 (불량 학습 없음)
        S, Sv = KnnHead().fit(normal_bank).score(Pt), KnnHead().fit(normal_bank).score(Pv)
        rows.append(dict(pcb=pcb, model='knn', train='none', head_seed=0, n_pos=0, **metrics(S, labels, dtypes, Sv, test_masks)))

        sets = {'none': (Pr, Lr)}
        for aug, gen in gens.items():
            Pg = patch_feats(data, ex, dev, gen, ref)
            Lg = np.stack([cell_labels(load_mask(data, x, gen_masks)).reshape(-1) for x in gen])
            sets[f'+{aug}'] = (np.concatenate([Pr, Pg]), np.concatenate([Lr, Lg]))
        for name, (Pd, Ld) in sets.items():
            pos = sample_cells(Pd, Ld, 1, None, rng)
            neg_in_defect = sample_cells(Pd, Ld, 0, a.normal_per_image, rng)   # 불량 사진의 결함 없는 패치 = 정상
            Pnorm = np.concatenate([normal_bank, neg_in_defect])
            for kind in ('devnet', 'mlp'):
                for hs in range(a.head_seeds):
                    h = PatchHead(kind, seed=hs, steps=a.steps, dev=dev).fit(Pnorm, pos)
                    m = metrics(h.score(Pt), labels, dtypes, h.score(Pv), test_masks)
                    rows.append(dict(pcb=pcb, model=kind, train=name, head_seed=hs, n_pos=len(pos), **m))
                ms = [r for r in rows if r['pcb'] == pcb and r['model'] == kind and r['train'] == name]
                print(f'  {pcb} {kind:6s} {name:18s} 불량 패치 {len(pos):5d} | FNR {np.mean([r["fnr"] for r in ms]):.3f} '
                      f'AUROC {np.mean([r["auroc"] for r in ms]):.3f} 픽셀AUROC {np.mean([r["pixel_auroc"] for r in ms]):.3f} '
                      f'위치적중 {np.mean([r["loc_hit"] for r in ms]):.2f}', flush=True)
        kn = [r for r in rows if r['pcb'] == pcb and r['model'] == 'knn'][0]
        print(f'  {pcb} knn    (불량 학습 없음)      | FNR {kn["fnr"]:.3f} AUROC {kn["auroc"]:.3f} 픽셀AUROC {kn["pixel_auroc"]:.3f} '
              f'위치적중 {kn["loc_hit"]:.2f}', flush=True)

    out = ROOT / 'AnomalyDetection/results' / f'patch_ad_K{a.k}_seed{a.seed}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wmean(rs, t):
        num = sum(r[f'n_{t}'] * r[f'fnr_{t}'] for r in rs if r[f'n_{t}'])
        return num / max(sum(r[f'n_{t}'] for r in rs if r[f'n_{t}']), 1)

    print(f'\n패치 AD 요약 (PCB 평균, 헤드 시드 {a.head_seeds}개 평균). FNR 낮을수록, 나머지는 높을수록 좋음. 종류별 FNR은 PCB 합침')
    combos = [('knn', 'none')] + [(k, n) for k in ('devnet', 'mlp') for n in ['none'] + [f'+{g}' for g in a.augs]]
    for kind, name in combos:
        rs = [r for r in rows if r['model'] == kind and r['train'] == name]
        per = lambda key: np.mean([np.mean([r[key] for r in rs if r['pcb'] == p]) for p in a.pcbs])  # noqa: E731
        print(f'  {kind:6s} {name:18s} FNR {per("fnr"):.3f} AUROC {per("auroc"):.3f} 픽셀AUROC {per("pixel_auroc"):.3f} '
              f'위치적중 {per("loc_hit"):.2f} | ' + ' '.join(f'{t} {wmean(rs, t):.2f}' for t in TYPES))
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
