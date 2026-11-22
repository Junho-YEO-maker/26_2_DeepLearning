"""VisA 공식 분할로 공개 논문 수치와 비교한다 (11/20). 분할 파일은 official_splits.py가 만든다.

1단계 --protocol 1cls          정상만 학습(PatchCore). 공개된 PatchCore 수치와 맞는지 = 우리 실험 장치 점검
2단계 --protocol 2cls_fewshot  공식 "불량 조금" 분할(PCB마다 학습 불량 20장 + 정상 약 200장, 시험 불량 80 + 정상 약 800)
                               patchcore(정상만) · devnet(사진 전체 임베딩) · seg · seg_aug(+촬영 조건 무작위화) [+ 생성]
                               공식 지표 + 같은 시험 불량으로 만든 공정 시험셋(fair_ad와 같은 규칙)
지표 (공개 논문과 같은 것)
  image_auroc   사진 점수로 정상·불량을 가르는 AUROC
  pixel_auroc   지도 픽셀 AUROC (EVAL×EVAL에서, 지도는 쌍선형 확대 후 가우시안 σ=4, PatchCore 논문과 같음)
  aupro         결함 덩어리별 겹침(PRO)을 과검 픽셀 비율 0–0.3 구간에서 적분해 0.3으로 나눈 값 (MVTec AD 정의)
  fair_*        fair_ad.metrics (2cls_fewshot만). 공식 분할엔 검증 정상이 없어 v_*의 임계값은 시험 정상으로 잡는다(해석용)
PatchCore 설정 --pc "해상도:자르기:coreset" (예: 512:0:0.01 = 우리 설정, 256:224:0.1 = 256으로 줄이고 가운데 224 자르기, 10%)

    python -m AnomalyDetection.ad.official_eval --protocol 1cls --pc 512:0:0.01 256:224:0.1
    python -m AnomalyDetection.ad.official_eval --protocol 2cls_fewshot --augs sd_lora_official_pinfix
결과: AnomalyDetection/results/official/{protocol}.csv, 이미지별 점수 {protocol}_{pcb}.npz
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from . import fair_ad
from .data import PCBS, ROOT, Data, Item, read_csv
from .features import PatchFeatures, device, embed, to_tensor
from .local_ad import Backbone, SegModel, aggregate
from .models import EmbeddingHead, PatchCore
from .patch_ad import load_mask, mask_path
from .prep import Canon

SPLITS = ROOT / 'Dataset/visa/splits_official'
DATA_ROOT = ROOT / 'Dataset/visa/VisA_official/VisA_PCB'
OUT = ROOT / 'AnomalyDetection/results/official'
RES, EVAL = 512, 256


# ---------------- 지표 ----------------
def _curve(s, pos, w_pos=None):
    """점수 내림차순으로 임계값을 내리며 (과검 비율, 검출 비율) 곡선. 같은 점수는 한 걸음으로 묶는다."""
    o = np.argsort(-s, kind='stable')
    s, pos = s[o], pos[o]
    neg = ~pos
    fp = np.cumsum(neg, dtype=np.float64) / max(neg.sum(), 1)
    tp = np.cumsum(pos if w_pos is None else w_pos[o], dtype=np.float64)
    if w_pos is None:
        tp /= max(pos.sum(), 1)
    keep = np.r_[s[1:] != s[:-1], True]
    return np.r_[0.0, fp[keep]], np.r_[0.0, tp[keep]]


def image_auroc(scores, labels):
    fp, tp = _curve(np.asarray(scores, np.float64), np.asarray(labels) == 1)
    return float(np.trapezoid(tp, fp))


def pixel_metrics(maps, masks, limit=0.3):
    """maps, masks (N, H, W). 픽셀 AUROC와 AUPRO(과검 0–limit)."""
    s = maps.astype(np.float32).ravel()
    gt = masks.ravel().astype(bool)
    fp, tp = _curve(s, gt)
    pix = float(np.trapezoid(tp, fp))
    lab = np.zeros(masks.shape, np.int32)
    n = 0
    for i, m in enumerate(masks):
        l, k = ndimage.label(m)
        lab[i] = np.where(l > 0, l + n, 0)
        n += k
    lab = lab.ravel()
    size = np.bincount(lab, minlength=n + 1).astype(np.float64)
    w = np.where(lab > 0, 1.0 / (size[lab] * max(n, 1)), 0.0)
    fp, pro = _curve(s, gt, w_pos=w)
    i = int(np.searchsorted(fp, limit, side='right'))
    x = np.r_[fp[:i], limit]
    y = np.r_[pro[:i], np.interp(limit, fp, pro)]
    return pix, float(np.trapezoid(y, x) / limit)


def to_eval(maps, size=EVAL, sigma=4):
    """(N, h, w) 지도 → (N, size, size), 쌍선형 확대 + 가우시안."""
    t = F.interpolate(torch.as_tensor(maps, dtype=torch.float32)[:, None], size=(size, size), mode='bilinear',
                      align_corners=False)[:, 0].numpy()
    return np.stack([ndimage.gaussian_filter(m, sigma) for m in t]) if sigma else t


# ---------------- 데이터 ----------------
class CanonOff(Canon):
    """정렬 기준이 공식 분할의 train_normal 첫 장이라 캐시 폴더를 따로 둔다."""

    def __init__(self, data, pcb, protocol, res=RES):
        import hashlib  # noqa: F401
        self.data, self.res = data, res
        nl = data.train_normal(pcb)
        with Image.open(data.path(nl[0].image)) as im:
            self.ref = im.convert('RGB').copy()
        self.cache = ROOT / 'Dataset/cache' / f'canon{res}_official_{protocol}' / pcb
        self.cache.mkdir(parents=True, exist_ok=True)
        A = np.stack([self.item(x.image) for x in nl]).astype(np.float32)
        m, s = A.mean((1, 2)), A.std((1, 2))
        self.mu, self.sd = m.mean(0), s.mean(0)


def load_geom(data, rel, res, crop):
    a = data.load(rel, res)
    if crop:
        o = (res - crop) // 2
        a = a[o:o + crop, o:o + crop]
    return a


def mask_geom(data, item, res, crop, size):
    """GT 마스크를 이미지와 같은 줄이기·자르기 뒤 size×size로."""
    if item.label == 0:
        return np.zeros((size, size), bool)
    with Image.open(mask_path(data, item.image, {})) as im:
        m = np.asarray(im.convert('L').resize((res, res), Image.NEAREST)) > 0
    if crop:
        o = (res - crop) // 2
        m = m[o:o + crop, o:o + crop]
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127


class PatchCoreMap(PatchCore):
    @torch.no_grad()
    def maps(self, arrays, bs=4, chunk=8192):
        out = []
        for i in range(0, len(arrays), bs):
            m = self.ex(to_tensor(np.stack(arrays[i:i + bs])).to(self.dev)).float()
            b, c, h, w = m.shape
            p = m.permute(0, 2, 3, 1).reshape(-1, c)
            d = torch.cat([torch.cdist(p[j:j + chunk], self.bank).min(1).values for j in range(0, len(p), chunk)])
            out.append(d.reshape(b, h, w).cpu())
        return torch.cat(out).numpy()


def train_defects(data, pcb):
    rows = [r for r in read_csv(data.splits / 'train_K20_seed0.csv') if r['category'] == pcb]
    return data._guard([Item(r['image'], 1, r['defect_type'], 'real') for r in rows])


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', required=True, choices=['1cls', '2cls_fewshot'])
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--pc', nargs='*', default=['512:0:0.01'], help='PatchCore 설정 목록 "해상도:자르기:coreset"')
    ap.add_argument('--models', nargs='*', default=['devnet', 'seg', 'seg_aug'], help='2cls_fewshot에서 PatchCore 외 모델')
    ap.add_argument('--augs', nargs='*', default=[], help='seg_aug에 더할 생성 (Dataset/generated/{aug}/K20_seed0)')
    ap.add_argument('--seeds', type=int, default=3, help='devnet·seg 헤드 시드 수')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--tag', default='')
    ap.add_argument('--skip-none', action='store_true', help='생성 없는 seg는 건너뜀 (생성 세트만 추가로 돌릴 때)')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = Data(DATA_ROOT, splits=SPLITS / a.protocol)
    fair = a.protocol == '2cls_fewshot'
    if fair:
        fair_ad.CACHE = ROOT / 'Dataset/cache' / f'fair_official_{a.protocol}'
        for pcb in a.pcbs:
            fair_ad.fair_set(data, pcb)
    dev = device()
    ex = PatchFeatures().to(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pcb in a.pcbs:
        t0 = time.time()
        test, nl = data.test(pcb), data.train_normal(pcb)
        labels = np.array([x.label for x in test])
        types = [x.defect_type for x in test]
        scores = {}
        fs = fair_ad.fair_set(data, pcb) if fair else None
        if fair:
            assert list(fs['image']) == [x.image for x in test if x.label == 1]

        def add(model, train, hs, st, maps=None, gt=None, sf=None):
            r = dict(pcb=pcb, protocol=a.protocol, model=model, train=train, head_seed=hs,
                     image_auroc=image_auroc(st, labels), pixel_auroc=np.nan, aupro=np.nan)
            if maps is not None:
                r['pixel_auroc'], r['aupro'] = pixel_metrics(maps, gt)
            if sf is not None:
                r.update(fair_ad.metrics({'test': st, **sf}, st[labels == 0], labels, types))
            rows.append(r)
            scores[f'{model}|{train}|{hs}|test'] = np.asarray(st, np.float32)
            for p, v in (sf or {}).items():
                scores[f'{model}|{train}|{hs}|{p}'] = np.asarray(v, np.float32)
            print(f'  {pcb} {model} {train} {hs}: 이미지 AUROC {r["image_auroc"]:.3f} 픽셀 {r["pixel_auroc"]:.3f} '
                  f'AUPRO {r["aupro"]:.3f}' + (f' | 공정@5 {r["fair_fnr5"]:.3f} DR {r["v_DR"]:.2f} ND {r["v_ND"]:.2f}'
                                               if sf is not None else ''), flush=True)

        # PatchCore (정상만)
        for cfg in a.pc:
            res, crop, ratio = cfg.split(':')
            res, crop, ratio = int(res), int(crop), float(ratio)
            size = crop or res
            ev = min(EVAL, size)
            An = np.stack([load_geom(data, x.image, res, crop) for x in nl])
            At = np.stack([load_geom(data, x.image, res, crop) for x in test])
            gt = np.stack([mask_geom(data, x, res, crop, ev) for x in test])
            t1 = time.time()
            pc = PatchCoreMap(ex, dev, coreset_ratio=ratio).fit(An)
            mt = pc.maps(At)
            st = mt.reshape(len(mt), -1).max(1)
            sf = None
            if fair:
                prep = (lambda arr: arr) if (res, crop) == (RES, 0) else \
                    (lambda arr: np.stack([load_geom_arr(v, res, crop) for v in arr]))
                sf = {p: pc.maps(prep(fs[p])).reshape(len(fs[p]), -1).max(1) for p in fair_ad.PARTS}
            add('patchcore', cfg, 0, st, to_eval(mt, ev), gt, sf)
            print(f'    ({time.time() - t1:.0f}s, 정상 {len(nl)}장, 시험 {len(test)}장)', flush=True)
            del pc, An, At
            torch.cuda.empty_cache()

        if a.protocol == '1cls':
            np.savez_compressed(OUT / f'{a.protocol}{a.tag}_{pcb}.npz', **scores)
            continue

        real = train_defects(data, pcb)
        load = lambda items: np.stack([data.load(x.image, RES) for x in items])  # noqa: E731
        At, An, Ar = load(test), load(nl), load(real)
        gt = np.stack([mask_geom(data, x, RES, 0, EVAL) for x in test])

        if 'devnet' in a.models:
            c = CanonOff(data, pcb, a.protocol)

            def cemb(arrs):
                out = []
                for i in range(0, len(arrs), 16):
                    x = to_tensor(np.stack([c.fix(v) for v in arrs[i:i + 16]])).to(dev)
                    out.append(embed(ex(x)).float().cpu().numpy())
                return np.concatenate(out)

            item = lambda items: cemb([c.item(x.image) for x in items])  # noqa: E731
            Xt, Xn, Xr = item(test), item(nl), item(real)
            Xf = {p: cemb(fs[p + '_canon']) for p in fair_ad.PARTS}
            for hs in range(a.seeds):
                h = EmbeddingHead('devnet', seed=hs).fit(Xn, Xr)
                add('devnet', 'none', hs, h.score(Xt), sf={p: h.score(Xf[p]) for p in fair_ad.PARTS})

        segs = [m for m in ('seg', 'seg_aug') if m in a.models]
        if segs:
            bb = Backbone(ex).to(dev).eval()
            ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
            Mr = np.stack([load_mask(data, x, {}) for x in real])
            plan = [] if a.skip_none else [(m, 'none', ()) for m in segs]
            for aug in a.augs:
                man = data.generated / aug / 'K20_seed0' / 'manifest.csv'
                if not man.exists():
                    print(f'생성 없음, 건너뜀: {man}', flush=True)
                    continue
                gens = data.generated_defects(aug, pcb, 20, 0)
                gm = {r['image']: r['mask'] for r in read_csv(man)}
                plan.append(('seg_aug', f'+{aug}', (load(gens), np.stack([load_mask(data, x, gm) for x in gens]))))
            for model, name, extra in plan:
                for hs in range(a.seeds):
                    t1 = time.time()
                    sm = SegModel(bb, ref, dev, seed=hs, steps=a.steps, aug=model == 'seg_aug').fit(An, Ar, Mr, *extra)
                    mt = sm.score(At)
                    sf = {p: aggregate(sm.score(fs[p]), 'top10') for p in fair_ad.PARTS}
                    add(model, name, hs, aggregate(mt, 'top10'), to_eval(mt), gt, sf)
                    print(f'    ({time.time() - t1:.0f}s)', flush=True)
                    del sm
            torch.cuda.empty_cache()
        np.savez_compressed(OUT / f'{a.protocol}{a.tag}_{pcb}.npz', **scores)
        print(f'{pcb} 끝 ({time.time() - t0:.0f}s)', flush=True)

    out = OUT / f'{a.protocol}{a.tag}.csv'
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('\nPCB별 (헤드 시드 평균)')
    for key in dict.fromkeys((r['model'], r['train']) for r in rows):
        rs = [r for r in rows if (r['model'], r['train']) == key]
        line = ' '.join(f'{p} {np.mean([r["image_auroc"] for r in rs if r["pcb"] == p]):.3f}/'
                        f'{np.mean([r["pixel_auroc"] for r in rs if r["pcb"] == p]):.3f}/'
                        f'{np.mean([r["aupro"] for r in rs if r["pcb"] == p]):.3f}' for p in a.pcbs)
        extra = (f' | 공정@5 {np.mean([r["fair_fnr5"] for r in rs]):.3f}' if 'fair_fnr5' in rs[0] else '')
        print(f'  {key[0]:9s} {key[1]:28s} 이미지/픽셀/AUPRO {line}{extra}')
    print('저장:', out)


def load_geom_arr(arr, res, crop):
    a = np.asarray(Image.fromarray(arr).resize((res, res), Image.BILINEAR))
    if crop:
        o = (res - crop) // 2
        a = a[o:o + crop, o:o + crop]
    return a


if __name__ == '__main__':
    main()
