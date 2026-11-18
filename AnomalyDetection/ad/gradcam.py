"""모델이 결함을 보는가: 이미지 DevNet Grad-CAM vs 위치 seg 결함 지도 (11/18, 발표 슬라이드 12·14).

모델 (K장 = train_K{k}_seed{s}, 학습은 fair_ad와 같다)
  devnet, mlp  canon 임베딩 + 헤드. 히트맵 = Grad-CAM: 점수를 WRN50 layer3 출력(1024×32×32)으로 미분해
               채널마다 공간 평균한 기울기로 활성을 가중합하고 ReLU. 같은 역전파로 LayerCAM(ReLU(기울기)×활성, 칸마다)도 낸다.
               canon은 기판을 기준 사진에 맞춰 돌리고 옮긴 좌표라, 히트맵을 원래 사진 좌표로 되돌린 뒤 잰다.
  seg          local_ad SegModel 출력(128×128)의 sigmoid. 점수는 fair_ad와 같은 top10 집계.
  학습 = 실제 K장만(none) / +{aug} (--augs)
사진: fair_ad 공정 시험셋(Dataset/cache/fair)의 D(시험 불량) / DR(D의 결함을 정상 픽셀로 지움) / ND(시험 정상 + D의 결함).
     모든 모델에 같은 사진을 쓴다.
지표 (사진마다, 512×512 원래 좌표)
  mass_in   히트맵 질량 중 결함 영역 안 비율. 결함 영역 = 마스크를 원본 해상도에서 40px 넓힘. DR은 지운 자리(D 마스크 넓힘).
  area      결함 영역 면적 비율. 히트맵이 사진 전체에 고르게 퍼지면 mass_in = area.
  top_in    가장 뜨거운 픽셀을 결함 영역 크기만큼 골랐을 때 그중 영역 안 비율 (배경 바닥값·크기와 무관). 무작위면 area.
  peak_in   최고점이 결함 영역 안인가
  spread50  히트맵 질량 절반을 덮는 최소 면적 비율 (작을수록 한곳에 모임)
  flagged   점수가 시험 정상 과검률 1% 임계값을 넘었나 (불량 판정)
예시 그림: PCB마다 공정 시험셋 불량 --n-examples쌍을 rng(--fig-seed)로 무작위로 고른다. 헤드 시드 0.
  열 = 사진 / DevNet 실제만 / DevNet +생성 / seg 실제만 / seg +생성, 행 = D / DR / ND. 흰 선 = 결함 영역, 자홍 원 = 지도 최고점.
  색 범위는 열마다 그 그림에 나온 D 지도들의 최댓값으로 고정한다(Grad-CAM·seg 같은 규칙). 그래서 한 열 안에서 D·DR·ND는
  같은 눈금이고, 열끼리는 눈금이 다르다(점수 단위가 모델마다 다름). 예시 지도는 _maps.npz에 있어 --summarize가 다시 그린다.

    python -m AnomalyDetection.ad.gradcam --pcbs pcb1 --data-root ...      # PCB마다 따로 (병렬)
    python -m AnomalyDetection.ad.gradcam --summarize                      # PCB 결과 합치기 + 3 PCB 예시 그림
결과: AnomalyDetection/results/gradcam/K{k}_seed{s}_{pcb}.csv (사진마다), _{pcb}.jpg, _{pcb}_maps.npz
      AnomalyDetection/results/gradcam_K{k}_seed{s}.csv (요약), gradcam/K{k}_seed{s}_all.jpg
"""
import argparse
import copy
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import PCBS, ROOT, Data, read_csv
from .eval import threshold_at_fpr
from .fair_ad import PX, board_transform, dilate, fair_set  # board_transform: Augmentation/common/kshot
from .features import PatchFeatures, device, embed, to_tensor
from .local_ad import Backbone, SegModel, aggregate
from .models import EmbeddingHead
from .patch_ad import load_mask
from .prep import Canon

RES = 512
PARTS = ('D', 'DR', 'ND')
OUT = ROOT / 'AnomalyDetection/results/gradcam'
LABEL = {'devnet': 'Image DevNet', 'mlp': 'Image MLP', 'seg': 'Location seg'}


# ---------------- 좌표 ----------------
def _open(p):
    with Image.open(p) as im:
        return im.convert('RGB').copy()


def region(mask_full, px=PX):
    """원본 해상도 bool 마스크 → px 넓혀 RES×RES bool."""
    m = dilate(mask_full, px).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(m).resize((RES, RES), Image.NEAREST)) > 127


def uncanon(m, tf, size, ref_size):
    """canon 좌표 지도 (h, w) → 원래 사진 좌표 RES×RES. prep.Canon.align의 역변환
    (align: 180° 돌림 → out(x, y) = im(x − dx, y − dy) → RES로 줄임)."""
    im = Image.fromarray(np.ascontiguousarray(m, np.float32)).resize(ref_size, Image.BILINEAR)
    im = im.transform(size, Image.AFFINE, (1, 0, tf['dx'], 0, 1, tf['dy']), resample=Image.BILINEAR, fillcolor=0.0)
    if tf['rot180']:
        im = im.rotate(180)
    return np.asarray(im.resize((RES, RES), Image.BILINEAR))


# ---------------- 히트맵 ----------------
def cam_maps(ex, head, arrs, dev, bs=8):
    """canon 사진 (N, RES, RES, 3) uint8(fix 뒤) → 점수 (N,), Grad-CAM (N, 32, 32), LayerCAM (N, 32, 32).
    PatchFeatures.forward + embed + 헤드를 그대로 다시 쓰되 layer3 출력에서 기울기를 받는다."""
    net = copy.deepcopy(head.net).to(dev).eval()
    mu, sd = head.mu.to(dev), head.sd.to(dev)
    S, G, L = [], [], []
    for i in range(0, len(arrs), bs):
        x = to_tensor(arrs[i:i + bs]).to(dev)
        with torch.no_grad():
            f2 = ex.layer2(ex.stem(x))
            f3 = ex.layer3(f2)
        f3 = f3.detach().requires_grad_(True)
        p2 = F.avg_pool2d(f2, 3, 1, 1)
        p3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=p2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.einsum('bchw,cd->bdhw', torch.cat([p2, p3], 1), ex.pool)
        e = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        s = net((e - mu) / sd).squeeze(1)
        grad, = torch.autograd.grad(s.sum(), f3)       # 사진끼리 독립(eval)이라 합의 기울기 = 사진별 기울기
        with torch.no_grad():
            G.append(F.relu((grad.mean((2, 3), keepdim=True) * f3).sum(1)).cpu().numpy())
            L.append(F.relu((F.relu(grad) * f3).sum(1)).cpu().numpy())
            S.append(s.cpu().numpy())
    return np.concatenate(S), np.concatenate(G), np.concatenate(L)


def stats(h, reg):
    """h (RES, RES) ≥ 0, reg bool → 사진 하나의 지표."""
    a = float(reg.mean())
    flat = h.ravel().astype(np.float64)
    tot = flat.sum()
    o = np.argsort(-flat, kind='stable')
    n = max(1, int(reg.sum()))
    out = dict(area=a, peak_in=float(reg.ravel()[o[0]]), top_in=float(reg.ravel()[o[:n]].mean()))
    if tot <= 0:
        return dict(out, mass_in=float('nan'), spread50=float('nan'))
    c = np.cumsum(flat[o])
    out.update(mass_in=float(flat[reg.ravel()].sum() / tot), spread50=float((np.searchsorted(c, .5 * tot) + 1) / flat.size))
    return out


# ---------------- 그림 ----------------
def figure(path, ex_maps, title):
    """ex_maps: {'img': {part: (n, RES, RES, 3)}, 'reg': {part: (n, RES, RES)}, 'cols': [(이름, {part: (n, RES, RES)},
    {part: 점수 (n,)}, {part: 판정 (n,)}, vmax)]}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def overlay(h, vmax):   # 낮은 값은 투명하게: 색 = jet, 불투명도 = 값에 비례(최대 0.75)
        v = np.clip(h / vmax, 0, 1)
        rgba = plt.cm.jet(v)
        rgba[..., 3] = .75 * v ** .7
        return rgba

    cols = ex_maps['cols']
    n = len(ex_maps['img']['D'])
    rows = [(j, p) for j in range(n) for p in PARTS]
    fig, axes = plt.subplots(len(rows), 1 + len(cols), figsize=(2.6 * (1 + len(cols)), 2.6 * len(rows)), squeeze=False)
    names = {'D': 'D: test defect', 'DR': 'DR: defect removed', 'ND': 'ND: normal + defect'}
    for r, (j, p) in enumerate(rows):
        img, reg = ex_maps['img'][p][j], ex_maps['reg'][p][j]
        ax = axes[r, 0]
        ax.imshow(img)
        ax.contour(reg, levels=[.5], colors='w', linewidths=.8)
        ax.set_ylabel(f'{ex_maps["names"][j]}\n{names[p]}', fontsize=8)
        for c, (cname, maps, sc, fl, _) in enumerate(cols, 1):
            vmax = max(float(maps['D'].max()), 1e-6)    # 열마다 그림 속 D 지도 최댓값
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(overlay(maps[p][j], vmax))
            ax.contour(reg, levels=[.5], colors='w', linewidths=.8)
            py, px = np.unravel_index(np.argmax(maps[p][j]), maps[p][j].shape)   # 최고점 (peak_in과 같은 점)
            ax.plot(px, py, 'o', ms=11, mfc='none', mec='magenta', mew=1.6)
            ax.set_xlabel(f'score {sc[p][j]:.2f} → {"DEFECT" if fl[p][j] else "normal"}', fontsize=8,
                          color='crimson' if fl[p][j] else 'k')
            if r == 0:
                ax.set_title(cname, fontsize=9, weight='bold')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------- 실행 ----------------
def run_pcb(a, data, pcb, ex, bb, dev, augs, gen_masks):
    t0 = time.time()
    fs = fair_set(data, pcb)
    test, nl = data.test(pcb), data.train_normal(pcb)
    real = data.train_defects(pcb, a.k, a.seed)
    gens = {g: data.generated_defects(g, pcb, a.k, a.seed) for g in augs}
    trow = {r['image']: r for r in read_csv(data.splits / 'test.csv') if r['category'] == pcb}
    Dit = [x for x in test if x.label == 1]
    Nit = [x for x in test if x.label == 0]
    assert [x.image for x in Dit] == list(fs['image']), '공정 시험셋과 test.csv 순서가 다름'
    nd = len(Dit)
    c = Canon(data, pcb, RES)
    ref_size = c.ref.size

    # 결함 영역과 canon 역변환 정보 (D·DR = D 사진 좌표, ND = 바탕 정상 N 좌표)
    reg = {p: [] for p in PARTS}
    geo = {p: [] for p in PARTS}
    for i, x in enumerate(Dit):
        D = _open(data.path(x.image))
        m_D = np.asarray(Image.open(data.path(trow[x.image]['mask'])).convert('L')) > 0
        N = _open(data.path(fs['bg'][i]))
        m_N = np.asarray(Image.fromarray(fs['mask_ND'][i].astype(np.uint8) * 255).resize(N.size, Image.NEAREST)) > 127
        rD, rN = region(m_D), region(m_N)
        tD, tN = board_transform(D, c.ref), board_transform(N, c.ref)
        for p, r_, t_, im in (('D', rD, tD, D), ('DR', rD, tD, D), ('ND', rN, tN, N)):
            reg[p].append(r_)
            geo[p].append((t_, im.size))
    reg = {p: np.stack(v) for p, v in reg.items()}
    print(f'{pcb}: 공정 시험셋 {nd}쌍, 결함 영역 면적 D {reg["D"].mean():.3f} ND {reg["ND"].mean():.3f} '
          f'({time.time() - t0:.0f}s)', flush=True)

    rng = np.random.default_rng([a.fig_seed, PCBS.index(pcb)])
    ex_idx = np.sort(rng.choice(nd, min(a.n_examples, nd), replace=False))
    load = lambda items: np.stack([data.load(x.image, RES) for x in items])  # noqa: E731
    imgs = {'D': load(Dit), 'DR': fs['DR'], 'ND': fs['ND']}
    ex_maps = {'img': {p: imgs[p][ex_idx] for p in PARTS}, 'reg': {p: reg[p][ex_idx] for p in PARTS}, 'cols': [],
               'names': [f'{pcb} #{i} {Dit[i].defect_type}' for i in ex_idx]}
    rows = []

    def record(model, train, hs, kind, maps, scores, thr):
        """maps {part: (nd, RES, RES)} 원래 좌표 히트맵."""
        for p in PARTS:
            for i in range(nd):
                rows.append(dict(pcb=pcb, k=a.k, seed=a.seed, model=model, train=train, head_seed=hs, map=kind, part=p,
                                 idx=i, type=Dit[i].defect_type, score=float(scores[p][i]),
                                 flagged=int(scores[p][i] > thr), **stats(maps[p][i], reg[p][i])))

    # ---- 이미지 단위 (canon 임베딩 + 헤드) ----
    def cemb(arrs):
        out = []
        for i in range(0, len(arrs), 16):
            x = to_tensor(np.stack([c.fix(v) for v in arrs[i:i + 16]])).to(dev)
            with torch.no_grad():
                out.append(embed(ex(x)).float().cpu().numpy())
        return np.concatenate(out)

    item = lambda items: cemb([c.item(x.image) for x in items])  # noqa: E731
    Xn, Xr, XN = item(nl), item(real), item(Nit)
    cimg = {'D': np.stack([c.fix(c.item(x.image)) for x in Dit]),
            'DR': np.stack([c.fix(v) for v in fs['DR_canon']]), 'ND': np.stack([c.fix(v) for v in fs['ND_canon']])}
    train_sets = {'none': Xr, **{f'+{g}': np.concatenate([Xr, item(gens[g])]) for g in augs}}
    print(f'  {pcb} canon 임베딩 끝 ({time.time() - t0:.0f}s)', flush=True)
    for kind in a.img_kinds:
        for name, Xa in train_sets.items():
            for hs in range(a.img_seeds):
                h = EmbeddingHead(kind, seed=hs).fit(Xn, Xa)
                thr = threshold_at_fpr(h.score(XN), .01)
                sc, gc, lc = {}, {}, {}
                for p in PARTS:
                    sc[p], g_, l_ = cam_maps(ex, h, cimg[p], dev)
                    gc[p] = np.stack([uncanon(m, *geo[p][i], ref_size) for i, m in enumerate(g_)])
                    lc[p] = np.stack([uncanon(m, *geo[p][i], ref_size) for i, m in enumerate(l_)])
                if hs == 0:   # 헤드 점수와 다시 쓴 순전파 점수가 같은지 (canon 임베딩 경로 확인)
                    ref_s = h.score(cemb([c.item(Dit[i].image) for i in ex_idx]))
                    assert np.allclose(ref_s, sc['D'][ex_idx], rtol=1e-3, atol=1e-2), (ref_s, sc['D'][ex_idx])
                record(kind, name, hs, 'gradcam', gc, sc, thr)
                record(kind, name, hs, 'layercam', lc, sc, thr)
                if hs == 0 and kind == 'devnet':
                    vmax = float(np.quantile(gc['D'].reshape(nd, -1).max(1), .99)) or 1.0
                    ex_maps['cols'].append((f'{LABEL[kind]} {name}\nGrad-CAM', {p: gc[p][ex_idx] for p in PARTS},
                                            {p: sc[p][ex_idx] for p in PARTS},
                                            {p: sc[p][ex_idx] > thr for p in PARTS}, vmax))
        print(f'  {pcb} {kind} 끝 ({time.time() - t0:.0f}s)', flush=True)

    # ---- 위치 seg ----
    if a.seg_seeds:
        An, Ar = load(nl), load(real)
        ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
        Mr = np.stack([load_mask(data, x, gen_masks) for x in real])
        G = {g: (load(gens[g]), np.stack([load_mask(data, x, gen_masks) for x in gens[g]])) for g in augs}
        AN = load(Nit)
        for name, extra in [('none', ())] + [(f'+{g}', G[g]) for g in augs]:
            for hs in range(a.seg_seeds):
                t1 = time.time()
                sm = SegModel(bb, ref, dev, seed=hs, steps=a.seg_steps).fit(An, Ar, Mr, *extra)
                thr = threshold_at_fpr(aggregate(sm.score(AN), 'top10'), .01)
                sc, mp = {}, {}
                for p in PARTS:
                    lg = sm.score(imgs[p])
                    sc[p] = aggregate(lg, 'top10')
                    pr = torch.sigmoid(torch.as_tensor(lg)[:, None])
                    mp[p] = F.interpolate(pr, size=(RES, RES), mode='bilinear', align_corners=False)[:, 0].numpy()
                record('seg', name, hs, 'seg', mp, sc, thr)
                if hs == 0:
                    ex_maps['cols'].append((f'{LABEL["seg"]} {name}\nmap', {p: mp[p][ex_idx] for p in PARTS},
                                            {p: sc[p][ex_idx] for p in PARTS},
                                            {p: sc[p][ex_idx] > thr for p in PARTS}, 1.0))
                print(f'  {pcb} seg {name} 시드 {hs} ({time.time() - t1:.0f}s)', flush=True)
                del sm
        torch.cuda.empty_cache()

    stem = OUT / f'K{a.k}_seed{a.seed}{a.tag}_{pcb}'
    OUT.mkdir(parents=True, exist_ok=True)
    with open(f'{stem}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure(f'{stem}.jpg', ex_maps, f'{pcb} K={a.k} seed={a.seed}, examples rng({a.fig_seed}, {PCBS.index(pcb)}), '
                                  'number = score, DEFECT = above 1% FPR threshold of test normals')
    np.savez_compressed(f'{stem}_maps.npz', ex_idx=ex_idx, names=np.array(ex_maps['names']),
                        **{f'img_{p}': ex_maps['img'][p] for p in PARTS}, **{f'reg_{p}': ex_maps['reg'][p] for p in PARTS},
                        **{f'col{c}_{k}_{p}': np.asarray(v[p]) for c, col in enumerate(ex_maps['cols'])
                           for k, v in zip(('map', 'score', 'flag'), col[1:4]) for p in PARTS},
                        col_names=np.array([col[0] for col in ex_maps['cols']]),
                        col_vmax=np.array([col[4] for col in ex_maps['cols']]))
    print(f'  {pcb} 저장: {stem}.csv, .jpg ({time.time() - t0:.0f}s)', flush=True)


def summarize(a):
    import pandas as pd
    files = sorted(OUT.glob(f'K{a.k}_seed{a.seed}{a.tag}_pcb?.csv'))
    d = pd.concat([pd.read_csv(f) for f in files])
    print(f'PCB {sorted(d.pcb.unique())}, 사진 {len(d)}행')
    keys = ['model', 'train', 'map', 'part']
    cols = ['mass_in', 'top_in', 'area', 'peak_in', 'spread50', 'flagged']
    g = d.groupby(keys + ['pcb'])[cols].mean().groupby(keys).mean()        # PCB 평균 (헤드 시드·사진 평균 뒤)
    g['n_pcb'] = d.groupby(keys).pcb.nunique()
    out = ROOT / 'AnomalyDetection/results' / f'gradcam_K{a.k}_seed{a.seed}{a.tag}.csv'
    g.reset_index().to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(g.round(3).to_string())
    # DR을 불량이라 한 경우만: 히트맵이 어디 있나
    fl = d[(d.part == 'DR') & (d.flagged == 1)].groupby(['model', 'train', 'map'])[['mass_in', 'top_in', 'area', 'spread50']]
    print('\nDR을 불량으로 판정한 사진만 (사진 평균)')
    print(pd.concat([fl.mean().round(3), fl.size().rename('n')], axis=1).to_string())
    print(f'저장: {out}')
    # 3 PCB 예시(PCB마다 첫 예시) 한 장
    parts = []
    for f in files:
        z = np.load(str(f).replace('.csv', '_maps.npz'))
        parts.append(z)
    for f, z in zip(files, parts):
        ex_maps = {'img': {p: z[f'img_{p}'] for p in PARTS}, 'reg': {p: z[f'reg_{p}'] for p in PARTS},
                   'names': [str(v) for v in z['names']],
                   'cols': [(str(z['col_names'][c]), *({p: z[f'col{c}_{k}_{p}'] for p in PARTS} for k in ('map', 'score', 'flag')),
                             None) for c in range(len(z['col_names']))]}
        pcb = str(f)[-8:-4]
        figure(str(f).replace('.csv', '.jpg'), ex_maps, f'{pcb} K={a.k} seed={a.seed}, examples rng({a.fig_seed}, '
               f'{PCBS.index(pcb)}), score > 1% FPR threshold of test normals = DEFECT')
    if parts:
        ncol = len(parts[0]['col_names'])
        ex_maps = {'img': {p: np.concatenate([z[f'img_{p}'][:1] for z in parts]) for p in PARTS},
                   'reg': {p: np.concatenate([z[f'reg_{p}'][:1] for z in parts]) for p in PARTS},
                   'names': [str(z['names'][0]) for z in parts], 'cols': []}
        for c in range(ncol):
            cat = lambda k: {p: np.concatenate([z[f'col{c}_{k}_{p}'][:1] for z in parts]) for p in PARTS}  # noqa: E731
            vmax = float(np.max([z['col_vmax'][c] for z in parts]))   # PCB마다 다르면 가장 큰 값
            ex_maps['cols'].append((str(parts[0]['col_names'][c]), cat('map'), cat('score'), cat('flag'), vmax))
        path = OUT / f'K{a.k}_seed{a.seed}{a.tag}_all.jpg'
        figure(path, ex_maps, f'K={a.k} seed={a.seed}, first random example of each PCB (fig seed {a.fig_seed})')
        print(f'그림: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--augs', nargs='*', default=['sd_lora_v3'], help='생성 불량 (Dataset/generated 아래 이름). 없으면 건너뜀')
    ap.add_argument('--img-kinds', nargs='+', default=['devnet', 'mlp'], choices=['devnet', 'mlp'])
    ap.add_argument('--img-seeds', type=int, default=3)
    ap.add_argument('--seg-seeds', type=int, default=1)
    ap.add_argument('--seg-steps', type=int, default=2000)
    ap.add_argument('--n-examples', type=int, default=3)
    ap.add_argument('--fig-seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    if a.summarize:
        return summarize(a)
    torch.set_num_threads(a.threads)
    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev).requires_grad_(False)
    bb = Backbone(ex).to(dev).eval()
    augs = []
    for g in a.augs:
        man = data.generated / g / f'K{a.k}_seed{a.seed}' / 'manifest.csv'
        if man.exists() and data.path(read_csv(man)[0]['image']).exists():
            augs.append(g)
        else:
            print(f'생성 없음, 건너뜀: {g} K{a.k}_seed{a.seed}', flush=True)
    gen_masks = {r['image']: r['mask'] for g in augs for r in read_csv(data.generated / g / f'K{a.k}_seed{a.seed}' / 'manifest.csv')}
    for pcb in a.pcbs:
        run_pcb(a, data, pcb, ex, bb, dev, augs, gen_masks)


if __name__ == '__main__':
    main()
