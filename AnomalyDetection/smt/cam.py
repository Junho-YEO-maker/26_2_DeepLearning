"""SMT: 모델이 결함을 보는가 — 사진 단위 DevNet Grad-CAM vs 위치 모델 결함 지도 (11/27). VisA ad/gradcam.py의 SMT판.

모델 (K장 = train_K{k}_seed{s}, 실제 불량만)
  devnet  smt/img_ad.py와 같은 이미지 임베딩 + DevNet 머리(머리 시드 = seed×100). 히트맵 = Grad-CAM (ad/gradcam.cam_maps:
          점수를 WRN50 layer3 출력 32×32로 미분해 채널 평균 기울기로 활성을 가중합, ReLU)
  seg     smt/seg_ad.py MixSeg (2000 step) 출력 128×128의 sigmoid. 점수 = top10 집계
사진 (모두 512×512, SMT 원본 그대로)
  test  시험셋 불량 전체 (결함 영역 = 불량 상자를 8px 넓힘)
  D / DR / ND  seg_ad.diag_images의 pool 불량 중 학습 K장을 뺀 것 / 그 결함을 정상 픽셀로 지운 것 / 정상 사진 + 그 결함
지표 (사진마다, ad/gradcam.stats)
  mass_in  히트맵 질량 중 결함 영역 안 비율 (히트맵이 고르게 퍼지면 area와 같음)
  top_in   가장 뜨거운 픽셀을 결함 영역 크기만큼 골랐을 때 그중 영역 안 비율 (무작위면 area)
  peak_in  최고점이 결함 영역 안인가,  area  결함 영역 면적 비율
  flagged  점수가 시험 정상 과검률 1% 임계값을 넘었나

    python -m AnomalyDetection.smt.cam --category pre --k 1 --seed 0
결과: AnomalyDetection/results/smt/cam_K{k}_seed{s}_{category}.csv (사진마다, 지표만)
      Dataset/aihub_smt/cam/K{k}_seed{s}_{category}_types.jpg (유형별 그림), _DR_ND.jpg (지운 · 붙인 사진 예시), _types.npz
      (AIHub 사진이 들어 있어 git 제외)
    python -m AnomalyDetection.smt.cam --category pre --k 1 --seed 0 --redraw   # 학습 없이 유형별 그림만 다시
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..ad.data import ROOT, read_csv
from ..ad.eval import threshold_at_fpr
from ..ad.features import PatchFeatures, device
from ..ad.gradcam import cam_maps, figure, stats
from ..ad.local_ad import Backbone, aggregate
from ..ad.models import EmbeddingHead
from .data import SPLITS, SmtData
from .img_ad import embeddings
from .seg_ad import OUT_DIR, RES, MixSeg, defect_mask, diag_images

FIG_DIR = ROOT / 'Dataset/aihub_smt/cam'


def up(maps):
    """(N, h, w) → (N, RES, RES) 쌍선형."""
    return F.interpolate(torch.as_tensor(maps, dtype=torch.float32)[:, None], size=(RES, RES), mode='bilinear',
                         align_corners=False)[:, 0].numpy()


TYPE_EN = {'nm': 'no solder', 'nb': 'insufficient', 'ns': 'bridge', 'pb': 'solder ball', 'om': 'offset', 'mf': 'bad shape',
           'cc': 'cold joint', 'ml': 'shifted', 'sh': 'short', 'ws': 'wrong part', 'ms': 'missing part', 'rs': 'reversed',
           'fl': 'flipped', 'st': 'tombstone', 'ph': 'pinhole', 'ic': 'icicle'}


def type_figure(path, names, imgs, regs, cols, title, per_row=2):
    """유형마다 [사진 + 결함 영역 | DevNet Grad-CAM | 위치 모델 지도]. 지도는 사진마다 자기 최댓값으로 나눠, 어디를 보는지만 비교한다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(names)
    nr = (n + per_row - 1) // per_row
    fig, axes = plt.subplots(nr, 3 * per_row, figsize=(2.3 * 3 * per_row, 2.55 * nr), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for i in range(n):
        r, c0 = i // per_row, (i % per_row) * 3
        ax = axes[r, c0]
        ax.imshow(imgs[i])
        ax.contour(regs[i], levels=[.5], colors='w', linewidths=1.2)
        ax.set_title(f'{names[i]} ({TYPE_EN.get(names[i], "")})', fontsize=9, weight='bold')
        for j, (m, maps, sc, fl) in enumerate(cols, 1):
            h = maps[i] / max(float(maps[i].max()), 1e-6)
            rgba = plt.cm.jet(h)
            rgba[..., 3] = .7 * h ** .8
            ax = axes[r, c0 + j]
            ax.imshow(imgs[i])
            ax.imshow(rgba)
            ax.contour(regs[i], levels=[.5], colors='w', linewidths=1.2)
            py, px = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
            ax.plot(px, py, 'o', ms=10, mfc='none', mec='magenta', mew=1.6)
            ax.set_title(('DevNet Grad-CAM' if m == 'devnet' else 'location model'), fontsize=8)
            ax.text(.5, .03, f'{"DEFECT" if fl[i] else "normal"} ({sc[i]:.2f})', transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8, color='crimson' if fl[i] else 'k', bbox=dict(boxstyle='round,pad=.2', fc='w', ec='none', alpha=.85))
    fig.suptitle(title + '  |  white line = defect box, magenta = map peak', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def redraw_types(cat, k, seed, load):
    stem = f'K{k}_seed{seed}_{cat}_types'
    z = np.load(FIG_DIR / f'{stem}.npz')
    tsel = [int(i) for i in z['idx']]
    defects = [r for r in read_csv(SPLITS / 'test.csv') if r['category'] == cat and r['label'] == '1']
    res = [r for r in read_csv(OUT_DIR / f'cam_K{k}_seed{seed}_{cat}.csv') if r['part'] == 'test']
    cols = []
    for m in ('devnet', 'seg'):
        by = {int(r['idx']): r for r in res if r['model'] == m}
        cols.append((m, z[m].astype(np.float32), np.array([float(by[i]['score']) for i in tsel]),
                     np.array([by[i]['flagged'] == '1' for i in tsel])))
    type_figure(FIG_DIR / f'{stem}.jpg', [defects[i]['defect_type'] for i in tsel], np.stack([load(defects[i]['image']) for i in tsel]),
                np.stack([defect_mask(defects[i], cat, dilate=8) for i in tsel]), cols,
                f'SMT {cat}, K={k}, seed {seed}: test defects, one per type')
    print('그림:', (FIG_DIR / f'{stem}.jpg').relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--category', required=True, choices=['pre', 'solder'])
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-examples', type=int, default=4)
    ap.add_argument('--redraw', action='store_true', help='학습 없이 저장된 _types.npz와 csv로 유형별 그림만 다시 그림')
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev, cat, k, seed = device(), a.category, a.k, a.seed
    D = SmtData('none')
    load = lambda rel: D.load(rel, RES)  # noqa: E731
    if a.redraw:
        redraw_types(cat, k, seed, load)
        return
    t0 = time.time()
    An = np.stack([load(it.image) for it in D.train_normal(cat)])
    test_rows = [r for r in read_csv(SPLITS / 'test.csv') if r['category'] == cat]
    At = np.stack([load(r['image']) for r in test_rows])
    yt = np.array([int(r['label']) for r in test_rows])
    pool, diag = diag_images(D, cat, load)
    pool_rows = [r for r in read_csv(SPLITS / 'pool.csv') if r['category'] == cat]
    assert [r['image'] for r in pool_rows] == pool
    kshot = [r for r in read_csv(SPLITS / f'train_K{k}_seed{seed}.csv') if r['category'] == cat]
    held = np.array([p not in {r['image'] for r in kshot} for p in pool])
    Ar = np.stack([load(r['image']) for r in kshot])
    Mr = np.stack([defect_mask(r, cat) for r in kshot])
    reg = {'test': np.stack([defect_mask(r, cat, dilate=8) for r in test_rows if r['label'] == '1'])}
    preg = np.stack([defect_mask(r, cat, dilate=8) for r in pool_rows])[held]
    imgs = {'test': At[yt == 1], **{p: diag[p][held] for p in ('D', 'DR', 'ND')}}
    for p in ('D', 'DR', 'ND'):
        reg[p] = preg
    types = {'test': [r['defect_type'] for r in test_rows if r['label'] == '1'],
             **{p: [r['defect_type'] for r, h in zip(pool_rows, held) if h] for p in ('D', 'DR', 'ND')}}
    print(f'{cat} K={k} seed={seed}: 데이터 {time.time() - t0:.0f}s (시험 불량 {len(imgs["test"])}, pool {held.sum()})', flush=True)

    # DevNet + Grad-CAM
    ex = PatchFeatures().to(dev)
    head = EmbeddingHead('devnet', seed=seed * 100).fit(embeddings(ex, dev, list(An)), embeddings(ex, dev, list(Ar)))
    thr = {'devnet': threshold_at_fpr(head.score(embeddings(ex, dev, list(At[yt == 0]))), 0.01)}
    maps, scores = {'devnet': {}, 'seg': {}}, {'devnet': {}, 'seg': {}}
    for p, x in imgs.items():
        s, g, _ = cam_maps(ex, head, x, dev)
        maps['devnet'][p], scores['devnet'][p] = up(g), s
    print(f'  DevNet Grad-CAM {time.time() - t0:.0f}s', flush=True)
    # 위치 모델
    bb = Backbone(PatchFeatures().to(dev)).to(dev).eval()
    ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
    seg = MixSeg(bb, ref, dev, seed=seed, steps=2000).fit(An, Ar, Mr)
    thr['seg'] = threshold_at_fpr(aggregate(seg.score(At[yt == 0]), 'top10'), 0.01)
    for p, x in imgs.items():
        raw = seg.score(x)
        maps['seg'][p], scores['seg'][p] = up(1 / (1 + np.exp(-raw))), aggregate(raw, 'top10')
    print(f'  위치 모델 {time.time() - t0:.0f}s', flush=True)

    out = OUT_DIR / f'cam_K{k}_seed{seed}_{cat}.csv'
    rows = []
    for m in maps:
        for p in imgs:
            for i in range(len(imgs[p])):
                st = stats(maps[m][p][i], reg[p][i])
                rows.append(dict(category=cat, K=k, seed=seed, model=m, part=p, idx=i, defect_type=types[p][i],
                                 score=float(scores[m][p][i]), flagged=int(scores[m][p][i] > thr[m]), **st))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in maps:
        for p in imgs:
            x = [r for r in rows if r['model'] == m and r['part'] == p]
            print(f'  {m:6s} {p:4s} n={len(x):3d} mass_in={np.nanmean([r["mass_in"] for r in x]):.2f} '
                  f'top_in={np.mean([r["top_in"] for r in x]):.2f} peak_in={np.mean([r["peak_in"] for r in x]):.2f} '
                  f'area={np.mean([r["area"] for r in x]):.2f} flagged={np.mean([r["flagged"] for r in x]):.2f}', flush=True)

    # 유형별 그림: 시험 불량에서 유형마다 1장 (그 유형 하나만, 상자 1개 우선, rng 0). 지도는 사진마다 자기 최댓값으로 맞춘다
    rng = np.random.default_rng(0)
    tsel = []
    for t_ in sorted({x for x in types['test'] if '|' not in x}):
        idx = [i for i, x in enumerate(types['test']) if x == t_]
        tsel.append(int(rng.choice(idx)))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    type_figure(FIG_DIR / f'K{k}_seed{seed}_{cat}_types.jpg', [types['test'][i] for i in tsel], imgs['test'][tsel], reg['test'][tsel],
                [(m, maps[m]['test'][tsel], scores[m]['test'][tsel], scores[m]['test'][tsel] > thr[m]) for m in ('devnet', 'seg')],
                f'SMT {cat}, K={k}, seed {seed}: test defects, one per type')
    np.savez_compressed(FIG_DIR / f'K{k}_seed{seed}_{cat}_types.npz', idx=np.array(tsel),
                        devnet=maps['devnet']['test'][tsel].astype(np.float16), seg=maps['seg']['test'][tsel].astype(np.float16))

    # 예시 그림: pool 불량 중 서로 다른 유형에서 n장 (rng 0)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(types['D']))
    pick, seen = [], set()
    for i in order:
        if types['D'][i] not in seen and '|' not in types['D'][i]:
            pick.append(i)
            seen.add(types['D'][i])
        if len(pick) == a.n_examples:
            break
    pick = np.array(pick)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = [(name, {p: maps[m][p][pick] for p in ('D', 'DR', 'ND')}, {p: scores[m][p][pick] for p in ('D', 'DR', 'ND')},
             {p: scores[m][p][pick] > thr[m] for p in ('D', 'DR', 'ND')}, None)
            for name, m in (('image DevNet (Grad-CAM)', 'devnet'), ('location model (seg map)', 'seg'))]
    ex_maps = {'img': {p: imgs[p][pick] for p in ('D', 'DR', 'ND')}, 'reg': {p: reg[p][pick] for p in ('D', 'DR', 'ND')},
               'cols': cols, 'names': [types['D'][i] for i in pick]}
    figure(FIG_DIR / f'K{k}_seed{seed}_{cat}_DR_ND.jpg', ex_maps, f'SMT {cat}, K={k}, seed {seed}: where does the model look?')
    print('  결과:', out.relative_to(ROOT), '| 그림:', (FIG_DIR / f'K{k}_seed{seed}_{cat}_types.jpg').relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
