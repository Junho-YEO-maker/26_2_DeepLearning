"""공정 시험셋 구성 예시 그림 (11/18, 발표 슬라이드 13).

PCB마다 불량 한 장을 골라 D / DR / N / ND / NR 다섯 장을 결함 주변으로 확대해 나란히 보인다.
  D   시험 불량 (촬영 조건 B)          양성
  DR  D의 결함을 정상 픽셀로 지움 (B)  음성
  N   짝지은 시험 정상 (촬영 조건 A)   음성
  ND  N + D의 결함 (A)                양성
  NR  N + 다른 정상의 같은 자리 (A)    음성, 붙이기 대조
흰 선 = 결함 마스크. 종류는 PCB마다 다르게 정했고(pcb1 bent, pcb2 missing, pcb3 melt), 그 종류 안의 사진은 rng(--seed)로 고른다.
데이터는 ad/fair_ad.py가 만든 Dataset/cache/fair/{pcb}_dil40.npz.

    python -m AnomalyDetection.ad.fair_figure --data-root ...
결과: AnomalyDetection/results/ppt/fair_test_examples.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from .data import ROOT, Data  # noqa: E402
from .fair_ad import CACHE, PX, RES  # noqa: E402

PICK = {'pcb1': 'bent', 'pcb2': 'missing', 'pcb3': 'melt'}
COLS = [('D', 'Test defect\n(capture B)', 1), ('DR', 'Defect erased\n(capture B)', 0),
        ('N', 'Test normal\n(capture A)', 0), ('ND', 'Normal + defect\n(capture A)', 1),
        ('NR', 'Normal + normal patch\n(capture A, paste control)', 0)]
POS, NEG, INK, MUTED = '#D55E00', '#0072B2', '#1a1a1a', '#5a5a5a'


def open512(path, mode='RGB', resample=Image.BILINEAR):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode).resize((RES, RES), resample))


def crop_box(mask, min_w=112, pad=2.2):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    w = max(min_w, pad * max(np.ptp(ys), np.ptp(xs)))
    w = min(w, RES)
    y0 = int(np.clip(cy - w / 2, 0, RES - w))
    x0 = int(np.clip(cx - w / 2, 0, RES - w))
    return y0, x0, int(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-root')
    a = ap.parse_args()
    data = Data(a.data_root)
    rng = np.random.default_rng(a.seed)
    fig, axes = plt.subplots(3, 5, figsize=(13, 8.4))
    for r, (pcb, t) in enumerate(PICK.items()):
        z = np.load(CACHE / f'{pcb}_dil{PX}.npz')
        idx = np.flatnonzero(z['type'] == t)
        i = int(rng.choice(idx))
        rel = str(z['image'][i])
        D = open512(data.path(rel))
        mD = open512(data.path(rel.replace('/Images/', '/Masks/').rsplit('.', 1)[0] + '.png'), 'L', Image.NEAREST) > 0
        N = open512(data.path(str(z['bg'][i])))
        mN = z['mask_ND'][i]
        imgs = {'D': (D, mD), 'DR': (z['DR'][i], mD), 'N': (N, mN), 'ND': (z['ND'][i], mN), 'NR': (z['NR'][i], mN)}
        for c, (key, title, pos) in enumerate(COLS):
            ax = axes[r, c]
            im, m = imgs[key]
            y0, x0, w = crop_box(m)
            ax.imshow(im[y0:y0 + w, x0:x0 + w], interpolation='lanczos')
            edge = m[y0:y0 + w, x0:x0 + w] ^ ndimage.binary_erosion(m[y0:y0 + w, x0:x0 + w], iterations=1)
            yy, xx = np.nonzero(edge)
            ax.scatter(xx, yy, s=1.5, c='white', alpha=0.9, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(POS if pos else NEG)
                sp.set_linewidth(2.5)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK)
                ax.text(0.5, 1.22, 'POSITIVE' if pos else 'negative', transform=ax.transAxes, ha='center',
                        fontsize=9, weight='bold', color=POS if pos else NEG)
            if c == 0:
                ax.set_ylabel(f'{pcb.upper()}  {t}', fontsize=10, color=INK)
        print(f'{pcb} {t}: {rel} (normal {z["bg"][i]})')
    fig.text(0.01, 0.005, 'Crops around the defect at 512x512 model input. White line = defect mask. '
             f'Frame color = label. Fair test: every defect D gets a paired DR / ND / NR (example seed {a.seed}).',
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = ROOT / 'AnomalyDetection/results/ppt/fair_test_examples.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white')
    print('저장:', out)


if __name__ == '__main__':
    main()
