"""발표용 VisA "모델이 본 곳" 세 장: [사진 | 사진 단위 DevNet Grad-CAM | 위치 모델 지도] (본편 10 · 17번).

VisA 결함은 작아서(결함 영역이 사진의 1–3%) 위치 모델 지도가 결함 자리 한 점에 모인다. 사진 전체로 보면 그 점이
거의 안 보여서, 오른쪽 아래에 결함 주변을 크게 확대한 칸을 붙인다(세 장 모두 같은 자리 · 같은 배율, 사진에 흰 사각형으로 표시).
지도 색은 ad/gradcam.figure와 같다: jet, 불투명도 = 값에 비례, 눈금 = 그 지도의 최댓값. 자홍 원 = 지도에서 가장 높은 곳.

입력: ad/gradcam.py 결과의 예시 지도 (AnomalyDetection/results/gradcam/K1_seed0_r2_pcb1_maps.npz, 첫 예시 = PCB1 #30 녹은 납)
    python -m AnomalyDetection.ad.gradcam --pcbs pcb1 --tag _r2 --data-root Dataset/visa/VisA_official/VisA_PCB
    python ppt_build/make_visa_cam_panels.py
출력: ppt_figures/s17_visa_cam_{photo,devnet,seg}.jpg (VisA는 CC BY 4.0이라 git에 넣는다)
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / 'AnomalyDetection/results/gradcam/K1_seed0_r2_pcb1_maps.npz'
OUT = ROOT / 'ppt_figures'
COLS = {'devnet': 'Image DevNet none\nGrad-CAM', 'seg': 'Location seg none\nmap'}
WIN, INSET = 96, 0.46          # 확대 창 (512 좌표 픽셀), 확대 칸 크기 (사진 한 변 대비)


def overlay(h, vmax):
    v = np.clip(h / vmax, 0, 1)
    rgba = plt.cm.jet(v)
    rgba[..., 3] = .75 * v ** .7
    return rgba


def panel(path, img, reg, m=None):
    n = img.shape[0]
    ys, xs = np.nonzero(reg)
    cy, cx = int(ys.mean()), int(xs.mean())
    x0, y0 = int(np.clip(cx - WIN // 2, 0, n - WIN)), int(np.clip(cy - WIN // 2, 0, n - WIN))
    fig = plt.figure(figsize=(6, 6), dpi=128)
    ax = fig.add_axes((0, 0, 1, 1))
    ins = fig.add_axes((1 - INSET - .02, .02, INSET, INSET))
    for a, (xl, yl) in ((ax, ((-.5, n - .5), (n - .5, -.5))), (ins, ((x0 - .5, x0 + WIN - .5), (y0 + WIN - .5, y0 - .5)))):
        a.imshow(img)
        if m is not None:
            a.imshow(overlay(m, max(float(m.max()), 1e-6)))
        a.contour(reg, levels=[.5], colors='w', linewidths=1.6)
        if m is not None:
            py, px = np.unravel_index(np.argmax(m), m.shape)
            a.plot(px, py, 'o', ms=16 if a is ax else 30, mfc='none', mec='magenta', mew=2.6)
        a.set_xlim(*xl)
        a.set_ylim(*yl)
        a.set_xticks([])
        a.set_yticks([])
    ax.add_patch(Rectangle((x0, y0), WIN, WIN, fill=False, ec='w', lw=1.8, ls='--'))
    for s in ins.spines.values():
        s.set_edgecolor('w')
        s.set_linewidth(3)
    fig.savefig(path, dpi=128, pil_kwargs={'quality': 92})
    plt.close(fig)


def main():
    z = np.load(NPZ)
    names = [str(c) for c in z['col_names']]
    img, reg = z['img_D'][0], z['reg_D'][0]
    print('예시:', str(z['names'][0]))
    panel(OUT / 's17_visa_cam_photo.jpg', img, reg)
    for key, col in COLS.items():
        c = names.index(col)
        panel(OUT / f's17_visa_cam_{key}.jpg', img, reg, z[f'col{c}_map_D'][0].astype(np.float32))
    print('저장:', ', '.join(f's17_visa_cam_{k}.jpg' for k in ('photo', 'devnet', 'seg')), '→', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
