"""canon 전처리 뒤에도 남는 촬영 조건 차이 측정 (11/16).

canon(기판 정렬 + 채널 평균·표준편차 맞춤) 뒤에도 결함을 지운 불량 사진을 DevNet이 불량으로 본다(pcb1 0.56, pcb2 0.61, pcb3 0.14).
무엇이 남았는지 사진 통계로 본다. train_normal 500장 vs pool 불량 사진(약 40장, 시험셋 아님).
  illum_x, illum_y  큰 규모 조명 기울기(16×16 평균 밝기에 평면을 맞춘 기울기, 평균 밝기로 나눔)
  sharp             선명도: 회색조 라플라시안 절댓값 평균
  noise             잔결: 회색조 - 가우시안(σ=1) 의 표준편차
  sat               채도 평균
  rot180            기준 정상 대비 180° 돌아가 있었는지 (정렬 전)
결함 영역은 마스크 + 40px를 빼고 잰다(결함 자체가 통계에 섞이지 않게). 효과 크기 d = (불량 평균 - 정상 평균) / 정상 표준편차.

    python -m AnomalyDetection.ad.capture_probe [--prep canon|std] [--data-root ...]
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter

from .data import PCBS, ROOT, Data, read_csv
from .prep import Canon, Std

import sys
sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import board_transform  # noqa: E402

STATS = ('illum_x', 'illum_y', 'sharp', 'noise', 'sat')


def stats(a, keep):
    g = a.astype(np.float32).mean(2)
    h = g.shape[0] // 16
    low = g.reshape(16, h, 16, h).mean((1, 3))
    yy, xx = np.mgrid[0:16, 0:16]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(256)], 1)
    cx, cy, _ = np.linalg.lstsq(A, low.ravel(), rcond=None)[0]
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    blur = np.asarray(Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), np.float32)
    hsv = np.asarray(Image.fromarray(a).convert('HSV'), np.float32)
    k = keep[1:-1, 1:-1]
    return dict(illum_x=cx * 16 / g.mean(), illum_y=cy * 16 / g.mean(), sharp=float(lap[k].mean()),
                noise=float((g - blur)[keep].std()), sat=float(hsv[..., 1][keep].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root')
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--prep', default='canon', choices=('canon', 'std'))
    ap.add_argument('--every', type=int, default=1, help='train_normal을 몇 장마다 하나씩 쓸지(속도)')
    a = ap.parse_args()
    data = Data(a.data_root)
    pool = read_csv(data.splits / 'pool.csv')
    for pcb in a.pcbs:
        c = (Std if a.prep == 'std' else Canon)(data, pcb, 512)
        rows = []
        for x in data.train_normal(pcb)[::a.every]:
            arr = c.fix(c.item(x.image))
            with Image.open(data.path(x.image)) as im:
                rot = board_transform(im.convert('RGB'), c.ref)['rot180']
            rows.append(dict(kind='normal', rot180=int(rot), **stats(arr, np.ones(arr.shape[:2], bool))))
        for r in [r for r in pool if r['category'] == pcb]:
            with Image.open(data.path(r['image'])) as im:
                img = im.convert('RGB')
                tf = board_transform(img, c.ref)
            with Image.open(data.path(r['mask'])) as mm:
                m = mm.convert('L')
            m = m.rotate(180) if tf['rot180'] else m
            m = m.transform(c.ref.size, Image.AFFINE, (1, 0, -tf['dx'], 0, 1, -tf['dy']), resample=Image.NEAREST)
            m = m.resize((512, 512), Image.NEAREST).filter(ImageFilter.MaxFilter(2 * (40 * 512 // c.ref.size[0]) + 1))
            keep = np.asarray(m) == 0
            arr = c.fix(c.item(r['image']))
            rows.append(dict(kind='defect', rot180=int(tf['rot180']), **stats(arr, keep)))
        N = [r for r in rows if r['kind'] == 'normal']
        D = [r for r in rows if r['kind'] == 'defect']
        print(f'\n{pcb}: 정상 {len(N)}장 (180° {np.mean([r["rot180"] for r in N]):.0%}), '
              f'불량 {len(D)}장 (180° {np.mean([r["rot180"] for r in D]):.0%})')
        groups = {'정상 전체': N}
        if 0 < np.mean([r['rot180'] for r in N]) < 1:
            groups['정상 0°'] = [r for r in N if not r['rot180']]
            groups['정상 180°'] = [r for r in N if r['rot180']]
        for s in STATS:
            mu, sd = np.mean([r[s] for r in N]), np.std([r[s] for r in N]) + 1e-9
            line = f'  {s:8s} 불량 d={(np.mean([r[s] for r in D]) - mu) / sd:+.2f}'
            for gname, G in groups.items():
                if gname != '정상 전체':
                    line += f' | {gname} 대비 d={(np.mean([r[s] for r in D]) - np.mean([r[s] for r in G])) / sd:+.2f}'
            print(line, flush=True)


if __name__ == '__main__':
    main()
