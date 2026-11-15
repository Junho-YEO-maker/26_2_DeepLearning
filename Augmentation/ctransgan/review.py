"""생성 결과 눈으로 확인: PCB·종류마다 한 장씩 [K장 원본 결함 | 정상 바탕 | 생성 | 마스크] 크롭을 한 장에 모은다.

    cd Augmentation
    python -m ctransgan.review --k 2 --seed 0            # → ctransgan/review/K2_seed0.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common.kshot import GENERATED, LANCZOS, PCBS, RAW_PREFIX, ROOT, TYPES, TYPE_ID, check_manifest, read_rows

HERE = Path(__file__).resolve().parent
TITLES = ('K-shot real defect', 'Normal base', 'cTransGAN', 'Mask')


def review(folder, data_root, out, index=0):
    rows = read_rows(folder / 'manifest.csv')
    raw = Path(data_root) if data_root else ROOT / RAW_PREFIX

    def open_raw(rel):
        return Image.open(raw / rel[len(RAW_PREFIX):])

    picked = {}
    for row in rows:
        key = (row['category'], row['defect_type'])
        if Path(row['image']).stem.endswith(f'_{index:03d}'):
            picked[key] = row
    cell_w, cell_h, pad = 190, 130, 22
    sheet = Image.new('RGB', (4 * cell_w + 10, len(PCBS) * len(TYPES) * (cell_h + pad) + 10), 'white')
    draw = ImageDraw.Draw(sheet)
    changes = []
    for n, key in enumerate((p, t) for p in PCBS for t in TYPES):
        if key not in picked:
            continue
        row = picked[key]
        info = json.loads(row['condition'])
        with Image.open(ROOT / row['image']) as gen, Image.open(ROOT / row['mask']) as mask, \
                open_raw(row['base_normal_image']) as base, open_raw(row['source_image']) as donor:
            ids = np.asarray(mask)
            region = ids == TYPE_ID[row['defect_type']]
            assert gen.size == mask.size == base.size and region.any() and int(region.sum()) == info['mask_area']
            x0, y0, x1, y1 = mask.getbbox()
            window = (x0 - 12, y0 - 12, x1 + 12, y1 + 12)
            diff = np.abs(np.asarray(gen.convert('RGB'), np.int16) - np.asarray(base.convert('RGB'), np.int16))
            changes.append(float(diff[region].mean()))
            donor_ids = np.asarray(Image.open(raw / row['source_image'][len(RAW_PREFIX):]
                                              .replace('/Images/', '/Masks/').replace('.JPG', '.png')))
            a, b, c, d = Image.fromarray((donor_ids == TYPE_ID[row['defect_type']]).astype(np.uint8)).getbbox()
            panels = (donor.crop((a - 12, b - 12, c + 12, d + 12)), base.crop(window), gen.crop(window),
                      mask.crop(window).point(lambda v: 255 if v else 0))
            y = 5 + n * (cell_h + pad)
            draw.text((5, y), f'{key[0]} {key[1]}  ({Path(row["image"]).name})', fill='black')
            for i, (title, panel) in enumerate(zip(TITLES, panels)):
                x = 5 + i * cell_w
                draw.text((x, y + 11), title, fill='gray')
                panel = panel.convert('RGB')
                panel.thumbnail((cell_w - 10, cell_h - 14), LANCZOS)
                sheet.paste(panel, (x, y + pad + 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--index', type=int, default=0, help='PCB·종류마다 몇 번째 장을 볼지')
    p.add_argument('--folder', help='기본: Dataset/generated/ctransgan/K{k}_seed{s}')
    p.add_argument('--data-root')
    p.add_argument('--out')
    a = p.parse_args()
    folder = Path(a.folder) if a.folder else GENERATED / 'ctransgan' / f'K{a.k}_seed{a.seed}'
    out = Path(a.out) if a.out else HERE / 'review' / f'K{a.k}_seed{a.seed}.png'
    print(check_manifest(folder / 'manifest.csv'))
    changes = review(folder, a.data_root, out, a.index)
    print(f'{out} 저장. 마스크 안 평균 RGB 변화(바탕 대비) min/median/max:',
          np.round(np.quantile(changes, (0, .5, 1)), 1))


if __name__ == '__main__':
    main()
