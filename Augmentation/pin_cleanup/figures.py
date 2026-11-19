"""발표 그림 (11/19): 잔상 제거한 세트로 다시 만든 SD+LoRA v3 예시 시트와 잔상 제거 전후 그림.

    cd Augmentation
    python -m pin_cleanup.figures --data-root /path/to/VisA_PCB
결과 (Augmentation/figures/)
  sd_lora_v3_fixed_pcb{1,2,3}_sheet.jpg  종류마다 한 줄, 예 2개, 각 [real K=1 defect | base normal | generated].
                                         옛 sd_lora_v3_pcb*_sheet.jpg(잔상 제거 전)와 같은 장을 고른다.
                                         창 = 마스크 중심, 한 변 = 마스크 긴 변 × 3 (256–512px).
  pinfix_before_after.jpg                bent, K1 seed 0. 왼쪽 SD+LoRA v3, 오른쪽 cTransGAN.
                                         PCB마다 [real K=1 defect | before | after], 잔상 제거로 가장 많이 바뀐 장.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.kshot import GENERATED, LANCZOS, PCBS, RAW_PREFIX, ROOT, TYPES, TYPE_ID, read_rows

FIG = Path(__file__).resolve().parents[1] / 'figures'
CELL, GAP, HEAD = 260, 5, 26
# 옛 시트(ppt_figures/s08_sd_lora_v3_pcb*_sheet.jpg)에 적힌 score로 찾은 장 번호
PICKS = {
    'pcb1': {'bent': (21, 3), 'melt': (30, 29), 'scratch': (12, 37), 'missing': (28, 20)},
    'pcb2': {'bent': (38, 27), 'melt': (7, 23), 'scratch': (0, 27), 'missing': (9, 24)},
    'pcb3': {'bent': (12, 21), 'melt': (19, 23), 'scratch': (16, 29), 'missing': (23, 3)},
}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Raw:
    def __init__(self, data_root):
        self.root = Path(data_root) if data_root else ROOT / RAW_PREFIX

    def path(self, rel):
        return self.root / rel[len(RAW_PREFIX):]

    def image(self, rel):
        return Image.open(self.path(rel)).convert('RGB')

    def defect_box(self, rel, t):
        ids = np.asarray(Image.open(str(self.path(rel)).replace('/Images/', '/Masks/').replace('.JPG', '.png')))
        m = ids == TYPE_ID[t]
        if not m.any():
            m = ids > 0
        ys, xs = np.nonzero(m)
        return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def square(box, side, size):
    """box 중심의 side×side 창 (사진 밖으로 나가지 않게 민다)."""
    w, h = size
    side = int(min(side, w, h))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, x0 + side, y0 + side


def mask_box(path):
    m = np.asarray(Image.open(path)) > 0
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def panel(img, box):
    return img.crop(box).resize((CELL, CELL), LANCZOS)


def by_name(rows):
    return {Path(r['image']).name: r for r in rows}


def sd_sheets(raw, k, seed):
    rows = by_name(read_rows(GENERATED / 'sd_lora_v3' / f'K{k}_seed{seed}' / 'manifest.csv'))
    f = font(13)
    half = 3 * CELL + 2 * GAP
    outs = []
    for p in PCBS:
        sheet = Image.new('RGB', (2 * half + GAP, len(TYPES) * (CELL + HEAD)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, t in enumerate(TYPES):
            for j, idx in enumerate(PICKS[p][t]):
                r = rows[f'{p}_{t}_{idx:03d}.jpg']
                cond = json.loads(r['condition'])
                gen = Image.open(ROOT / r['image']).convert('RGB')
                mb = mask_box(ROOT / r['mask'])
                side = int(np.clip(3 * max(mb[2] - mb[0], mb[3] - mb[1]), 256, 512))
                box = square(mb, side, gen.size)
                side = box[2] - box[0]
                donor = raw.image(r['source_image'])
                dbox = square(raw.defect_box(r['source_image'], t), side, donor.size)
                x, y = j * (half + GAP), i * (CELL + HEAD)
                draw.text((x + 4, y + 6), f'{p} {t}  score={float(r["score"]):.2f}    real K={k} defect | base normal | generated',
                          fill='black', font=f)
                for c, im in enumerate((panel(donor, dbox), panel(raw.image(r['base_normal_image']), box), panel(gen, box))):
                    sheet.paste(im, (x + c * (CELL + GAP), y + HEAD))
        out = FIG / f'sd_lora_v3_fixed_{p}_sheet.jpg'
        sheet.save(out, quality=92)
        outs.append(out)
    return outs


def before_after(raw, k, seed):
    gens = (('SD+LoRA v3', 'sd_lora_v3'), ('cTransGAN', 'ctransgan_diverse'))
    block = 3 * CELL + 2 * GAP
    title_h = 34
    sheet = Image.new('RGB', (2 * block + 4 * GAP, title_h + len(PCBS) * (CELL + HEAD)), 'white')
    draw = ImageDraw.Draw(sheet)
    big, f = font(20), font(13)
    for g, (title, name) in enumerate(gens):
        x0 = g * (block + 4 * GAP)
        draw.text((x0 + 4, 6), f'{title}: bent, K={k} seed {seed}', fill='black', font=big)
        rows = read_rows(GENERATED / name / f'K{k}_seed{seed}' / 'manifest.csv')
        for i, p in enumerate(PCBS):
            cand = [(json.loads(r['condition']).get('pinfix', {}).get('px', 0), r) for r in rows
                    if r['category'] == p and r['defect_type'] == 'bent']
            px, r = max(cand, key=lambda c: c[0])
            pf = json.loads(r['condition'])['pinfix']
            after = Image.open(ROOT / r['image']).convert('RGB')
            before = Image.open(ROOT / pf['from']).convert('RGB')
            diff = np.abs(np.asarray(after, np.int16) - np.asarray(before, np.int16)).max(2) > 12
            mb = mask_box(ROOT / r['mask'])
            if diff.any():
                ys, xs = np.nonzero(diff)
                mb = (min(mb[0], xs.min()), min(mb[1], ys.min()), max(mb[2], xs.max() + 1), max(mb[3], ys.max() + 1))
            side = max(mb[2] - mb[0], mb[3] - mb[1]) + 80
            box = square(mb, max(side, 220), after.size)
            donor = raw.image(r['source_image'])
            dbox = square(raw.defect_box(r['source_image'], 'bent'), box[2] - box[0], donor.size)
            y = title_h + i * (CELL + HEAD)
            draw.text((x0 + 4, y + 6), f'{p}  real K={k} defect | before (ghost pins) | after cleanup   ({px} px fixed)',
                      fill='black', font=f)
            for c, im in enumerate((panel(donor, dbox), panel(before, box), panel(after, box))):
                sheet.paste(im, (x0 + c * (CELL + GAP), y + HEAD))
    out = FIG / 'pinfix_before_after.jpg'
    sheet.save(out, quality=92)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', help='VisA_PCB 원본 (기본: Dataset/visa/raw)')
    p.add_argument('--k', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    raw = Raw(a.data_root)
    for out in sd_sheets(raw, a.k, a.seed) + [before_after(raw, a.k, a.seed)]:
        print('저장:', out)


if __name__ == '__main__':
    main()
