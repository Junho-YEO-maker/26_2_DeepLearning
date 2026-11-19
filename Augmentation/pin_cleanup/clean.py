"""생성 불량에서 원래 핀 잔상 지우기 (bent·missing).

문제: bent·missing을 만들 때 정상 바탕의 원래 곧은 핀이 다 지워지지 않고 남아, 핀이 이중으로 보이거나
"지워지다 만" 토막이 생긴다. cTransGAN, SD+LoRA, copy-paste 모두 같다 (pcb3 bent가 가장 심함).
원인: 지울 영역을 찾는 common/kshot.py change_region이 실제 결함 사진과 바탕의 **흑백** 밝기 차이를 쓴다.
pcb3의 은색 핀과 초록 기판은 흑백으로 밝기가 비슷해 핀이 토막으로만 잡히고, 헤더 너머의 핀은
마스크에서 20px보다 멀어 빠진다.

방법 (생성기와 무관한 후처리. 생성 이미지·마스크·manifest만 있으면 된다):
  1) 실제 결함 사진(K장 소스)을 manifest의 정렬값으로 바탕 좌표에 옮긴다.
  2) 핀처럼 보이는 픽셀(무채색이고 밝음 = 은색 핀)을 **색**으로 찾는다. 바탕에는 핀이 있는데 실제 결함 사진의
     같은 자리(±6px)에는 없고, 마스크에서 45px 안에 닿는 덩어리 = 바탕에만 있는 원래 핀. 두 사진에 다 있는
     정상 핀은 몇 px 어긋나도 건드리지 않는다. 정답 마스크 밖만 고친다.
  3) 그 자리를 실제 결함 사진의 같은 자리(핀이 없는 배경)로 메운다. 밝기는 주변 고리의 중앙값 차이로 맞추고,
     경계 3px는 섞는다. 실제 사진의 결함 자리와 겹치는 픽셀은 가까운 배경 값으로 메운다.
  정답 마스크와 마스크 안 픽셀은 바꾸지 않는다. 고치지 않는 종류의 행은 파일을 그대로 복사한다.

기본은 bent만 고친다 (11/18, K1 seed 0 예시로 확인):
  - bent: pcb3에서 cTransGAN·SD+LoRA의 곧은 핀 잔상이 지워진다. SD+LoRA는 헤더 옆 짧은 토막이 조금 남는다.
    pcb1·2는 정상 핀을 건드리지 않는다. copy-paste는 패치를 뒤집어 붙인 장에서 헤더 근처가 번질 수 있다.
  - missing: cTransGAN은 원래 잔상이 없고, SD+LoRA는 이 방법이 핀 가장자리를 파먹어 뺐다(--types로 켤 수는 있음).

    cd Augmentation
    python -m pin_cleanup.clean --aug sd_lora_v3 --k 1 --seed 0 --data-root $DR
    python -m pin_cleanup.clean --aug ctransgan_diverse --k 1 --seed 0 1 2 3 4 --data-root $DR
    python -m pin_cleanup.clean --manifest /경로/K1_seed0/manifest.csv --out-name ctransgan_diverse_pinfix --k 1 --seed 0
결과: Dataset/generated/{aug}_pinfix/K{k}_seed{s}/{images,masks,manifest.csv} (+ review.jpg 전후 비교), 끝에 누수 검사
"""
import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from common.kshot import (GENERATED, JPEG_QUALITY, KShot, MANIFEST_FIELDS, ROOT, STRUCTURE_TYPES, TYPE_ID,
                          aligned_donor, check_manifest, dilate, read_rows, rel, shift_mask)

PAD = 220          # 마스크 주변 몇 px까지 볼지
GRAY = .30         # 핀처럼 보이는 픽셀: 채도((max−min)/max)가 이보다 낮고
BRIGHT = 110       #   가장 밝은 채널이 이보다 밝다 (은색 핀. 검은 헤더·초록/파랑 기판·검은 배경은 빠진다)
TOL = 6            # 실제 결함 사진에 이 거리 안에 핀이 있으면 "원래 있던 정상 핀"으로 보고 건드리지 않는다
OPEN = 3           # 얇은 띠(조금 어긋난 글자·가장자리) 제거
TOUCH = 45         # 마스크(또는 실제 사진의 결함 자리)에서 이 거리 안에 닿는 덩어리만. 헤더 너머 핀까지 닿게 45
GROW = 5           # 핀 가장자리·그림자까지 덮도록 넓히는 px
FEATHER = 3        # 경계 섞는 폭


def shift_of(cond):
    """manifest condition → 소스를 바탕 좌표로 옮긴 실제 이동량 (생성기마다 기록 방식이 조금 다르다)."""
    if 'applied_shift_xy' in cond:
        return [int(v) for v in cond['applied_shift_xy']]
    ox, oy = cond.get('offset_xy', [0, 0])
    return [int(cond['align']['dx'] + ox), int(cond['align']['dy'] + oy)]


def pin_like(a):
    top = a.max(-1)
    return ((top - a.min(-1)) / (top + 1) < GRAY) & (top > BRIGHT)


def residue_region(base, donor_al, mask, donor_def, touch=TOUCH):
    """bool (바탕 좌표): 바탕에는 핀이 있는데 실제 결함 사진의 같은 자리(±TOL)에는 핀이 없는 곳 − 정답 마스크.
    두 사진에 다 있는 정상 핀은 몇 px 어긋나도 건드리지 않는다."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, h)
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, w)
    blur = ImageFilter.GaussianBlur(2)
    b = np.asarray(base.crop((x0, y0, x1, y1)).filter(blur), np.float32)
    d = np.asarray(donor_al.crop((x0, y0, x1, y1)).filter(blur), np.float32)
    donor_pin = pin_like(d)
    cand = ndimage.binary_opening(pin_like(b) & ~dilate(donor_pin, TOL), iterations=OPEN)
    labels, _ = ndimage.label(cand)
    seed = mask[y0:y1, x0:x1] | donor_def[y0:y1, x0:x1]
    ids = np.unique(labels[dilate(seed, touch)])
    keep = ndimage.binary_closing(np.isin(labels, ids[ids > 0]), iterations=4)
    keep = dilate(keep, GROW) & ~dilate(donor_pin, 2)      # 메울 때 실제 사진의 핀을 끌어오지 않게
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = keep
    return out & ~mask


def clean_image(gen, base, donor, donor_region, mask, cond, touch=TOUCH):
    """돌아가는 값: (고친 이미지, 고친 영역 bool)."""
    sx, sy = shift_of(cond)
    rot = bool(cond['align']['rot180'])
    donor_al = aligned_donor(donor, base, {'align': cond['align'], 'applied_shift_xy': [sx, sy]})
    oriented = donor_region[::-1, ::-1] if rot else donor_region
    donor_def = dilate(shift_mask(oriented, sx, sy)[0], 4)
    region = residue_region(base, donor_al, mask, donor_def, touch)
    if not region.any():
        return gen, region
    g = np.asarray(gen, np.float32)
    d = np.asarray(donor_al, np.float32)
    ring = dilate(region, 14) & ~dilate(region, 3) & ~mask & ~donor_def
    fill = d + (np.median(g[ring] - d[ring], 0) if ring.sum() > 50 else 0)
    bad = region & donor_def                    # 실제 사진의 결함 자리: 가장 가까운 배경 값 + 흐림
    if bad.any():
        _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
        near = fill[iy, ix]
        soft = np.stack([ndimage.gaussian_filter(near[..., c], 3) for c in range(3)], -1)
        fill = np.where(bad[..., None], soft, fill)
    alpha = np.clip(ndimage.distance_transform_edt(region) / FEATHER, 0, 1)[..., None]
    out = g * (1 - alpha) + fill * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), region


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


_data = None
_sources = {}


def _init(k, seed, data_root, splits):
    global _data
    _data = KShot(k, seed, data_root, splits)


def _source(pcb, defect, image):
    key = (pcb, defect, image)
    if key not in _sources:
        row = next(r for r in _data.defects(pcb, defect) if r['image'] == image)
        donor, ids = _data.load(row)
        _sources[key] = (donor.convert('RGB'), ids == TYPE_ID[defect])
    return _sources[key]


def _work(job):
    """한 행: 고칠 종류면 고쳐서 저장, 아니면 복사. 돌아가는 값: (고친 px 수, 크롭 박스)."""
    row, out_img, out_mask, types, touch = job
    src_img, src_mask = resolve(row['image']), resolve(row['mask'])
    shutil.copyfile(src_mask, out_mask)
    if row['defect_type'] not in types:
        shutil.copyfile(src_img, out_img)
        return 0, None
    cond = json.loads(row['condition'])
    gen = Image.open(src_img).convert('RGB')
    mask = np.array(Image.open(src_mask)) > 0
    base = Image.open(_data.path(row['base_normal_image'])).convert('RGB')
    donor, donor_region = _source(row['category'], row['defect_type'], row['source_image'])
    fixed, region = clean_image(gen, base, donor, donor_region, mask, cond, touch)
    if not region.any():
        shutil.copyfile(src_img, out_img)
        return 0, None
    fixed.save(out_img, quality=JPEG_QUALITY, subsampling=2)
    ys, xs = np.nonzero(mask | region)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 50
    return int(region.sum()), (cx - half, cy - half, cx + half, cy + half)


def review(pairs, path, tile=220):
    """PCB·종류마다 한 장씩: 위 = 원래, 아래 = 고친 것."""
    if not pairs:
        return
    sheet = Image.new('RGB', (len(pairs) * (tile + 4), 2 * (tile + 4) + 16), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, (name, before, after, box) in enumerate(pairs):
        draw.text((j * (tile + 4) + 2, 2), name, fill='black')
        for i, p in enumerate((before, after)):
            with Image.open(p) as im:
                sheet.paste(im.convert('RGB').crop(box).resize((tile, tile)), (j * (tile + 4), 16 + i * (tile + 4)))
    sheet.save(path, quality=90)


def run(manifest, out_dir, k, seed, a):
    rows = read_rows(manifest)
    if (out_dir / 'manifest.csv').exists() and not a.force:
        print(f'이미 있음: {out_dir / "manifest.csv"} (다시 만들려면 --force)')
        return None
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'masks').mkdir(exist_ok=True)
    jobs = [(r, out_dir / 'images' / Path(r['image']).name, out_dir / 'masks' / Path(r['mask']).name,
             set(a.types), a.touch) for r in rows]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(k, seed, a.data_root, a.splits)) as ex:
        results = list(ex.map(_work, jobs, chunksize=4))
    new_rows, pairs, seen = [], [], set()
    for (r, out_img, out_mask, _, _), (px, box) in zip(jobs, results):
        cond = json.loads(r['condition'])
        if r['defect_type'] in a.types:
            cond['pinfix'] = {'px': px, 'touch': a.touch, 'tol': TOL, 'from': rel(resolve(r['image']))}
        new_rows.append({**r, 'image': rel(out_img), 'mask': rel(out_mask), 'condition': json.dumps(cond, sort_keys=True)})
        key = (r['category'], r['defect_type'])
        if box and key not in seen:
            seen.add(key)
            pairs.append((f'{key[0]} {key[1]}', resolve(r['image']), out_img, box))
    with (out_dir / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(new_rows)
    review(sorted(pairs), out_dir / 'review.jpg')
    fixed = [px for (r, *_), (px, _) in zip(jobs, results) if r['defect_type'] in a.types]
    print(f'K{k}_seed{seed}: {len(rows)}행, 고친 {sum(p > 0 for p in fixed)}/{len(fixed)}장 '
          f'(고친 면적 중앙값 {int(np.median([p for p in fixed if p] or [0]))}px), {time.time() - t0:.0f}s', flush=True)
    result = check_manifest(out_dir / 'manifest.csv', a.splits)
    print(result, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aug', help='Dataset/generated 아래 이름 (예: sd_lora_v3, ctransgan_diverse)')
    p.add_argument('--manifest', help='--aug 대신 manifest 경로 하나 (프로젝트 밖 결과용). --k·--seed는 하나씩')
    p.add_argument('--out-name', help='기본: {aug}_pinfix')
    p.add_argument('--k', type=int, nargs='+', required=True)
    p.add_argument('--seed', type=int, nargs='+', default=[0])
    p.add_argument('--types', nargs='+', default=['bent'], choices=STRUCTURE_TYPES,
                   help='고칠 결함 종류 (기본 bent만. missing은 SD+LoRA에서 핀 가장자리를 파먹어 11/18 시험에서 뺐다)')
    p.add_argument('--touch', type=int, default=TOUCH)
    p.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: <프로젝트>/Dataset/visa/raw)')
    p.add_argument('--splits', help='기본: <프로젝트>/Dataset/visa/splits')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if not (a.aug or a.manifest) or (a.manifest and not a.out_name):
        p.error('--aug, 또는 --manifest와 --out-name을 주세요')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    out_root = GENERATED / (a.out_name or f'{a.aug}_pinfix')
    for k in a.k:
        for seed in a.seed:
            manifest = Path(a.manifest) if a.manifest else GENERATED / a.aug / f'K{k}_seed{seed}' / 'manifest.csv'
            if not manifest.exists():
                print(f'없음, 건너뜀: {manifest}')
                continue
            run(manifest, out_root / f'K{k}_seed{seed}', k, seed, a)


if __name__ == '__main__':
    main()
