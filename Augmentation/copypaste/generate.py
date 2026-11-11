"""K장 실험용 copy-paste 생성 (공유 분할표 기준).

조건 (K, seed)마다 PCB1–3 × 결함 종류 4개 × 종류별 n장을 만든다.
  - 결함 소스: train_K{k}_seed{s}.csv 의 해당 PCB·종류 불량
  - 정상 바탕: train_normal.csv 의 해당 PCB 정상
  - 원본 해상도 그대로, 마스크 픽셀값은 종류 ID(1=bent, 2=melt, 3=scratch, 4=missing)
  - 결함 원본과 바탕의 기판 위치를 맞춘 뒤(평행이동 + PCB1의 180° 회전) 같은 부품 위치에 붙인다.
    jitter는 bent·missing ±5px, melt·scratch ±30px.
  - 패치에 좌우 반전·±3° 회전·크기 ±10%·밝기/대비 ±5%를 건다(PATCH_AUG)
  - 이미지는 원본처럼 JPEG로 저장한다(PNG로 두면 "압축 흔적 없음 = 생성"이라는 지름길이 생긴다).
    VisA PCB 원본은 IJG quality 75, 4:2:0이다(양자화 테이블로 확인, 11/11). 같은 설정으로 저장해 압축 흔적을 맞춘다.

출력 (AD README 9절 ⑧):
  Dataset/generated/copypaste/K{k}_seed{s}/
    images/{pcb}_{type}_{i:03d}.jpg
    masks/{pcb}_{type}_{i:03d}.png
    manifest.csv   image, mask, label, source_image, score, category, defect_type,
                   base_normal_image, gen_seed, condition

사용 예 (Augmentation 에서):
  python -m copypaste.generate --k 2 --seed 0
  python -m copypaste.generate --all                       # K {1,2,4,8} × seed 0–4
  python -m copypaste.generate --all --data-root /content/VisA_PCB

AD 팀 버전 생성기(2c713be, 옛 visa_pcb/src/generate/k_copypaste.py)를 옮긴 것이다. 난수 규칙이 같아서
그 버전으로 만든 Dataset/generated/copypaste/*/manifest.csv 를 그대로 다시 만든다.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt, label

from common.kshot import ROOT as PROJECT_ROOT, PCBS, TYPES, TYPE_ID, KShot, board_transform, check_manifest
from common.kshot import JITTER_BY_TYPE

METHOD = 'copypaste'
FIELDS = ['image', 'mask', 'label', 'source_image', 'score', 'category', 'defect_type',
          'base_normal_image', 'gen_seed', 'condition']


# 난수: 예전 src/seeds.py 규칙 그대로 (copypaste = 1). 바꾸면 기존 결과가 재현되지 않는다.
def _entropy(pcb, k, split_seed):
    return [7017, 1, int(pcb[-1]), k, split_seed]


def condition_seed(method, pcb, k, split_seed):
    state = np.random.SeedSequence(_entropy(pcb, k, split_seed)).generate_state(2, np.uint32)
    return (int(state[0]) << 31) ^ int(state[1])


def sample_rng(method, pcb, k, split_seed, index):
    return np.random.default_rng(np.random.SeedSequence(_entropy(pcb, k, split_seed) + [index]))


def rel(path):
    """프로젝트 루트 기준 상대경로. 프로젝트 밖(테스트용 임시 폴더)이면 절대경로."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# 결함 패치에만 거는 기본 증강 (11/11 AD 팀 요청: 크게 돌리지 않는다).
# 같은 PCB 사진은 촬영 방향·조명이 거의 같아서, 실제로 나올 법한 작은 변화만 준다.
PATCH_AUG = dict(
    hflip_p=0.5,            # 좌우 반전 확률
    rotation_deg=3.0,       # ±3° 회전
    scale=(0.9, 1.1),       # 크기 ±10%
    brightness=0.05,        # 밝기 ±5%
    contrast=0.05,          # 대비 ±5%
)


def augment_patch(patch, support, rng, aug):
    """패치(RGB float32)와 결함 영역(bool)에 같은 기하 변환을 걸고, 패치에만 밝기·대비를 바꾼다.
    돌아가는 값: (패치, 결함 영역, 실제로 쓴 변환 dict)."""
    applied = {}
    img = Image.fromarray(patch.clip(0, 255).astype(np.uint8))
    sup = Image.fromarray(support.astype(np.uint8) * 255)
    if aug.get('hflip_p', 0) and rng.random() < aug['hflip_p']:
        img, sup = ImageOps.mirror(img), ImageOps.mirror(sup)
        applied['hflip'] = True
    angle = float(rng.uniform(-aug.get('rotation_deg', 0), aug.get('rotation_deg', 0)))
    lo, hi = aug.get('scale', (1.0, 1.0))
    scale = float(rng.uniform(lo, hi))
    if abs(angle) > 1e-6 or abs(scale - 1) > 1e-6:
        w, h = img.size
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img, sup = img.resize((nw, nh), Image.Resampling.BILINEAR), sup.resize((nw, nh), Image.Resampling.NEAREST)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
        sup = sup.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
    applied.update(rotation_deg=round(angle, 3), scale=round(scale, 4))
    out = np.asarray(img, dtype=np.float32)
    s = np.asarray(sup) > 127
    b = float(rng.uniform(-aug.get('brightness', 0), aug.get('brightness', 0)))
    c = float(rng.uniform(-aug.get('contrast', 0), aug.get('contrast', 0)))
    if s.any() and (b or c):
        mean = out[s].mean(axis=0)
        out = (out - mean) * (1 + c) + mean * (1 + b)
    applied.update(brightness=round(b, 4), contrast=round(c, 4))
    if not s.any():                       # 너무 작은 결함이 변환으로 사라지면 원본 패치로 되돌린다
        return patch, support, {'fallback': 'identity'}
    ys, xs = np.nonzero(s)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return out[y0:y1, x0:x1], s[y0:y1, x0:x1], applied


def paste_same_place(target, donor, mask, rng, jitter, feather, aug=None):
    """결함을 원래 있던 자리(± jitter px)에 붙인다. aug가 있으면 패치에 작은 변환(PATCH_AUG)을 건다.
    같은 PCB 이미지들은 기판 위치가 거의 같아서, 결함이 기판 밖 배경에 떨어지지 않는다."""
    assert target.size == donor.size, '같은 PCB 안에서는 해상도가 같아야 합니다'
    x0, y0, x1, y1 = mask.getbbox()
    w, h = target.size
    support = np.asarray(mask.crop((x0, y0, x1, y1))) > 0
    patch = np.asarray(donor.crop((x0, y0, x1, y1)), dtype=np.float32)
    # 기판 정렬: 결함 중심을 바탕 기판의 같은 부품 위치로 옮긴다
    tf = board_transform(donor, target)
    cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    if tf['rot180']:
        patch, support = patch[::-1, ::-1], support[::-1, ::-1]
        cx, cy = (w - 1) - cx, (h - 1) - cy
    cx, cy = cx + tf['dx'], cy + tf['dy']
    applied = {}
    if aug:
        patch, support, applied = augment_patch(patch, support, rng, aug)
    ph, pw = support.shape
    x0, y0 = int(round(cx - (pw - 1) / 2)), int(round(cy - (ph - 1) / 2))
    x0, y0 = int(np.clip(x0, 0, w - pw)), int(np.clip(y0, 0, h - ph))
    x1, y1 = x0 + pw, y0 + ph
    dx, dy = (int(v) for v in rng.integers(-jitter, jitter + 1, 2))
    dx = int(np.clip(dx, -x0, w - x1))
    dy = int(np.clip(dy, -y0, h - y1))
    distance = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    blend = (np.minimum(1, distance / max(feather, 1e-6)) * support)[..., None]
    roi = np.s_[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    source = np.asarray(target)
    result = source.copy()
    result[roi] = np.rint(source[roi] * (1 - blend) + patch * blend).clip(0, 255).astype('uint8')
    out_mask = np.zeros((h, w), np.uint8)
    out_mask[roi] = support * 255
    area = support.sum() / (w * h)
    condition = {'placement': 'same', 'offset_xy': [dx, dy], 'jitter': jitter,
                 'align': {'rot180': bool(tf['rot180']), 'dx': int(tf['dx']), 'dy': int(tf['dy']),
                           'score': round(tf['score'], 4)},
                 'area_fraction': float(area), 'connected_components': int(label(support)[1]),
                 'patch_aug': applied}
    return Image.fromarray(result), Image.fromarray(out_mask), condition


def generate_condition(data_root, k, seed, out_root, per_type=40, pcbs=PCBS, jpeg_quality=75,
                       jitter=None, feather=2.0, force=False, patch_aug=PATCH_AUG, splits_dir=None):
    ks = KShot(k, seed, data_root, splits_dir)
    out = Path(out_root) / f'K{k}_seed{seed}'
    manifest = out / 'manifest.csv'
    if manifest.exists() and not force:
        print(f'[건너뜀] 이미 있음: {manifest}')
        return manifest
    if out.exists():
        shutil.rmtree(out)
    (out / 'images').mkdir(parents=True)
    (out / 'masks').mkdir()

    rows = []
    for pcb in pcbs:
        donors = [r for r in ks.k_rows if r['category'] == pcb]
        normals = ks.normals(pcb)
        gen_seed = condition_seed(METHOD, pcb, k, seed)
        for ti, t in enumerate(TYPES):
            typed = [r for r in donors if r['defect_type'] == t]
            for i in range(per_type):
                rng = sample_rng(METHOD, pcb, k, seed, ti * 100000 + i)
                donor = typed[int(rng.integers(len(typed)))]
                base = normals[int(rng.integers(len(normals)))]
                d_img, d_ids = ks.load(donor)
                b_img, _ = ks.load(base)
                d_mask = Image.fromarray(((d_ids == TYPE_ID[t]) * 255).astype(np.uint8))
                j = JITTER_BY_TYPE[t] if jitter is None else jitter
                image, out_mask, cond = paste_same_place(b_img, d_img, d_mask, rng, j, feather, patch_aug)
                ids = (np.asarray(out_mask) > 0).astype(np.uint8) * TYPE_ID[t]
                stem = f'{pcb}_{t}_{i:03d}'
                image.save(out / 'images' / f'{stem}.jpg', quality=jpeg_quality, subsampling=2)
                Image.fromarray(ids).save(out / 'masks' / f'{stem}.png')
                rows.append(dict(
                    image=rel(out / 'images' / f'{stem}.jpg'), mask=rel(out / 'masks' / f'{stem}.png'),
                    label=1, source_image=donor['image'], score='', category=pcb, defect_type=t,
                    base_normal_image=base['image'], gen_seed=gen_seed,
                    condition=json.dumps(cond, sort_keys=True)))
        print(f'  K{k}_seed{seed} {pcb}: {per_type * len(TYPES)}장', flush=True)

    tmp = manifest.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(manifest)
    print(check_manifest(manifest, splits_dir), flush=True)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--seed', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--all', action='store_true', help='--k, --seed 기본값 전체 실행')
    p.add_argument('--pcb', nargs='+', default=list(PCBS))
    p.add_argument('--per-type', type=int, default=40)
    p.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: <프로젝트>/Dataset/visa/raw)')
    p.add_argument('--splits', help='기본: <프로젝트>/Dataset/visa/splits')
    p.add_argument('--out', default=str(PROJECT_ROOT / 'Dataset/generated' / METHOD))
    p.add_argument('--jitter', type=int, default=None, help='기본: 종류별 (bent·missing 5px, melt·scratch 30px)')
    p.add_argument('--no-patch-aug', action='store_true', help='패치 증강(좌우 반전, ±3°, 크기·밝기) 끄기')
    p.add_argument('--force', action='store_true', help='이미 만든 조건도 다시 생성')
    a = p.parse_args()
    try:  # Windows 콘솔 한글 깨짐 방지
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    if not a.all and (a.k == [1, 2, 4, 8] and a.seed == [0, 1, 2, 3, 4]):
        p.error('--k 와 --seed 를 지정하거나 --all 을 붙이세요')
    for k in a.k:
        for seed in a.seed:
            generate_condition(a.data_root, k, seed, a.out, a.per_type, a.pcb, jitter=a.jitter, force=a.force,
                               patch_aug=None if a.no_patch_aug else PATCH_AUG, splits_dir=a.splits)
