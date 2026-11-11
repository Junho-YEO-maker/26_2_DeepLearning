"""K장 실험용 copy-paste 생성 (공유 분할표 기준).

조건 (K, seed)마다 PCB1–3 × 결함 종류 4개 × 종류별 n장을 만든다.
  - 결함 소스: train_K{k}_seed{s}.csv 의 해당 PCB·종류 불량
  - 정상 바탕: train_normal.csv 의 해당 PCB 정상
  - 원본 해상도 그대로, 마스크 픽셀값은 종류 ID(1=bent, 2=melt, 3=scratch, 4=missing)
  - 이미지는 원본처럼 JPEG로 저장한다(PNG로 두면 "압축 흔적 없음 = 생성"이라는 지름길이 생긴다)

출력 (AD README 9절 ⑧):
  Dataset/generated/copypaste/K{k}_seed{s}/
    images/{pcb}_{type}_{i:03d}.jpg
    masks/{pcb}_{type}_{i:03d}.png
    manifest.csv   image, mask, label, source_image, score, category, defect_type,
                   base_normal_image, gen_seed, condition

사용 예 (Augmentation/visa_pcb 에서):
  python -m src.generate.k_copypaste --k 2 --seed 0
  python -m src.generate.k_copypaste --all                       # K {1,2,4,8} × seed 0–4
  python -m src.generate.k_copypaste --all --data-root /content/VisA_PCB
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from src.data.shared_splits import PROJECT_ROOT, PCBS, TYPES, TYPE_ID, SharedSplits, check_manifest
from src.models.copy_paste import paste
from src.seeds import condition_seed, sample_rng

METHOD = 'copypaste'
FIELDS = ['image', 'mask', 'label', 'source_image', 'score', 'category', 'defect_type',
          'base_normal_image', 'gen_seed', 'condition']


def rel(path):
    """프로젝트 루트 기준 상대경로. 프로젝트 밖(테스트용 임시 폴더)이면 절대경로."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def paste_same_place(target, donor, mask, rng, jitter, feather):
    """결함을 원래 있던 자리(± jitter px)에 회전 없이 붙인다.
    같은 PCB 이미지들은 기판 위치가 거의 같아서, 결함이 기판 밖 배경에 떨어지지 않는다."""
    assert target.size == donor.size, '같은 PCB 안에서는 해상도가 같아야 합니다'
    x0, y0, x1, y1 = mask.getbbox()
    w, h = target.size
    dx, dy = (int(v) for v in rng.integers(-jitter, jitter + 1, 2))
    dx = int(np.clip(dx, -x0, w - x1))
    dy = int(np.clip(dy, -y0, h - y1))
    support = np.asarray(mask.crop((x0, y0, x1, y1))) > 0
    patch = np.asarray(donor.crop((x0, y0, x1, y1)), dtype=np.float32)
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
                 'area_fraction': float(area), 'connected_components': int(label(support)[1])}
    return Image.fromarray(result), Image.fromarray(out_mask), condition


def generate_condition(splits, k, seed, out_root, per_type=40, pcbs=PCBS, jpeg_quality=95,
                       placement='same', jitter=30, rotation_degrees=(-180, 180), feather=2.0, force=False):
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
        donors = splits.defect_rows(k, seed, pcb)
        normals = splits.normal_rows(pcb)
        gen_seed = condition_seed(METHOD, pcb, k, seed)
        for ti, t in enumerate(TYPES):
            typed = [r for r in donors if r['defect_type'] == t]
            for i in range(per_type):
                rng = sample_rng(METHOD, pcb, k, seed, ti * 100000 + i)
                donor = typed[int(rng.integers(len(typed)))]
                base = normals[int(rng.integers(len(normals)))]
                d_img, d_ids = splits.open(donor)
                b_img, _ = splits.open(base)
                d_mask = Image.fromarray(((d_ids == TYPE_ID[t]) * 255).astype(np.uint8))
                if placement == 'same':
                    image, out_mask, cond = paste_same_place(b_img, d_img, d_mask, rng, jitter, feather)
                else:   # 2단계 기준선: 임의 위치·회전 (기판 밖 배경에 붙을 수 있음)
                    image, out_mask, cond = paste(b_img, d_img, d_mask, rng, list(rotation_degrees), feather)
                    cond['placement'] = 'random'
                ids = (np.asarray(out_mask) > 0).astype(np.uint8) * TYPE_ID[t]
                stem = f'{pcb}_{t}_{i:03d}'
                image.save(out / 'images' / f'{stem}.jpg', quality=jpeg_quality)
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
    print(check_manifest(manifest, splits.splits), flush=True)
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
    p.add_argument('--placement', choices=['same', 'random'], default='same',
                   help='same: 원래 자리 ± jitter (기본), random: 2단계 기준선처럼 임의 위치·회전')
    p.add_argument('--jitter', type=int, default=30)
    p.add_argument('--force', action='store_true', help='이미 만든 조건도 다시 생성')
    a = p.parse_args()
    try:  # Windows 콘솔 한글 깨짐 방지
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    if not a.all and (a.k == [1, 2, 4, 8] and a.seed == [0, 1, 2, 3, 4]):
        p.error('--k 와 --seed 를 지정하거나 --all 을 붙이세요')
    s = SharedSplits(a.splits, a.data_root)
    for k in a.k:
        for seed in a.seed:
            generate_condition(s, k, seed, a.out, a.per_type, a.pcb, placement=a.placement,
                               jitter=a.jitter, force=a.force)
