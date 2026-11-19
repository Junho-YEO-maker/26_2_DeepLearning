"""cTransGAN 생성: 조건 (K, seed)마다 PCB 3개 × 결함 종류 4개 × 종류별 40장.

Wang et al., Conditional TransGAN-Based Data Augmentation for PCB Electronic Component Inspection
(Computational Intelligence and Neuroscience, 2023, DOI 10.1155/2023/2024237)의 구조를 축소해 K장 실험에 맞췄다.

PCB마다 그 PCB의 K장(종류당 K장, 총 4K장)으로만 새로 학습한다.
  1) Transformer 분류기를 K장 결함 크롭(32×32)으로 학습하고, 분류층 직전 특징의 종류별 평균을 조건 임베딩으로 쓴다.
  2) 생성기 = Transformer(잡음 + 조건 임베딩 → 8×8 → 16×16 → 32×32 토큰, pixel shuffle) → 토큰마다 픽셀 패치로 펼쳐
     256×256 결함 크롭을 직접 그린다(model.py).
  3) 판별기 = 조건 임베딩을 추가 토큰으로 받는 Transformer. 생성 결과를 기판 정렬한 정상 바탕 크롭에
     "다시 그릴 영역"만 끼워 넣은 것(32×32로 줄임)을 본다.
  4) 생성기 손실 = 적대 ×0.25 + 다시 그릴 영역(+12px) L1 ×35 + 경계(1차 차분) L1 + 고주파(3·7px) L1 + 크롭 전체 L1 ×6.
     K장이 너무 적어 적대 손실만으로는 노이즈만 나오므로, 결함마다 잠재 벡터를 하나 고정(anchor)하고 재구성 손실을 같이 쓴다.

다시 그릴 영역: melt·scratch는 결함 마스크. bent·missing은 마스크 + 원래 핀·부품 자리(common/kshot.py change_region,
sd_lora v2와 같은 함수 + 어둡게 찍힌 핀 보강)를 6px 넓힌 것. 마스크만 다시 그리면 정상 바탕의 곧은 핀이 남아
"다리가 하나 더" 생기기 때문이다.
정답 마스크(masks/)는 VisA 주석 모양 그대로다.

생성 규칙은 copy-paste·sd_lora와 같다(common/kshot.py).
  - 결함 소스는 train_K{k}_seed{s}.csv, 정상 바탕은 train_normal.csv 만 (test/val 은 KShot이 막는다)
  - K장 결함 마스크를 바탕 기판의 같은 부품 자리로 옮긴다: 기판 정렬(평행이동 + PCB1의 180° 회전) ± 종류별 jitter
    (melt·scratch 30px). bent·missing은 jitter 대신 ±6px 안에서 바탕의 부품 자리에 정밀하게 맞춘다(refine.py):
    다시 그릴 영역에 핀·기판 선이 들어가서 몇 px만 어긋나도 이중으로 보이기 때문이다.
  - 바탕은 매 장 train_normal 에서 무작위로 5장을 뽑아, 결함 주변이 잘 맞는 순으로 보다가 다시 그릴 영역을
    생성 크롭이 95% 이상 덮는 첫 바탕을 쓴다(같은 결함에 같은 바탕은 한 번만).
  - 다시 그릴 영역만 원본 해상도 바탕에 붙이고 JPEG q75 4:2:0 으로 저장한다.

    cd Augmentation
    python -m ctransgan.generate --k 2 --seed 0
    python -m ctransgan.generate --all                          # K {1,2,4} × seed 0
    python -m ctransgan.generate --all --data-root /content/VisA_PCB
결과: Dataset/generated/ctransgan/K{k}_seed{s}/{images,masks,manifest.csv}
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.nn import functional as F

from common.kshot import (
    GENERATED, JITTER_BY_TYPE, KShot, LANCZOS, ManifestWriter, PCBS, STRUCTURE_TYPES, TYPES, TYPE_ID,
    aligned_crop, aligned_donor, board_transform, change_region, crop_box, dilate, gen_seed, map_box, paste_back,
    rng_for,
    shift_mask,
)
from .refine import refine_shift
from .model import build_models

METHOD = 'ctransgan'
BOX = getattr(Image, 'Resampling', Image).BOX
CROP, MARGIN = 64, 12          # 결함 주변 크롭: 최소 64px, 다시 그릴 영역 + 12px 여백 (더 크면 창을 키우고 res로 줄인다)
SPLIT_ABOVE, MAX_PARTS = 320, 4   # 크롭이 320px보다 크면 떨어진 덩어리별로(최대 4개) 나눠 따로 그린다
TRAIN_BASES = 3                # 학습 때 결함마다 쓰는 정렬 바탕 수 (무작위 6장 중 주변이 잘 맞는 3장)
NOISE = 0.12                   # 결함마다 고정한 잠재 벡터(anchor)에 더하는 잡음 크기
FOCUS = 12                     # 재구성 손실을 거는 범위: 다시 그릴 영역 + 12px (생성 해상도 기준)
ALLOW = 12                     # 생성 때 그려도 되는 범위: 학습 바탕 과반의 다시 그릴 영역 + 12px
COVER = 0.95                   # 바탕 고르기: 다시 그릴 영역 중 생성 크롭이 덮는 비율이 이 이상인 바탕을 쓴다
STRUCTURE_DILATE = 6           # bent·missing: 원래 핀 가장자리까지 덮도록 다시 그릴 영역을 넓히는 픽셀 수
REACH = 30                     # bent·missing: 놓친 핀을 다시 찾는 범위 (change_region 결과의 바깥 30px까지)
COLOR_CLAMP, COLOR_SIGMA = 25, 16   # bent·missing: 바탕과 결함 원본의 주변 밝기 차이 보정 한도(0–255 단위), 퍼뜨리는 폭(px)
NETWORK = dict(noise_dim=128, encoder_width=128, encoder_depth=2, generator_width=1024,
               generator_depths=(0, 2, 1), heads=4, gradient_checkpointing=False)


def to_tensor(image, size):
    array = np.asarray(image.convert('RGB').resize((size, size), LANCZOS)).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1


def mask_tensor(mask, size):
    array = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), BOX)).copy()
    return torch.from_numpy(array).float().unsqueeze(0) / 255


class Source:
    """K장 결함 하나: 원본, 결함 마스크, 바탕 고르기용 창(box), 생성 크롭 창들(parts, split에서 정함)."""

    def __init__(self, row, donor, ids, defect):
        self.row, self.donor, self.ids, self.defect = row, donor, ids, defect
        self.label = TYPES.index(defect)
        self.region = ids == TYPE_ID[defect]
        if not self.region.any():
            raise ValueError(f'결함 ID가 마스크에 없음: {defect} {row["mask"]}')
        self.box = crop_box(self.region, CROP, donor.size, margin=MARGIN)
        x0, y0, x1, y1 = self.box
        crop_mask = self.region[y0:y1, x0:x1]
        self.donor_crop = donor.crop(self.box)
        self.ring = dilate(crop_mask, 10) & ~dilate(crop_mask, 2)
        self.parts, self.owner, self.allowed = [], None, None


def place(src, base, tf, rng=None, jitter=None, refine=True):
    """src 결함 마스크를 base 기판의 같은 부품 자리로 옮긴다. rng가 없으면 jitter 없이 정렬만.
    refine=False면 bent·missing의 정밀 정렬을 건너뛴다(바탕 후보를 빨리 비교할 때).
    돌아가는 값: (base 좌표 bool 마스크, 적용한 변환 dict(rot180, dx, dy), manifest condition dict)."""
    rot = bool(tf['rot180'])
    oriented = src.region[::-1, ::-1] if rot else src.region
    info = {'placement': 'aligned', 'align': {'rot180': rot, 'dx': int(tf['dx']), 'dy': int(tf['dy']),
                                              'score': round(float(tf['score']), 4)}}
    if src.defect in STRUCTURE_TYPES:        # bent·missing: jitter 대신 바탕의 부품 자리에 정밀하게 맞춘다 (refine.py)
        placed, shift = shift_mask(oriented, tf['dx'], tf['dy'])
        info.update(jitter=0, applied_shift_xy=list(shift))
        if refine:
            extra = change_region(src.donor, base, placed, info) & ~placed     # 바탕에만 있는 원래 핀: 비교에서 뺀다
            placed, info = refine_shift(src.donor, src.region, base, placed, info, ignore=dilate(extra, 6))
    else:
        j = JITTER_BY_TYPE[src.defect] if jitter is None else jitter
        jx, jy = (int(v) for v in rng.integers(-j, j + 1, 2)) if rng is not None and j else (0, 0)
        placed, shift = shift_mask(oriented, tf['dx'] + jx, tf['dy'] + jy)
        info.update(jitter=j, offset_xy=[jx, jy], applied_shift_xy=list(shift))
    dx, dy = info['applied_shift_xy']
    return placed, {'rot180': rot, 'dx': int(dx), 'dy': int(dy)}, info


def context_error(src, base, applied):
    """정렬한 바탕 크롭과 결함 원본 크롭의 결함 주변(2–10px 고리) 평균 RGB 차이. 작을수록 같은 부품 자리."""
    crop = aligned_crop(base, src.box, applied)
    error = np.abs(np.asarray(crop, np.float32) - np.asarray(src.donor_crop, np.float32))
    return float(error[src.ring].mean())


def structure_region(donor, base, placed, info):
    """bent·missing: 마스크 + 마스크에 이어진 원래 핀·부품 자리(base 좌표).
    공용 change_region(sd_lora v2와 같은 함수)에, 그 주변 REACH px 안에서 같은 방식으로 임계값만 낮춰(30) 구한 차이를
    7px closing → 4px opening 한 것을 더한다. 금속 핀이 어둡게 찍힌 바탕은 핀 양쪽 가장자리만 차이가 커서,
    change_region(임계값 45, opening 4px)만으로는 핀 몸통을 놓쳐 핀이 반만 지워진다. 핀 그림자도 같이 잡힌다."""
    region = change_region(donor, base, placed, info)
    ys, xs = np.nonzero(region)
    height, width = region.shape
    y0, y1 = max(ys.min() - REACH, 0), min(ys.max() + REACH + 1, height)
    x0, x1 = max(xs.min() - REACH, 0), min(xs.max() + REACH + 1, width)
    box = (x0, y0, x1, y1)
    blur = ImageFilter.GaussianBlur(2)
    shifted = np.asarray(aligned_donor(donor, base, info).crop(box).convert('L').filter(blur), np.int16)
    normal = np.asarray(base.crop(box).convert('L').filter(blur), np.int16)
    loose = ndimage.binary_opening(np.abs(shifted - normal) > 30, iterations=1)
    labels, _ = ndimage.label(loose)
    touching = np.unique(labels[dilate(placed[y0:y1, x0:x1], 20)])       # change_region과 같이 마스크 20px 안에 닿는 덩어리만
    edge = 16
    loose = np.pad(np.isin(labels, touching[touching > 0]), edge)
    filled = ndimage.binary_opening(ndimage.binary_closing(loose, iterations=7), iterations=4)
    region[y0:y1, x0:x1] |= filled[edge:-edge, edge:-edge]
    return region


def paint_region(src, base, placed, info, structure=True):
    """다시 그릴 영역(base 좌표). bent·missing은 structure_region을 조금 넓힌 것. 정답 마스크는 placed 그대로."""
    if structure and src.defect in STRUCTURE_TYPES:
        return dilate(structure_region(src.donor, base, placed, info), STRUCTURE_DILATE)
    return placed


def _move(region, applied, size, to_base):
    width, height = size
    ys, xs = np.nonzero(region)
    if to_base:
        if applied['rot180']:
            xs, ys = width - 1 - xs, height - 1 - ys
        xs, ys = xs + applied['dx'], ys + applied['dy']
    else:
        xs, ys = xs - applied['dx'], ys - applied['dy']
        if applied['rot180']:
            xs, ys = width - 1 - xs, height - 1 - ys
    keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    out = np.zeros((height, width), bool)
    out[ys[keep], xs[keep]] = True
    return out


def to_donor(region, applied, size):
    """base 좌표 bool 영역을 donor 좌표로 (place의 역변환: 이동을 되돌리고 180° 회전)."""
    return _move(region, applied, size, to_base=False)


def to_base(region, applied, size):
    """donor 좌표 bool 영역을 base 좌표로 (place와 같은 변환, 이미지 밖은 버린다)."""
    return _move(region, applied, size, to_base=True)


def split(src, regions):
    """생성 크롭 창(src.parts)과 그려도 되는 범위(src.allowed)를 정한다. regions = 학습 바탕들에서 구한 다시 그릴 영역(donor 좌표).
    창 하나가 SPLIT_ABOVE보다 크면 떨어진 덩어리별로 나눠(src.owner = 덩어리 번호) 각자 res 해상도로 그린다."""
    union = np.logical_or.reduce([src.region, *regions])
    size = src.donor.size
    # 생성 때 그려도 되는 곳: 학습 바탕 과반에서 다시 그릴 영역이었던 곳 + ALLOW px.
    # 바탕 한 장에서만 생긴 차이(기판 가장자리가 어긋난 띠 등)는 그리지 않고 바탕 그대로 둔다.
    common = np.sum(regions, axis=0) > len(regions) / 2
    src.allowed = dilate(common | src.region, ALLOW)
    whole = crop_box(union, CROP, size, margin=MARGIN)
    src.parts, src.owner = [whole], None
    if whole[2] - whole[0] <= SPLIT_ABOVE:
        return
    for gap in (20, 40, 80):
        labels, count = ndimage.label(dilate(union, gap))
        if count <= MAX_PARTS:
            break
    else:
        return
    if count > 1:
        src.parts = [crop_box(union & (labels == i), CROP, size, margin=MARGIN) for i in range(1, count + 1)]
        src.owner = labels


def owned(src, part, region):
    """donor 좌표 영역 중 part번째 창이 맡는 부분."""
    return region if src.owner is None else region & (src.owner == part + 1)


def color_offset(donor_crop, base_crop, inside):
    """바탕과 결함 원본의 밝기 차이를 다시 그릴 영역 안으로 부드럽게 퍼뜨린 보정값 (H, W, 3), 0–255 단위.
    영역 주변(2–14px 고리)은 change_region이 "두 사진이 비슷하다"고 본 곳이라, 거기서 (바탕 − 원본)을 재면 조명 차이다.
    고리 값을 가우시안으로 정규화 평균해 영역 안을 채운다(고리에서 먼 곳은 고리 전체의 중앙값 차이)."""
    ring = dilate(inside, 14) & ~dilate(inside, 2)
    base, donor = np.asarray(base_crop, np.float32), np.asarray(donor_crop, np.float32)
    if ring.sum() < 50:
        return np.zeros_like(base)
    overall = np.median(base[ring], axis=0) - np.median(donor[ring], axis=0)
    weight = ndimage.gaussian_filter(ring.astype(np.float32), COLOR_SIGMA)
    field = np.stack([ndimage.gaussian_filter((base[..., c] - donor[..., c]) * ring, COLOR_SIGMA)
                      for c in range(3)], axis=-1) / np.maximum(weight, 1e-6)[..., None]
    field = np.where(weight[..., None] > 1e-3, field, overall)
    return np.clip(field, -COLOR_CLAMP, COLOR_CLAMP).astype(np.float32)


def rank_bases(src, data, normals, picks):
    """picks를 결함 주변이 잘 맞는 순으로: [(오차, row, 이미지, 기판 변환), ...]."""
    ranked = []
    for index in picks:
        row = normals[int(index)]
        base, _ = data.load(row)
        tf = board_transform(src.donor, base)
        _, applied, _ = place(src, base, tf, refine=False)
        ranked.append((context_error(src, base, applied), row, base, tf))
    return sorted(ranked, key=lambda item: item[0])


def plan(src, base, tf, rng, a):
    """base에 src를 놓았을 때: (정답 마스크, 변환, condition, 다시 그릴 영역 px 수,
    donor 좌표에서 창마다 실제로 그릴 영역 리스트, 그리는 비율)."""
    placed, applied, info = place(src, base, tf, rng, a.jitter)
    paint = paint_region(src, base, placed, info, a.structure)
    paint_donor = to_donor(paint, applied, src.donor.size) & src.allowed
    inside = [owned(src, part, paint_donor)[y0:y1, x0:x1] for part, (x0, y0, x1, y1) in enumerate(src.parts)]
    covered = np.zeros_like(paint_donor)
    for (x0, y0, x1, y1), region in zip(src.parts, inside):
        covered[y0:y1, x0:x1] |= region
    total = int(paint.sum())
    return placed, applied, info, total, inside, float(covered.sum()) / max(total, 1)


def scratch_warp(src, base, applied, inside, rng, resolution):
    """Return a shared inverse transform for a generated scratch and its label."""
    if len(src.parts) != 1 or not inside.any():
        return None
    box = src.parts[0]
    normal = np.asarray(aligned_crop(base, box, applied), dtype=np.int16)
    blue = normal[..., 2] - normal[..., 0] > 30
    board = ndimage.binary_fill_holes(ndimage.binary_closing(blue, iterations=3, border_value=1))
    board = ndimage.binary_dilation(board, iterations=2)
    y, x = np.nonzero(inside)
    height, width = inside.shape
    center = np.array([x.mean() + .5, y.mean() + .5])
    _, vectors = np.linalg.eigh(np.cov(np.stack((x, y)))) if len(x) > 2 else (None, np.eye(2))
    original = Image.fromarray(inside.astype(np.uint8) * 255)
    affine = Image.Transform.AFFINE
    for _ in range(10):
        angle = float(rng.choice((-1, 1)) * rng.uniform(4, 12))
        length = float(rng.choice((rng.uniform(.76, .89), rng.uniform(1.12, 1.23))))
        thickness = float(rng.choice((rng.uniform(.78, .9), rng.uniform(1.11, 1.26))))
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        forward = rotation @ vectors @ np.diag([thickness, length]) @ vectors.T
        corners = (np.stack((x + .5, y + .5), axis=1) - center) @ forward.T + center
        if np.any(corners < 1) or np.any(corners >= np.array([width, height]) - 1):
            continue
        inverse = np.linalg.inv(forward)
        offset = center - inverse @ center
        matrix = np.column_stack((inverse, offset))
        moved = np.asarray(original.transform((width, height), affine, tuple(matrix.ravel()),
                                              Image.Resampling.NEAREST)) > 0
        if not moved.any() or not board[moved].all():
            continue
        iou = (moved & inside).sum() / (moved | inside).sum()
        if .20 <= iou <= .85:
            scale = np.diag([resolution / width, resolution / height])
            transform = np.column_stack((scale @ inverse @ np.linalg.inv(scale), scale @ offset))
            return moved, tuple(transform.ravel()), dict(
                angle_deg=angle, length_scale=length, width_scale=thickness, source_mask_iou=float(iou))
    return None


def train(crops, a, device):
    """crops = 생성 크롭마다 dict(real, normals, masks, label). PCB 하나의 K장에서 나온 크롭들로 학습."""
    config = dict(NETWORK, patch=a.res // 32)
    classifier, generator, discriminator = (m.to(device) for m in build_models(config, len(TYPES)))
    real = torch.stack([c['real'] for c in crops]).to(device)
    normals = torch.stack([torch.stack(c['normals']) for c in crops]).to(device)
    masks = torch.stack([torch.stack(c['masks']) for c in crops]).to(device)
    labels = torch.tensor([c['label'] for c in crops], device=device)
    anchors = torch.randn(len(crops), generator.noise_dim, device=device)
    steps = a.steps * len(crops)

    def low(value):
        return F.adaptive_avg_pool2d(value, (32, 32))

    opt_c = torch.optim.Adam(classifier.parameters(), lr=2e-4)
    for _ in range(a.classifier_steps):
        opt_c.zero_grad()
        F.cross_entropy(classifier(low(real)), labels).backward()
        opt_c.step()
    classifier.eval()
    with torch.no_grad():
        features = classifier.encoder(low(real))
        embeddings = torch.stack([features[labels == i].mean(0) for i in range(len(TYPES))])

    opt_g = torch.optim.Adam(generator.parameters(), lr=a.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, steps, eta_min=a.lr * 0.05)
    for step in range(1, steps + 1):
        index = torch.randint(len(crops), (a.batch,), device=device)
        which = torch.randint(normals.shape[1], (a.batch,), device=device)
        target, mask, normal = real[index], masks[index, which], normals[index, which]
        embedding = embeddings[labels[index]]
        fake = generator(anchors[index] + NOISE * torch.randn_like(anchors[index]), embedding)
        merged = fake * mask + normal * (1 - mask)
        real_merged = target * mask + normal * (1 - mask)

        opt_d.zero_grad()
        loss_d = (F.softplus(-discriminator(low(real_merged), embedding)).mean()
                  + F.softplus(discriminator(low(merged.detach()), embedding)).mean()) / 2
        loss_d.backward()
        opt_d.step()

        discriminator.requires_grad_(False)
        opt_g.zero_grad()
        # 재구성 손실은 영역보다 FOCUS px 넓게 건다: 생성할 때 바탕마다 다시 그릴 영역이 조금씩 달라지기 때문
        focus = F.max_pool2d(mask, 2 * FOCUS + 1, stride=1, padding=FOCUS)
        area = 3 * focus.sum().clamp_min(1)
        residual = fake - target
        foreground = (residual.abs() * focus).sum() / area
        loss_g = (0.25 * F.softplus(-discriminator(low(merged), embedding)).mean()
                  + 35 * foreground + 6 * residual.abs().mean())
        for axis in (-1, -2):                                   # 경계: 영역 안 1차 차분
            edge_mask = torch.minimum(focus.narrow(axis, 1, focus.shape[axis] - 1),
                                      focus.narrow(axis, 0, focus.shape[axis] - 1))
            loss_g = loss_g + a.edge_weight * (residual.diff(dim=axis).abs() * edge_mask).sum() \
                / (3 * edge_mask.sum().clamp_min(1))
        for size in (3, 7):                                     # 텍스처: 고주파 성분
            high = residual - F.avg_pool2d(residual, size, stride=1, padding=size // 2, count_include_pad=False)
            loss_g = loss_g + a.texture_weight * (high.abs() * focus).sum() / area
        loss_g.backward()
        opt_g.step()
        schedule.step()
        discriminator.requires_grad_(True)
        if step == 1 or step % 100 == 0 or step == steps:
            print(f'  step {step}/{steps} D={loss_d.item():.3f} G={loss_g.item():.3f} '
                  f'L1(영역)={foreground.item():.3f}', flush=True)
    return generator.eval(), embeddings, anchors


def run(k, seed, a):
    out_root = Path(a.out)
    if (out_root / f'K{k}_seed{seed}' / 'manifest.csv').exists() and not a.force:
        print(f'K{k}_seed{seed}: 이미 있음, 건너뜀 (다시 만들려면 --force)', flush=True)
        return None
    data = KShot(k, seed, a.data_root, a.splits)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    writer = ManifestWriter(METHOD, k, seed, out_root=out_root, force=a.force)
    for pcb in a.pcb:
        started = time.time()
        torch.manual_seed(gen_seed(METHOD, pcb, k, seed))
        normals = data.normals(pcb)
        sources = [Source(row, *data.load(row), defect) for defect in TYPES for row in data.defects(pcb, defect)]

        # 학습 준비: 결함마다 정렬 바탕 3장, 그 바탕들에서의 다시 그릴 영역, 생성 크롭 창
        prep = rng_for(METHOD, pcb, k, seed, 0)
        crops, first = [], {}
        for s_index, src in enumerate(sources):
            scored = []
            for index in prep.choice(len(normals), 2 * TRAIN_BASES, replace=False):
                base, _ = data.load(normals[int(index)])
                tf = board_transform(src.donor, base)
                scored.append((context_error(src, base, place(src, base, tf, refine=False)[1]), base, tf))
            scored.sort(key=lambda item: item[0])
            chosen = []
            for error, base, tf in scored[:TRAIN_BASES]:
                placed, applied, info = place(src, base, tf)
                chosen.append((error, base, applied, placed, info))
            regions = [to_donor(paint_region(src, base, placed, info, a.structure), applied, src.donor.size)
                       for _, base, applied, placed, info in chosen]
            split(src, regions)
            first[s_index] = len(crops)
            for part, box in enumerate(src.parts):
                x0, y0, x1, y1 = box
                crops.append(dict(
                    real=to_tensor(src.donor.crop(box), a.res), label=src.label,
                    normals=[to_tensor(aligned_crop(base, box, applied), a.res) for _, base, applied, _, _ in chosen],
                    masks=[mask_tensor(owned(src, part, region)[y0:y1, x0:x1], a.res) for region in regions]))
        print(f'{pcb} | K{k}_seed{seed} | {device} | 결함 {len(sources)}개, 생성 크롭 {len(crops)}개로 학습', flush=True)
        generator, embeddings, anchors = train(crops, a, device)
        trained = time.time()

        used = {i: set() for i in range(len(sources))}       # 같은 결함에 같은 바탕을 두 번 쓰지 않는다
        for t_index, defect in enumerate(TYPES):
            group = [i for i, s in enumerate(sources) if s.defect == defect]
            for j in range(a.per_type):
                s_index = group[j % len(group)]                # K장을 돌아가며 같은 수만큼
                src = sources[s_index]
                size = src.donor.size
                rng = rng_for(METHOD, pcb, k, seed, 1, t_index, j)
                free = np.setdiff1d(np.arange(len(normals)), sorted(used[s_index]))
                picks = rng.choice(free, min(a.base_candidates, len(free)), replace=False)
                # 주변이 잘 맞는 순으로 보다가, 다시 그릴 영역을 생성 크롭이 COVER 이상 덮는 첫 바탕을 쓴다.
                # (바탕의 핀이 유난히 길거나 기판 가장자리가 어긋나 있으면 일부만 지워져서 어색하다. 다 안 되면 가장 많이 덮는 것)
                chosen = None
                warp = None
                for rank, (error, base_row, base, tf) in enumerate(rank_bases(src, data, normals, picks)):
                    result = plan(src, base, tf, rng, a)
                    proposal = (rank, error, base_row, base, result)
                    if chosen is None or result[-1] > chosen[-1][-1]:
                        chosen = proposal
                    if result[-1] >= COVER:
                        if not (a.diverse_scratch and defect == 'scratch' and len(src.parts) == 1):
                            break
                        warp = scratch_warp(src, base, result[1], result[4][0], rng, a.res)
                        if warp is not None:
                            chosen = proposal
                            break
                rank, error, base_row, base, (placed, applied, info, paint_px, insides, cover) = chosen
                used[s_index].add(normals.index(base_row))

                if a.diverse_scratch and defect == 'scratch':
                    if warp is not None:
                        moved, _, details = warp
                        x0, y0, x1, y1 = src.parts[0]
                        mask_donor = np.zeros_like(src.region)
                        mask_donor[y0:y1, x0:x1] = moved
                        placed = to_base(mask_donor, applied, size)
                        insides = [moved]
                        paint_px, cover = int(placed.sum()), 1.0
                        info['scratch_transform'] = details
                    else:
                        info['scratch_transform'] = None

                image, offsets, boxes = base, [], []
                for box, inside in zip(src.parts, insides):
                    x0, y0, x1, y1 = box
                    if not inside.any():
                        continue
                    around = aligned_crop(image, box, applied)             # 지금까지 붙인 결과에서 이 창 자리
                    mask = mask_tensor(inside, a.res)
                    crop = first[s_index] + src.parts.index(box)
                    with torch.inference_mode():
                        latent = anchors[crop:crop + 1]
                        fake = generator(latent + NOISE * torch.randn_like(latent),
                                         embeddings[src.label:src.label + 1])[0].cpu()
                    if warp is not None:
                        fake = torch.stack([
                            torch.from_numpy(np.array(Image.fromarray(channel.numpy()).transform(
                                (a.res, a.res), Image.Transform.AFFINE, warp[1], Image.Resampling.BILINEAR)))
                            for channel in fake
                        ])
                    if a.color_match and defect in STRUCTURE_TYPES:        # 바탕 밝기에 맞춘다
                        offset = color_offset(src.donor.crop(box), around, inside)
                        field = torch.from_numpy(offset).permute(2, 0, 1)[None]
                        fake = fake + F.interpolate(field, size=fake.shape[-2:], mode='bilinear',
                                                    align_corners=False)[0] / 127.5
                        offsets.append([round(float(v), 1) for v in offset[inside].mean(axis=0)])
                    fake = fake * mask + to_tensor(around, a.res) * (1 - mask)   # 영역 밖은 바탕 그대로
                    patch = Image.fromarray(((fake.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255)
                                            .byte().numpy())
                    if applied['rot180']:
                        patch = patch.rotate(180)
                    region = np.zeros(src.region.shape, bool)
                    region[y0:y1, x0:x1] = inside
                    region = to_base(region, applied, size)
                    target_box = map_box(box, applied, base.size)
                    soft = 3 if defect in STRUCTURE_TYPES else 2          # 배경 질감끼리 만나는 경계는 조금 더 넓게 섞는다
                    image = paste_back(image, patch, target_box, dilate(region, soft), feather=soft)
                    boxes.append([int(v) for v in target_box])
                writer.add(image, placed, pcb, defect, src.row, base_row, {
                    **info, 'crop_box': boxes[0] if boxes else None, 'crop_boxes': boxes, 'mask_area': int(placed.sum()),
                    'paint_px': paint_px, 'painted_frac': round(cover, 3), 'color_offset': offsets or None,
                    'base_context_mae': round(error, 2), 'base_candidates': a.base_candidates, 'base_rank': rank,
                    'gen_res': a.res, 'train_steps': a.steps * len(crops), 'edge_weight': a.edge_weight,
                    'texture_weight': a.texture_weight, 'noise_std': NOISE,
                })
        print(f'{pcb} 완료: 학습 {trained - started:.0f}s, 생성 {time.time() - trained:.0f}s '
              f'({len(TYPES) * a.per_type}장)', flush=True)
    result = writer.close(data.splits)
    print(result, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, nargs='+', choices=(1, 2, 4, 8))
    p.add_argument('--seed', type=int, nargs='+', default=[0], choices=range(5))
    p.add_argument('--all', action='store_true', help='K {1,2,4} × seed 0 전체 (README 범위)')
    p.add_argument('--pcb', nargs='+', default=list(PCBS), choices=PCBS)
    p.add_argument('--per-type', type=int, default=40, help='PCB·종류별 장수 (기본 40 → 조건당 480장)')
    p.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: <프로젝트>/Dataset/visa/raw)')
    p.add_argument('--splits', help='기본: <프로젝트>/Dataset/visa/splits')
    p.add_argument('--out', default=str(GENERATED / METHOD))
    p.add_argument('--jitter', type=int, default=None, help='melt·scratch 위치 흔들림 (기본 30px). bent·missing은 항상 0')
    p.add_argument('--base-candidates', type=int, default=5, help='장마다 뽑는 바탕 후보 수 (주변이 가장 맞는 것 사용)')
    p.add_argument('--res', type=int, default=256, choices=(128, 256), help='생성 크롭 해상도')
    p.add_argument('--steps', type=int, default=125, help='생성 크롭 1개당 GAN 업데이트 수 (PCB당 총 업데이트 = 이 값 × 크롭 수)')
    p.add_argument('--lr', type=float, default=5e-4, help='생성기 학습률 (cosine으로 5%%까지 줄인다)')
    p.add_argument('--classifier-steps', type=int, default=60)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--edge-weight', type=float, default=14)
    p.add_argument('--texture-weight', type=float, default=18)
    p.add_argument('--no-structure', dest='structure', action='store_false',
                   help='bent·missing에서 원래 핀·부품 자리까지 다시 그리는 처리를 끈다 (sd_lora와 같은 옵션)')
    p.add_argument('--no-color-match', dest='color_match', action='store_false',
                   help='bent·missing에서 바탕 밝기에 맞추는 보정을 끈다')
    p.add_argument('--device', choices=('cpu', 'cuda'))
    p.add_argument('--diverse-scratch', action='store_true', help='긁힘 패치와 정답 마스크를 함께 변형')
    p.add_argument('--threads', type=int, default=os.cpu_count())
    p.add_argument('--force', action='store_true', help='이미 만든 조건도 다시 생성')
    a = p.parse_args()
    if a.all:
        a.k = a.k or [1, 2, 4]
    elif not a.k:
        p.error('--k 를 지정하거나 --all 을 붙이세요')
    if min(a.per_type, a.base_candidates, a.steps, a.classifier_steps, a.batch) < 1:
        p.error('per-type, base-candidates, steps, classifier-steps, batch 는 1 이상')
    if not all(np.isfinite(w) and w >= 0 for w in (a.edge_weight, a.texture_weight)) or not a.lr > 0:
        p.error('손실 가중치는 0 이상의 유한한 값, 학습률은 양수')
    torch.set_num_threads(a.threads)
    for k in a.k:
        for seed in a.seed:
            run(k, seed, a)


if __name__ == '__main__':
    main()
