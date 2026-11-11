"""`classic` 증강 (README 5.1절, 9절 ⑥).

VisA PCB는 같은 라인에서 고정 카메라로 찍은 사진이라, 실제로 생길 법한 작은 변화만 준다.
  - 회전 ±3°, 이동 ±2%        : 기판이 놓이는 위치·각도의 작은 차이
  - 밝기·대비 ±10%            : 조명 변화
  - 가우시안 블러(p=0.2, σ≤1) : 초점 변화
뒤집기는 넣지 않는다. 기판 전체를 뒤집으면 실제 라인에 없는 부품 배치가 되고, 시험 이미지에는 뒤집힌 사진이 없다.
(결함 패치만 뒤집는 것은 copy-paste 쪽 PATCH_AUG에서 한다.)
정상과 불량에 똑같이 적용한다(불량에만 걸면 "변환 흔적 = 불량"이라는 지름길이 생긴다).
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CLASSIC = dict(rotation_deg=3.0, translate=0.02, brightness=0.10, contrast=0.10, blur_p=0.2, blur_sigma=1.0)


def classic(arr, rng, cfg=CLASSIC):
    """uint8 (H, W, 3) → 같은 크기의 증강 이미지."""
    im = Image.fromarray(arr)
    w, h = im.size
    angle = rng.uniform(-cfg['rotation_deg'], cfg['rotation_deg'])
    tx, ty = (rng.uniform(-cfg['translate'], cfg['translate'], 2) * (w, h))
    # 회전·이동으로 생긴 빈 가장자리는 가장자리 픽셀 평균색으로 채운다(검은 테두리 지름길 방지)
    edge = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean(0).astype(int)
    im = im.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=tuple(edge))
    im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-cfg['brightness'], cfg['brightness']))
    im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-cfg['contrast'], cfg['contrast']))
    if rng.random() < cfg['blur_p']:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, cfg['blur_sigma'])))
    return np.asarray(im)
