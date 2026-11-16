"""부품 자리에 묶인 결함(bent·missing)의 위치 맞춤.

bent·missing은 다시 그릴 영역에 핀·기판 가장자리 같은 구조가 들어간다. 몇 픽셀만 어긋나도 원래 부품이 반쯤 남거나
기판 선이 이중으로 보인다. 기판 정렬(common/kshot.py board_transform)은 1/4 해상도에서 계산해 ±2px쯤 오차가 있으므로,
그 뒤 ±6px 안에서 결함 주변(2–9px 고리) 패턴이 바탕과 가장 잘 맞는 자리로 옮긴다.
"""
import numpy as np

from common.kshot import dilate, shift_mask


def refine_shift(donor, region, base, mask, info, ignore=None):
    """donor 좌표 결함 영역 region을 base에 정밀하게 맞춘다.
    mask = 기판 정렬만 한 base 좌표 마스크, info = 그때의 condition(align, applied_shift_xy).
    ignore = 비교에서 뺄 base 좌표 영역(바탕에만 있는 원래 핀 등). 돌아가는 값: (맞춘 마스크, 갱신한 info)."""
    ring = dilate(region, 9) & ~dilate(region, 2)
    rotated = info["align"]["rot180"]
    donor_pixels = np.asarray(donor)
    if rotated:
        region, ring, donor_pixels = region[::-1, ::-1], ring[::-1, ::-1], donor_pixels[::-1, ::-1]
    y, x = np.nonzero(ring)
    y, x = y[::3], x[::3]
    if len(y) == 0:
        return mask, info
    reference = donor_pixels[y, x].astype(np.float32)
    base_pixels = np.asarray(base)
    dx, dy = info["applied_shift_xy"]
    height, width = region.shape
    yy, xx = np.nonzero(region)

    best_score, best = float("inf"), (dx, dy)
    for shift_y in range(dy - 6, dy + 7):
        for shift_x in range(dx - 6, dx + 7):
            if not (0 <= xx.min() + shift_x <= xx.max() + shift_x < width
                    and 0 <= yy.min() + shift_y <= yy.max() + shift_y < height):
                continue
            by, bx = y + shift_y, x + shift_x
            valid = (by >= 0) & (by < height) & (bx >= 0) & (bx < width)
            if ignore is not None:
                valid[valid] &= ~ignore[by[valid], bx[valid]]
            if valid.sum() < 0.4 * len(y):
                continue
            error = np.abs(base_pixels[by[valid], bx[valid]] - reference[valid]).mean()
            score = error + 0.15 * (abs(shift_x - dx) + abs(shift_y - dy))
            if score < best_score:
                best_score, best = score, (shift_x, shift_y)

    aligned, applied = shift_mask(region, *best)
    return aligned, {
        **info, "applied_shift_xy": list(applied),
        "local_refinement_xy": [applied[0] - dx, applied[1] - dy],
    }
