"""생성 방법들이 같이 쓰는 K장 실험 도구 (공유 분할 읽기, 크롭, 원본에 다시 붙이기, manifest 기록·누수 검사).

모든 방법이 같은 규칙을 따른다.
  - 결함 소스: train_K{k}_seed{s}.csv 의 해당 PCB·종류 불량만
  - 정상 바탕: train_normal.csv 의 해당 PCB 정상만 (test, val_normal 은 열 수 없음)
  - 결함 위치·모양: K장 중 하나의 결함 마스크를 바탕 기판의 같은 부품 자리로 옮긴다
    (기판 정렬: 평행이동 + PCB1의 180° 회전, 그다음 종류별 jitter: bent·missing 5px, melt·scratch 30px).
    copy-paste(copypaste/generate.py)도 이 모듈의 함수를 쓴다. 방법끼리는 "마스크 안을 무엇으로 채우나"만 다르다.
  - 출력: 원본 해상도 JPEG(q75, 4:2:0, 원본과 같음) + 종류 ID 마스크 PNG + manifest.csv

Augmentation 폴더에서 `python -m {방법}.{스크립트}` 로 실행하면 `from common.kshot import ...` 가 된다.
"""
import csv
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS   # Pillow 9 미만(AnomalyDiffusion 환경)도 지원
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

ROOT = Path(__file__).resolve().parents[2]          # 저장소 루트
SPLITS = ROOT / 'Dataset/visa/splits'
RAW_PREFIX = 'Dataset/visa/raw/'
GENERATED = ROOT / 'Dataset/generated'
TYPES = ('bent', 'melt', 'scratch', 'missing')
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}
PCBS = ('pcb1', 'pcb2', 'pcb3')
JPEG_QUALITY = 75     # VisA PCB 원본과 같은 압축(IJG q75, 4:2:0). copy-paste와 같음
MANIFEST_FIELDS = ['image', 'mask', 'label', 'source_image', 'score', 'category', 'defect_type',
                   'base_normal_image', 'gen_seed', 'condition']


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gen_seed(method, pcb, k, seed):
    """팀 시드 7017에서 조건별로 파생한 정수 시드 (manifest gen_seed 열)."""
    ss = np.random.SeedSequence([7017, zlib.crc32(method.encode()), int(pcb[-1]), k, seed])
    a, b = ss.generate_state(2, np.uint32)
    return (int(a) << 31) ^ int(b)


def rng_for(method, pcb, k, seed, *extra):
    return np.random.default_rng(np.random.SeedSequence(
        [7017, zlib.crc32(method.encode()), int(pcb[-1]), k, seed, *extra]))


class KShot:
    """조건 (K, seed) 하나의 허용된 입력만 내준다."""

    def __init__(self, k, seed, data_root=None, splits_dir=None):
        self.k, self.seed = k, seed
        self.splits = Path(splits_dir) if splits_dir else SPLITS
        self.data_root = Path(data_root) if data_root else ROOT / RAW_PREFIX
        banned = read_rows(self.splits / 'test.csv') + read_rows(self.splits / 'val_normal.csv')
        self.banned = {r['image'] for r in banned} | {r['image_sha256'] for r in banned}
        pool_rows = read_rows(self.splits / 'pool.csv')
        pool = {r['image'] for r in pool_rows}
        self.k_rows = read_rows(self.splits / f'train_K{k}_seed{seed}.csv')
        assert all(r['image'] in pool for r in self.k_rows), 'K장이 pool 밖에 있음'
        self.normal_rows = read_rows(self.splits / 'train_normal.csv')
        for r in self.k_rows + self.normal_rows:
            self._guard(r)
        for pcb in PCBS:
            for t in TYPES:
                n_pool = sum(r['category'] == pcb and r['defect_type'] == t for r in pool_rows)
                assert len(self.defects(pcb, t)) == min(k, n_pool), f'{pcb} {t}: K={k}장이 아님'   # K=10이면 pcb2 bent는 8장

    def _guard(self, row):
        assert row['image'] not in self.banned and row['image_sha256'] not in self.banned, \
            f'test/val 이미지 사용 시도: {row["image"]}'

    def defects(self, pcb, defect_type):
        return [r for r in self.k_rows if r['category'] == pcb and r['defect_type'] == defect_type]

    def normals(self, pcb):
        return [r for r in self.normal_rows if r['category'] == pcb]

    def path(self, rel):
        assert rel.startswith(RAW_PREFIX), rel
        p = (self.data_root / rel[len(RAW_PREFIX):]).resolve()
        assert p.is_file(), f'원본 없음: {p} (--data-root 확인)'
        return p

    def load(self, row):
        """(RGB 이미지, 종류 ID 마스크 배열). 정상이면 마스크는 0."""
        self._guard(row)
        p = self.path(row['image'])
        assert sha(p) == row['image_sha256'], f'원본 해시 불일치: {p}'
        with Image.open(p) as im:
            image = im.convert('RGB').copy()
        ids = (np.asarray(Image.open(self.path(row['mask']))) if row['mask']
               else np.zeros(image.size[::-1], np.uint8))
        return image, ids


# ---------- 마스크·크롭 ----------

# ---------- 기판 정렬 (copy-paste와 같은 규칙. 원래 AD 팀 copy-paste 생성기 2c713be) ----------

ALIGN_SCALE = 4                     # 정렬은 1/4 크기에서 계산
JITTER_BY_TYPE = {'bent': 5, 'missing': 5, 'melt': 30, 'scratch': 30}   # 부품 위치에 묶인 결함은 작게


def _small(img):
    return np.asarray(img.resize((img.width // ALIGN_SCALE, img.height // ALIGN_SCALE),
                                 BILINEAR), np.float32)


def board_mask(img):
    """파란 기판 영역(B − R > 30). 배경(회색 카펫, 초록 천)과 흰 부품은 빠진다."""
    s = _small(img)
    return ((s[..., 2] - s[..., 0]) > 30).astype(np.float32)


def _phase_corr(a, b):
    """a를 b에 맞추는 평행이동(dy, dx)."""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    R = A.conj() * B
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    y, x = np.unravel_index(r.argmax(), r.shape)
    y = y - r.shape[0] if y > r.shape[0] // 2 else y
    x = x - r.shape[1] if x > r.shape[1] // 2 else x
    return int(y), int(x)


def board_transform(donor, target):
    """donor 좌표 → target 좌표 변환. VisA PCB1은 기판이 180° 돌아간 사진이 섞여 있어서 0°/180° 둘 다 본다.
    평행이동은 기판 마스크로 구하고, 기판 마스크는 180°에 거의 대칭이므로 회전은 기판 안 회색조 패턴(핀·글자)의
    정규화 상관(NCC)으로 고른다. 돌아가는 값: dict(rot180, dy, dx, score) — 원본 해상도 픽셀 단위."""
    sa, sb = _small(donor), _small(target)
    ma, mb = ((sa[..., 2] - sa[..., 0]) > 30), ((sb[..., 2] - sb[..., 0]) > 30)
    ga, gb = sa.mean(2), sb.mean(2)
    best = None
    for rot in (False, True):
        m, g = (ma[::-1, ::-1], ga[::-1, ::-1]) if rot else (ma, ga)
        dy, dx = _phase_corr(m.astype(np.float32), mb.astype(np.float32))
        moved = np.roll(g, (dy, dx), (0, 1))
        region = mb & np.roll(m, (dy, dx), (0, 1))
        if region.sum() < 50:
            score = -1.0
        else:
            u, v = moved[region], gb[region]
            score = float(((u - u.mean()) * (v - v.mean())).mean() / (u.std() * v.std() + 1e-6))
        if best is None or score > best['score']:
            best = dict(rot180=rot, dy=dy * ALIGN_SCALE, dx=dx * ALIGN_SCALE, score=score)
    return best


def jitter_for(defect_type):
    """종류별 위치 흔들림: bent·missing 5px, melt·scratch 30px."""
    return int(JITTER_BY_TYPE[defect_type])


def shift_mask(m, dx, dy):
    """bool 마스크를 (dx, dy)만큼 옮긴다. 결함이 이미지 밖으로 나가지 않게 이동량을 줄인다."""
    ys, xs = np.where(m)
    assert len(ys), '빈 마스크'
    h, w = m.shape
    dx = int(np.clip(dx, -xs.min(), w - 1 - xs.max()))
    dy = int(np.clip(dy, -ys.min(), h - 1 - ys.max()))
    out = np.zeros_like(m)
    out[ys + dy, xs + dx] = True
    return out, (dx, dy)


def to_base_frame(m, tf):
    """donor 좌표의 bool 마스크를 base 좌표로 (180° 회전 후 평행이동)."""
    return shift_mask(m[::-1, ::-1] if tf['rot180'] else m, tf['dx'], tf['dy'])[0]


def place_mask(donor_img, donor_ids, defect_type, base_img, rng, jitter=None):
    """K장 결함 하나의 마스크를 base 기판의 같은 부품 자리(정렬) ± 종류별 jitter로 옮긴다.
    돌아가는 값: (base 좌표 bool 마스크, manifest condition에 넣을 dict)."""
    m = donor_ids == TYPE_ID[defect_type]
    tf = board_transform(donor_img, base_img)
    j = jitter_for(defect_type) if jitter is None else jitter
    jx, jy = (int(v) for v in rng.integers(-j, j + 1, 2))
    src = m[::-1, ::-1] if tf['rot180'] else m
    out, applied = shift_mask(src, tf['dx'] + jx, tf['dy'] + jy)
    info = {'placement': 'aligned', 'jitter': j, 'offset_xy': [jx, jy], 'applied_shift_xy': list(applied),
            'align': {'rot180': bool(tf['rot180']), 'dx': int(tf['dx']), 'dy': int(tf['dy']),
                      'score': round(float(tf['score']), 4)}}
    return out, info


def jitter_mask(m, defect_type, rng, jitter=None):
    """정렬 없이(같은 좌표계 안에서) 종류별 jitter만 준다."""
    j = jitter_for(defect_type) if jitter is None else jitter
    return shift_mask(m, *(int(v) for v in rng.integers(-j, j + 1, 2)))[0]


# ---------- 위치·모양 다양화 (sd_lora v3) ----------
# K=1이면 종류당 결함이 1장뿐이라 같은 자리 ± jitter만으로는 40장이 거의 같다(K1_seed0 v2 진단, sd_lora/README.md).
# 인페인팅은 마스크 안을 새로 그리므로, 마스크 모양·위치를 바꾸면 결함 모양도 같이 바뀐다.

def _transform_blob(blob, out, f, s, r):
    """덩어리 하나(bool)를 자기 중심 기준으로 반전·크기·회전해 out에 더한다."""
    ys, xs = np.where(blob)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    patch = Image.fromarray(blob[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255)
    if f:
        patch = patch.transpose(getattr(Image, 'Transpose', Image).FLIP_LEFT_RIGHT)
    patch = patch.resize((max(1, round(patch.width * s)), max(1, round(patch.height * s))), BILINEAR)
    p = np.asarray(patch.rotate(r, resample=BILINEAR, expand=True)) > 127
    h, w = out.shape
    ph, pw = p.shape
    y0, x0 = cy - ph // 2, cx - pw // 2
    ty0, tx0, ty1, tx1 = max(0, y0), max(0, x0), min(h, y0 + ph), min(w, x0 + pw)
    if ty1 > ty0 and tx1 > tx0:
        out[ty0:ty1, tx0:tx1] |= p[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]


def augment_mask(m, rng, rot=(-30, 30), scale=(0.75, 1.35), flip=True, base_img=None, tol=0.1, tries=5):
    """bool 마스크를 좌우 반전·크기·회전한다. 결함이 여러 덩어리면(pcb1 melt는 기판 가장자리에 5개)
    덩어리마다 자기 중심 기준으로 같은 변형을 건다(전체 중심 기준이면 바깥 덩어리가 기판 밖으로 밀려난다).
    base_img가 있으면 "기판 위 비율"이 원래 마스크보다 tol 넘게 줄지 않을 때만 받고, 아니면 원래 마스크를 돌려준다
    (기판 마스크는 흰 부품을 빼므로 원래 마스크도 1.0이 아닐 수 있다). tries번 다시 뽑는다.
    돌아가는 값: (마스크, condition dict). 변형을 못 했으면 dict는 비어 있다."""
    from scipy import ndimage
    lab, n = ndimage.label(m)
    bm = board_mask(base_img) > 0 if base_img is not None else None
    floor = _on_board(m, bm) - tol if bm is not None else None
    for _ in range(tries):
        f = bool(flip and rng.random() < 0.5)
        s, r = float(rng.uniform(*scale)), float(rng.uniform(*rot))
        out = np.zeros_like(m)
        for i in range(1, n + 1):
            _transform_blob(lab == i, out, f, s, r)
        if out.any() and (bm is None or _on_board(out, bm) >= floor):
            return out, {'flip': f, 'scale': round(s, 3), 'rotation_deg': round(r, 1)}
    return m, {}


def _on_board(m, bm):
    """마스크 픽셀 중 기판 마스크(1/ALIGN_SCALE 크기) 위에 있는 비율."""
    ys, xs = np.where(m)
    return float(bm[np.minimum(ys // ALIGN_SCALE, bm.shape[0] - 1), np.minimum(xs // ALIGN_SCALE, bm.shape[1] - 1)].mean())


def board_place(m, base_img, rng, tries=50, min_on_board=0.8):
    """마스크를 base 기판(파란 영역) 위 임의 위치로 옮긴다. 결함 픽셀의 min_on_board 이상이 기판 위인 자리를
    tries번 안에 찾는다(흰 실크 글자는 기판 마스크에서 빠지므로 1.0이 아니다). 못 찾으면 (None, None)."""
    bm = board_mask(base_img) > 0
    cand_y, cand_x = np.where(bm)
    if not len(cand_y):
        return None, None
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    for _ in range(tries):
        i = int(rng.integers(len(cand_y)))
        ty, tx = int(cand_y[i]) * ALIGN_SCALE, int(cand_x[i]) * ALIGN_SCALE
        out, _ = shift_mask(m, tx - cx, ty - cy)
        if _on_board(out, bm) >= min_on_board:
            return out, {'placement': 'board', 'center_xy': [tx, ty]}
    return None, None


# ---------- 비슷한 구조 자리로 옮기기·지우기 (sd_lora v4, 11/17) ----------
# K=1이면 bent·melt·missing 40장이 불량 원본 1장의 자리(± 5–30px)에만 찍힌다(pcb2·pcb3: 40장이 3–6군데,
# 퍼짐 0.02 vs 실제 0.15–0.20, gen_realism). 바탕 정상 사진에서 원래 결함 자리와 모양이 비슷한 곳
# (같은 헤더의 다른 핀, 같은 모양의 다른 SMD 자리)을 정규화 상관으로 찾아 그중 한 곳으로 옮긴다.

def _window_sum(x, h, w):
    c = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


def ncc_map(img, tmpl):
    """정규화 상관(valid). img, tmpl: 2D float. 평평한 곳(국소 표준편차가 템플릿의 ~20% 미만)은 낮게 나온다."""
    from scipy.signal import fftconvolve
    h, w = tmpl.shape
    t = tmpl - tmpl.mean()
    tn = float(np.sqrt((t ** 2).sum())) + 1e-6
    num = fftconvolve(img, t[::-1, ::-1], mode='valid')
    s, s2 = _window_sum(img, h, w), _window_sum(img ** 2, h, w)
    var = np.maximum(s2 - s ** 2 / t.size, 0.04 * tn ** 2)
    return num / (np.sqrt(var) * tn)


def subset_blobs(m, rng, p=0.5):
    """여러 덩어리 마스크(예: pcb2 missing은 SMD 12개)에서 덩어리마다 p로 남긴다(최소 1개). 빠진 부품 수·자리가 바뀐다."""
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n <= 1:
        return m, {}
    keep = [i for i in range(1, n + 1) if rng.random() < p] or [int(rng.integers(1, n + 1))]
    return np.isin(lab, keep), {'blobs_kept': len(keep), 'blobs': int(n)}


def similar_sites(mask, base_img, scale=2, margin=24, thr=0.7, color_tol=20, max_sites=16, max_frac=0.45):
    """base 사진에서 mask 자리(bbox + margin)와 모양이 비슷한 자리들의 이동량 [(dx, dy, ncc)] (원래 자리 (0, 0) 포함).
    회색조 정규화 상관이 thr 이상이고, 창의 평균 색(RGB)이 원래 자리와 채널마다 color_tol 안인 곳만 받는다
    (회색조만 보면 회색 바닥 위 핀이 파란 기판의 흰 글자 가장자리와도 맞는다). 템플릿이 사진의 max_frac보다 크면
    원래 자리만 돌려준다."""
    W, H = base_img.size
    ys, xs = np.where(mask)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, W)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, H)
    if (x1 - x0) > max_frac * W or (y1 - y0) > max_frac * H:
        return [(0, 0, 1.0)]
    rgb = np.asarray(base_img.convert('RGB').resize((W // scale, H // scale), BILINEAR), np.float32)
    g = rgb.mean(2)
    sx0, sy0, sx1, sy1 = x0 // scale, y0 // scale, x1 // scale, y1 // scale
    r = ncc_map(g, g[sy0:sy1, sx0:sx1])
    th, tw = sy1 - sy0, sx1 - sx0
    for c in range(3):                                     # 평균 색이 다른 자리는 뺀다
        mean_c = _window_sum(rgb[..., c], th, tw) / (th * tw)
        r[np.abs(mean_c - mean_c[sy0, sx0]) > color_tol] = -1
    sites = [(0, 0, 1.0)]
    r[max(sy0 - th // 2, 0):sy0 + th // 2 + 1, max(sx0 - tw // 2, 0):sx0 + tw // 2 + 1] = -1   # 원래 자리
    while len(sites) < max_sites:
        y, x = np.unravel_index(int(r.argmax()), r.shape)
        if r[y, x] < thr:
            break
        dx, dy = (x - sx0) * scale, (y - sy0) * scale
        if xs.min() + dx >= 0 and xs.max() + dx < W and ys.min() + dy >= 0 and ys.max() + dy < H:
            sites.append((int(dx), int(dy), round(float(r[y, x]), 3)))
        r[max(y - th // 2, 0):y + th // 2 + 1, max(x - tw // 2, 0):x + tw // 2 + 1] = -1
    return sites


def relocate(mask, place, base_img, rng, whole=False, **kw):
    """mask를 similar_sites 중 한 곳으로 옮긴다.
    덩어리 하나: 통째로 옮기고 place['applied_shift_xy']에도 더해서 change_region(aligned_donor)이
                 불량 원본을 같은 자리로 옮겨 비교하게 한다. whole=True면 여러 덩어리도 통째로
                 (bent: 덩어리들이 휜 핀 하나의 조각이다).
    여러 덩어리: 덩어리마다 따로 옮긴다(전체 bbox는 너무 커서 비슷한 자리가 없다). 이때 불량 원본과의 비교는
                 자리가 맞지 않으므로 place['relocate']['per_blob'] = True로 표시해 change_region을 건너뛰게 한다."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1 or whole:
        sites = similar_sites(mask, base_img, **kw)
        dx, dy, score = sites[int(rng.integers(len(sites)))]
        if dx or dy:
            mask = shift_mask(mask, dx, dy)[0]
            sx, sy = place['applied_shift_xy']
            place['applied_shift_xy'] = [sx + dx, sy + dy]
        place['relocate'] = {'sites': len(sites), 'shift_xy': [dx, dy], 'ncc': score}
        return mask
    out, info = np.zeros_like(mask), []
    for i in range(1, n + 1):
        blob = lab == i
        sites = similar_sites(blob, base_img, **kw)
        dx, dy, score = sites[int(rng.integers(len(sites)))]
        out |= shift_mask(blob, dx, dy)[0] if (dx or dy) else blob
        info.append([len(sites), dx, dy])
    place['relocate'] = {'per_blob': True, 'blobs': info}
    return out


def donor_fill(base_img, donor_img, place, mask0, region, px=8):
    """missing 바탕 채우기: 불량 원본의 실제 "빠진 모습"(빈 패드, 핀 없는 바닥)을 옮긴 자리에 맞춰 region 안에 붙인다.
    mask0 = relocate 전 마스크. 덩어리별로 옮겼으면 덩어리마다 그 이동량으로 불량 원본을 옮겨 붙인다."""
    from scipy import ndimage
    out = np.asarray(base_img).copy()
    rl = place.get('relocate', {})
    if rl.get('per_blob'):
        lab, _ = ndimage.label(mask0)
        sx, sy = place['applied_shift_xy']
        for i, (_, dx, dy) in enumerate(rl['blobs'], 1):
            d = np.asarray(aligned_donor(donor_img, base_img, dict(place, applied_shift_xy=[sx + dx, sy + dy])))
            r = dilate(shift_mask(lab == i, dx, dy)[0], px) & region
            out[r] = d[r]
    else:
        d = np.asarray(aligned_donor(donor_img, base_img, place))
        out[region] = d[region]
    return Image.fromarray(out)


def fill_region(img, region, sigmas=(3, 6, 12, 24, 48), min_w=0.3):
    """region(bool) 안을 바깥 픽셀로 채운다(정규화 가우시안: 작은 σ부터, 바깥 가중이 min_w 넘는 곳부터).
    부품·핀을 지운 매끈한 바탕이 된다. 질감은 그 뒤 인페인팅(strength < 1)이 입힌다. PIL 이미지를 돌려준다."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img, np.float32)
    known = (~region).astype(np.float32)
    out, todo = a.copy(), region.copy()
    for s in sigmas:
        wgt = gaussian_filter(known, s)
        sel = todo & (wgt > (min_w if s != sigmas[-1] else 1e-6))
        if sel.any():
            for c in range(3):
                out[..., c][sel] = (gaussian_filter(a[..., c] * known, s)[sel] / np.maximum(wgt[sel], 1e-6))
            todo &= ~sel
        if not todo.any():
            break
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def map_box(box, tf, image_wh):
    """donor 좌표의 창 (x0,y0,x1,y1)이 base 좌표에서 차지하는 창."""
    x0, y0, x1, y1 = box
    w, h = image_wh
    if tf['rot180']:
        x0, y0, x1, y1 = w - x1, h - y1, w - x0, h - y0
    return x0 + tf['dx'], y0 + tf['dy'], x1 + tf['dx'], y1 + tf['dy']


def aligned_crop(base_img, box, tf):
    """base 이미지에서 donor 좌표 창 box에 해당하는 부분을 donor 방향으로 잘라 온다(이미지 밖은 검정)."""
    crop = base_img.crop(map_box(box, tf, base_img.size))
    return crop.rotate(180) if tf['rot180'] else crop


def crop_box(mask, size, image_wh, rng=None, max_offset=0, margin=16):
    """결함을 담는 정사각형 창. 기본 size×size이고, 결함(+margin)이 더 크면 창을 키운다
    (쓰는 쪽에서 size로 줄여 모델에 넣는다). rng가 있으면 결함이 창 안에 남는 범위에서 위치를 흔든다."""
    ys, xs = np.where(mask)
    w, h = image_wh
    extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 + 2 * margin
    size = int(min(max(size, extent), w, h))
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0, y0 = cx - size // 2, cy - size // 2
    if rng is not None and max_offset:
        x0 += int(rng.integers(-max_offset, max_offset + 1))
        y0 += int(rng.integers(-max_offset, max_offset + 1))
    # 결함이 창 밖으로 나가지 않게, 그다음 이미지 밖으로 나가지 않게
    x0 = np.clip(x0, xs.max() + margin - size + 1, xs.min() - margin)
    y0 = np.clip(y0, ys.max() + margin - size + 1, ys.min() - margin)
    x0 = int(np.clip(x0, 0, w - size))
    y0 = int(np.clip(y0, 0, h - size))
    return x0, y0, x0 + size, y0 + size


def dilate(mask, px):
    if px <= 0:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(im) > 0


def paste_back(base, generated_crop, box, region, feather=4):
    """생성한 크롭을 원본 해상도 바탕의 box 위치에, region(전체 해상도 bool) 안만 부드럽게 붙인다.
    box가 이미지 밖으로 조금 나가도 된다(겹치는 부분만 붙인다)."""
    x0, y0, x1, y1 = box
    w, h = base.size
    crop = np.asarray(generated_crop.convert('RGB').resize((x1 - x0, y1 - y0), LANCZOS), np.float32)
    padded = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    padded[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = region[cy0:cy1, cx0:cx1] * 255
    alpha = Image.fromarray(padded)
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    a = (np.asarray(alpha, np.float32)[..., None] / 255)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    crop = crop[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    out = np.asarray(base).copy()
    roi = out[cy0:cy1, cx0:cx1].astype(np.float32)
    out[cy0:cy1, cx0:cx1] = np.rint(roi * (1 - a) + crop * a).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ---------- 구조 결함 (bent·missing): 마스크 밖에 남는 원래 부품 ----------
# VisA 주석은 휜 핀 토막·빠진 자리만 덮는다. 정상 바탕에는 그 핀이 곧게(또는 그대로) 있어서
# 마스크 안만 다시 그리면 원래 핀이 마스크 밖에 남아 "다리가 하나 더" 생긴다.
# 실제 불량과 바탕이 크게 다른 곳 중 마스크에 이어진 부분을 인페인팅 영역에 더해 원래 핀도 지운다.
# 출력 마스크(정답 라벨)는 그대로 VisA 주석 모양을 쓴다.
STRUCTURE_TYPES = ('bent', 'missing')


def aligned_donor(donor_img, base_img, info):
    """donor 이미지를 place_mask가 쓴 변환(180° 회전 + 실제 적용된 이동)으로 base 좌표에 옮긴다."""
    im = donor_img.rotate(180) if info['align']['rot180'] else donor_img
    dx, dy = info['applied_shift_xy']
    return im.transform(base_img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=BILINEAR)


def change_region(donor_img, base_img, mask, info, pad=200, thresh=45, blur=2, open_px=4, touch=20):
    """bool (base 좌표): mask ∪ (실제 불량과 바탕이 크게 다른 곳 중 mask에 이어진 부분, mask 주변 pad 안)."""
    from scipy import ndimage
    d = np.asarray(aligned_donor(donor_img, base_img, info).convert('L').filter(ImageFilter.GaussianBlur(blur)), np.int16)
    b = np.asarray(base_img.convert('L').filter(ImageFilter.GaussianBlur(blur)), np.int16)
    h, w = mask.shape
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    diff = np.abs(d - b)[y0:y1, x0:x1] > thresh
    if open_px:
        diff = ndimage.binary_opening(diff, iterations=open_px)      # 기판끼리 조금씩 다른 가장자리·테두리의 얇은 띠(~8px 미만) 제거, 핀(~20px)은 남는다
    m = mask[y0:y1, x0:x1]
    lab, _ = ndimage.label(diff)
    ids = np.unique(lab[dilate(m, touch)])           # 마스크에서 touch px 안에 닿는 차이 덩어리 (핀 뿌리가 헤더에 가려져도)
    out = mask.copy()
    out[y0:y1, x0:x1] |= np.isin(lab, ids[ids > 0])
    return out


# ---------- 출력 ----------

def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


class ManifestWriter:
    """Dataset/generated/{method}/K{k}_seed{s}/ 에 이미지·마스크·manifest를 쓴다."""

    def __init__(self, method, k, seed, out_root=None, force=False):
        self.method, self.k, self.seed = method, k, seed
        self.dir = Path(out_root or GENERATED / method) / f'K{k}_seed{seed}'
        self.manifest = self.dir / 'manifest.csv'
        if self.manifest.exists() and not force:
            raise FileExistsError(f'이미 있음: {self.manifest} (다시 만들려면 --force)')
        (self.dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.dir / 'masks').mkdir(exist_ok=True)
        self.rows = []

    def add(self, image, defect_mask, pcb, defect_type, source_row, base_row, condition, score=''):
        i = sum(r['category'] == pcb and r['defect_type'] == defect_type for r in self.rows)
        stem = f'{pcb}_{defect_type}_{i:03d}'
        ip, mp = self.dir / 'images' / f'{stem}.jpg', self.dir / 'masks' / f'{stem}.png'
        image.save(ip, quality=JPEG_QUALITY, subsampling=2)
        Image.fromarray(defect_mask.astype(np.uint8) * TYPE_ID[defect_type]).save(mp)
        self.rows.append(dict(image=rel(ip), mask=rel(mp), label=1, source_image=source_row['image'],
                              score=score, category=pcb, defect_type=defect_type,
                              base_normal_image=base_row['image'],
                              gen_seed=gen_seed(self.method, pcb, self.k, self.seed),
                              condition=json.dumps(condition, sort_keys=True)))

    def close(self, splits_dir=None):
        tmp = self.manifest.with_suffix('.tmp')
        with tmp.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        tmp.replace(self.manifest)
        return check_manifest(self.manifest, splits_dir)


def check_manifest(manifest, splits_dir=None):
    """AD README 5.2절 4번 누수 검사."""
    manifest = Path(manifest)
    k, seed = manifest.parent.name[1:].split('_seed')
    splits = Path(splits_dir) if splits_dir else SPLITS
    banned = {r['image'] for r in read_rows(splits / 'test.csv') + read_rows(splits / 'val_normal.csv')}
    k_rows = {(r['category'], r['image']) for r in read_rows(splits / f'train_K{k}_seed{seed}.csv')}
    normals = {(r['category'], r['image']) for r in read_rows(splits / 'train_normal.csv')}
    rows = read_rows(manifest)
    assert rows, '빈 manifest'
    for r in rows:
        assert r['source_image'] not in banned and r['base_normal_image'] not in banned, f'누수: {r["image"]}'
        assert (r['category'], r['source_image']) in k_rows, f'K장 밖 소스: {r["source_image"]}'
        assert (r['category'], r['base_normal_image']) in normals, f'train_normal 밖 바탕: {r["base_normal_image"]}'
    return {'status': 'PASS', 'manifest': str(manifest), 'rows': len(rows)}
