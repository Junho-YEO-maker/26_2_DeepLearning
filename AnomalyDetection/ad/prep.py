"""임베딩 전에 이미지를 맞추는 전처리 (촬영 조건 지름길 제거, 11/16).

VisA PCB 불량 사진은 정상 사진과 촬영 조건이 다르다(정상 train_normal 40장 vs 불량 pool 40장 실측):
  pcb1  정상의 72%가 기준 정상과 180° 돌아가 있는데 불량은 0%, 가로 위치 평균 −67px vs −9px
  pcb2  불량이 더 밝고(기판 RGB 36/98/134 → 40/107/145) 대비·선명도가 높다(+8%, +15%)
  pcb3  불량이 선명도 +9%, 가로 위치 +10px
이 차이만으로 AD가 불량을 맞힌다(defect_removal_check: 결함을 지운 불량 사진도 DevNet이 57%를 불량 판정).

prep 종류
  none       원본을 res로 줄이기만 한다 (run.py와 같음)
  photonorm  + 이미지마다 채널 평균을 그 PCB train_normal 평균으로 (shortcut_check와 같음)
  canon      기판 정렬(180° 회전 + 평행이동, 기준 = 그 PCB train_normal 첫 장, common.kshot.board_transform)
             → res로 줄이기 → 이미지마다 채널 평균·표준편차를 정렬된 train_normal 평균값으로 맞춤.
             정렬로 드러난 가장자리는 그 사진의 테두리 색(중앙값)으로 채운다(검은 띠가 새 단서가 되지 않게).
라벨·시험셋 정보는 쓰지 않는다(이미지마다 같은 규칙). 원본(raw)의 정렬 결과는 Dataset/cache/canon{res}/에 캐시.
"""
import hashlib
import sys

import numpy as np
from PIL import Image

from .data import RAW_PREFIX, ROOT
from .features import embed, to_tensor

sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import BILINEAR, board_transform  # noqa: E402

PREPS = ('none', 'photonorm', 'canon', 'std')


def _border_color(a, w=10):
    ring = np.concatenate([a[:w].reshape(-1, 3), a[-w:].reshape(-1, 3), a[:, :w].reshape(-1, 3), a[:, -w:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(ring, 0))


class Canon:
    def __init__(self, data, pcb, res=512):
        self.data, self.res = data, res
        nl = data.train_normal(pcb)
        with Image.open(data.path(nl[0].image)) as im:
            self.ref = im.convert('RGB').copy()
        self.cache = ROOT / 'Dataset/cache' / f'canon{res}' / pcb
        self.cache.mkdir(parents=True, exist_ok=True)
        A = np.stack([self.item(x.image) for x in nl]).astype(np.float32)
        m, s = A.mean((1, 2)), A.std((1, 2))          # 이미지별 채널 평균·표준편차
        self.mu, self.sd = m.mean(0), s.mean(0)

    def align(self, img):
        img = img.convert('RGB')
        tf = board_transform(img, self.ref)
        im = img.rotate(180) if tf['rot180'] else img
        fill = _border_color(np.asarray(im))
        out = im.transform(self.ref.size, Image.AFFINE, (1, 0, -tf['dx'], 0, 1, -tf['dy']), resample=BILINEAR,
                           fillcolor=fill)
        return np.asarray(out.resize((self.res, self.res), BILINEAR))

    def item(self, rel):
        raw = rel.startswith(RAW_PREFIX)
        cp = self.cache / (hashlib.sha1(rel.encode()).hexdigest()[:16] + '.npy')
        if raw and cp.exists():
            return np.load(cp)
        with Image.open(self.data.path(rel)) as im:
            a = self.align(im)
        if raw:
            np.save(cp, a)
        return a

    def fix(self, a):
        a = a.astype(np.float32)
        z = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6)
        return np.clip(z * self.sd + self.mu, 0, 255).astype(np.uint8)


class Std(Canon):
    """canon 정렬 뒤 남는 촬영 차이까지 맞춘다 (capture_probe: 조명 기울기, 선명도·잔결, 채도).
    정렬 → 조명 평탄화(16×16 칸 평균 밝기로 나눔) → YCbCr 채널별 히스토그램 맞춤(평탄화한 train_normal 전체 분포)
    → 선명도 맞춤(목표보다 선명하면 가우시안 블러, 목표 = 맞춤 뒤 train_normal 선명도 5% 분위).
    통계는 train_normal만으로 정한다(라벨·시험셋 정보 없음)."""
    SIGMAS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.4)

    def __init__(self, data, pcb, res=512):
        self.data, self.res = data, res
        nl = data.train_normal(pcb)
        with Image.open(data.path(nl[0].image)) as im:
            self.ref = im.convert('RGB').copy()
        self.cache = ROOT / 'Dataset/cache' / f'canon{res}' / pcb
        self.cache.mkdir(parents=True, exist_ok=True)
        flat = [self._flatten(self.item(x.image)) for x in nl[::5]]     # 통계는 train_normal 100장으로
        ycc = np.stack([self._ycc(f) for f in flat])
        self.qs = np.linspace(0, 1, 257)
        self.ref_q = np.stack([np.quantile(ycc[..., c], self.qs) for c in range(3)])
        self.target = float(np.percentile([self._sharp(self._match(f)) for f in flat], 5))

    @staticmethod
    def _gray(f):
        return f.mean(2)

    def _sharp(self, f):
        g = self._gray(f)
        return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())

    def _flatten(self, a):
        """조명 평탄화: 16×16 칸 평균(큰 규모 밝기)을 부드럽게 늘려 나눈다."""
        f = a.astype(np.float32)
        n = f.shape[0] // 16
        low16 = f.reshape(16, n, 16, n, 3).mean((1, 3))
        low = np.stack([np.asarray(Image.fromarray(low16[..., c], 'F').resize(f.shape[1::-1], Image.BICUBIC))
                        for c in range(3)], 2)
        return f / (np.maximum(low, 1.0)) * low16.mean((0, 1))

    def _deblur(self, f):
        from scipy.ndimage import gaussian_filter
        for s in self.SIGMAS:
            g = f if s == 0 else np.stack([gaussian_filter(f[..., c], s) for c in range(3)], 2)
            if self._sharp(g) <= self.target:
                return g
        return g

    @staticmethod
    def _ycc(f):
        return np.asarray(Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert('YCbCr'), np.float32)

    def _match(self, f):
        """밝기(Y)·색차(Cb, Cr) 채널마다 히스토그램을 train_normal 분포에 맞춘다(채도도 함께 맞춰짐). RGB float로 돌려준다."""
        y = self._ycc(f)
        out = np.empty_like(y)
        for c in range(3):
            out[..., c] = np.interp(y[..., c], np.quantile(y[..., c], self.qs), self.ref_q[c])
        rgb = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'YCbCr').convert('RGB')
        return np.asarray(rgb, np.float32)

    def fix(self, a):
        return np.clip(self._deblur(self._match(self._flatten(a))), 0, 255).astype(np.uint8)


def embedder(data, ex, dev, pcb, res=512, prep='photonorm', bs=16):
    """(items → 임베딩, PIL 이미지 목록 → 임베딩) 두 함수를 돌려준다. 같은 PCB 안의 모든 이미지에 같은 전처리."""
    assert prep in PREPS, prep
    if prep in ('canon', 'std'):
        c = (Std if prep == 'std' else Canon)(data, pcb, res)
        from_item, from_img, fix = c.item, c.align, c.fix
    else:
        from_item = lambda rel: data.load(rel, res)                                          # noqa: E731
        from_img = lambda img: np.asarray(img.convert('RGB').resize((res, res), BILINEAR))   # noqa: E731
        if prep == 'photonorm':
            ref = np.mean([data.load(x.image, res).astype(np.float32).mean((0, 1)) for x in data.train_normal(pcb)], 0)
            fix = lambda a: np.clip(a.astype(np.float32) - a.astype(np.float32).mean((0, 1)) + ref, 0, 255).astype(np.uint8)  # noqa: E731
        else:
            fix = lambda a: a                                                                 # noqa: E731

    def run(arrs):
        out = []
        for i in range(0, len(arrs), bs):
            out.append(embed(ex(to_tensor(np.stack([fix(a) for a in arrs[i:i + bs]])).to(dev))).float().cpu().numpy())
        return np.concatenate(out)

    return (lambda items: run([from_item(it.image) for it in items]),
            lambda imgs: run([from_img(im) for im in imgs]))
