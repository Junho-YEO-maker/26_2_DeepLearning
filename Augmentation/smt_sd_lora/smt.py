"""AIHub SMT용 SD 인페인팅 + LoRA에서 train·generate가 같이 쓰는 것: 경로, 종류·프롬프트, 박스 읽기, 분할 접근(누수 차단), 결함 자리.

VisA sd_lora와 다른 점
  - 이미지가 이미 512×512라 크롭·되붙이기 없이 이미지 전체를 문맥으로 인페인팅한다.
  - 마스크가 없고 박스만 있다. 인페인팅 영역 = 박스(8px 넓힘)인 직사각형.
  - 정상 이미지에도 '정상_{종류}' 박스가 있다(그 종류 결함이 생길 수 있는 자리). 결함은 바탕 정상 이미지의
    '정상_{t}' 박스 자리에, 크기는 K장 중 고른 실제 결함 박스 크기로 둔다(VisA "같은 부품 자리 ± jitter"에 해당).
  - category(사전공정 pre / 납땜공정 solder)마다 LoRA 하나, 종류는 프롬프트로 구분.
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLITS = ROOT / 'Dataset/aihub_smt/splits'
OUT = ROOT / 'Dataset/aihub_smt/generated'        # AIHub 파생물: git·공유 Drive 금지(.gitignore)
RUNS = HERE / 'runs'
BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
TEAM_SEED = 7017
RES = 512
CATEGORIES = ('pre', 'solder')
RECIPE = 'smt-v1-box'

# scripts/aihub_smt/build_index.py TYPE_CODE와 같은 대응 (라벨 이름 → 종류 코드)
TYPE_CODE = {
    'pre': {'미납': 'nm', '납부족': 'nb', '납쇼트': 'ns', '납볼': 'pb', '납좌표밀림': 'om', '납형성불량': 'mf'},
    'solder': {'미납': 'nm', '냉납': 'cc', '밀림': 'ml', '쇼트': 'sh', '오삽': 'ws', '미삽': 'ms', '역삽': 'rs',
               '뒤집힘': 'fl', '일어섬': 'st', '납볼': 'pb', '납금감핀홀': 'ph', '납고드름': 'ic'},
}
# 'sks' = 이 데이터 촬영 스타일 표시. 종류마다 "무엇이 어떻게 보이는지"를 쓴다. 정상 프롬프트에는 결함 단어를 넣지 않는다.
_PRE = 'close-up photo of sks circuit board with solder paste printed on pads, '
_SOL = 'close-up photo of sks circuit board after reflow soldering, '
PROMPT = {
    'pre': {
        'nm': _PRE + 'one pad has no solder paste, bare copper pad',
        'nb': _PRE + 'insufficient solder paste, a thin small deposit on the pad',
        'ns': _PRE + 'solder paste bridging across neighbouring pads',
        'pb': _PRE + 'small stray solder paste balls beside the pads',
        'om': _PRE + 'solder paste printed offset, shifted off the pad',
        'mf': _PRE + 'malformed solder paste deposit, smeared irregular shape',
    },
    'solder': {
        'nm': _SOL + 'a lead with no solder, unsoldered joint',
        'cc': _SOL + 'cold solder joint, dull grainy lumpy solder',
        'ml': _SOL + 'component shifted sideways off its pads',
        'sh': _SOL + 'solder bridge shorting adjacent leads',
        'ws': _SOL + 'wrong component mounted, different part body',
        'ms': _SOL + 'component missing, empty solder pads',
        'rs': _SOL + 'component mounted reversed, polarity mark on the wrong side',
        'fl': _SOL + 'component flipped upside down',
        'st': _SOL + 'chip component standing up on one end, tombstone',
        'pb': _SOL + 'small solder balls scattered near the joint',
        'ph': _SOL + 'solder joint with pinholes and voids',
        'ic': _SOL + 'solder icicle, a sharp spike of solder',
    },
}
PROMPT_NORMAL = {'pre': _PRE + 'clean, every pad evenly covered',
                 'solder': _SOL + 'clean shiny solder joints, all parts in place'}
NEGATIVE = 'blurry, cartoon, drawing, text, watermark'


def norm_type(s):
    return ''.join(ch for ch in s if ch not in ' ,/·_')


def boxes(row, cat):
    """label_json의 박스: [(정상|불량, 종류 코드, (x0, y0, x1, y1) int, 512 안으로 자름)]."""
    j = json.loads((ROOT / row['label_json']).read_text(encoding='utf-8'))
    names = {c['id']: c['name'] for c in j['categories']}
    out = []
    for an in j['annotations']:
        lab, typ = names[an['category_id']].split('_', 1)
        x, y, w, h = an['bbox']
        b = (max(0, int(x)), max(0, int(y)), min(RES, int(np.ceil(x + w))), min(RES, int(np.ceil(y + h))))
        if b[2] > b[0] and b[3] > b[1]:
            out.append((lab, TYPE_CODE[cat].get(norm_type(typ), norm_type(typ)), b))
    return out


def box_mask(bs, dilate=0):
    m = np.zeros((RES, RES), bool)
    for x0, y0, x1, y1 in bs:
        m[max(0, y0 - dilate):min(RES, y1 + dilate), max(0, x0 - dilate):min(RES, x1 + dilate)] = True
    return m


def load_rgb(rel):
    with Image.open(ROOT / rel) as im:
        return im.convert('RGB')


def rng_for(*key):
    """팀 시드 7017과 조건 키로 만든 난수열 (manifest gen_seed에도 같은 정수를 남긴다)."""
    s = zlib.crc32('|'.join(map(str, ('smt_sd_lora',) + key)).encode())
    return np.random.default_rng([TEAM_SEED, s]), s


class Split:
    """분할표 접근. 결함 소스 = train_K{k}_seed{s}, 정상 = train_normal만. test/val은 경로·해시로 막는다."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed
        tv = pd.concat([pd.read_csv(SPLITS / 'test.csv'), pd.read_csv(SPLITS / 'val_normal.csv')])
        self.forbidden, self.forbidden_sha = set(tv.image), set(tv.image_sha256)
        self.kshot = pd.read_csv(SPLITS / f'train_K{k}_seed{seed}.csv')
        self.normal = pd.read_csv(SPLITS / 'train_normal.csv')
        for df in (self.kshot, self.normal):
            assert not (df.image.isin(self.forbidden) | df.image_sha256.isin(self.forbidden_sha)).any()

    def types(self, cat):
        return sorted(self.kshot[self.kshot.category == cat].defect_type.unique())

    def defects(self, cat, t):
        """종류 t의 K장: [(row, 그 종류 불량 박스들)]."""
        rows = self.kshot[(self.kshot.category == cat) & (self.kshot.defect_type == t)]
        out = [(r, [b for lab, typ, b in boxes(r, cat) if lab == '불량' and typ == t]) for _, r in rows.iterrows()]
        assert len(out) == self.k and all(bs for _, bs in out), (cat, t, [len(bs) for _, bs in out])
        return out

    def normals(self, cat):
        """train_normal: [(row, [(종류, 정상 박스)])]. 실제 크기가 512×512가 아닌 것(pre 8장이 512×507인데
        라벨은 512×512라고 적혀 있음)은 박스 좌표를 믿을 수 없어 뺀다."""
        out = []
        for _, r in self.normal[self.normal.category == cat].iterrows():
            with Image.open(ROOT / r['image']) as im:
                if im.size != (RES, RES):
                    continue
            out.append((r, [(typ, b) for lab, typ, b in boxes(r, cat) if lab == '정상']))
        return out


def place(base_sites, t, donor_box, rng, jitter=0.05, scale=(0.9, 1.1)):
    """바탕의 '정상_t' 자리 하나에 실제 결함 박스 크기(× scale)로 둔다. 중심은 자리 크기의 ± jitter.
    돌아가는 값: (결함 박스, 고른 자리 박스) 또는 자리가 없으면 None."""
    sites = [b for typ, b in base_sites if typ == t]
    if not sites:
        return None
    sx0, sy0, sx1, sy1 = sites[int(rng.integers(len(sites)))]
    w = (donor_box[2] - donor_box[0]) * rng.uniform(*scale)
    h = (donor_box[3] - donor_box[1]) * rng.uniform(*scale)
    w, h = min(w, RES), min(h, RES)
    cx = (sx0 + sx1) / 2 + rng.uniform(-jitter, jitter) * (sx1 - sx0)
    cy = (sy0 + sy1) / 2 + rng.uniform(-jitter, jitter) * (sy1 - sy0)
    x0 = int(np.clip(cx - w / 2, 0, RES - w))
    y0 = int(np.clip(cy - h / 2, 0, RES - h))
    return (x0, y0, int(x0 + w), int(y0 + h)), (sx0, sy0, sx1, sy1)


def paste(base, gen, region, feather=3):
    """인페인팅 영역 안만 생성 결과로 (가장자리 feather px 흐림). 영역 밖은 바탕 픽셀 그대로."""
    a = Image.fromarray(region.astype(np.uint8) * 255)
    if feather:
        a = a.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(gen, base, a)
