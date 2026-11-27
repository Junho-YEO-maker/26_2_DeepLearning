"""15분 발표 pptx (v4). 흰 바탕 + 그림자 카드 + 파랑 한 가지 강조색, 짧은 개조식 문장.
내용은 v3와 같고, 맨 앞에 "왜 이 문제인가" 4장을 더했다. v3에서 부록으로 뺀 장(기존 모델 구조, 생성 모델 구조 2장,
합성 시행착오)과 줄였던 내용(RQ2 실제 8장 · 휜 핀, SMT 합성 자리 고침, RQ2 실험표)은 본편으로 되돌렸다.
합성 비율 실험(run_smt_ratio.sh) 결과를 RQ3에 넣었다.

    python ppt_build/build_deck_v4.py [--preview 미리보기폴더] [슬라이드번호,...]
결과: PCB_AD_presentation_v4.pptx (저장소 루트), 미리보기 ppt_build/preview_v4/slide-NN.png
먼저 python ppt_build/make_dataset_figures.py 로 VisA 예시 사진과 장수(ppt_figures/dataset_counts.json)를 만든다.

스타일
  - 바탕은 흰색. 왼쪽 위 작은 파랑 글씨 = 장 이름, 오른쪽 위 파랑 네모 = 쪽 번호. 제목은 "주제 - 한 줄 요점"
  - 내용은 그림자 있는 흰 카드에. 카드 안은 파랑 숫자 + 굵은 소제목 + 한 줄 설명(개조식)
  - 색: 파랑 한 가지로 강조, 나빠진 값만 주황. 표는 머리 행 아래 굵은 선, 칸 채우기 없음
  - 부록(deck_v4_appendix.py)은 v3 부록과 같은 배치에 색만 바꿨다
용어는 build_deck_v3.py 머리말과 같다 (놓친 불량, 결함 바꾼 시험, 밝기·색 바꾸기, 사진 단위 모델, 위치 모델, 합성, 반복).
글꼴: Pretendard (발표 PC에 설치 필요). 미리보기는 PRETENDARD_DIR(기본 ~/.local/share/fonts/pretendard)로 그린다.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deck_lib  # noqa: E402
from pptx.enum.chart import XL_LABEL_POSITION, XL_LEGEND_POSITION  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Pt  # noqa: E402

_fd = Path(os.environ.get('PRETENDARD_DIR', Path.home() / '.local/share/fonts/pretendard'))
deck_lib.FONT = 'Pretendard'
if (_fd / 'Pretendard-Regular.otf').exists():
    deck_lib.PREVIEW_FONTS = (str(_fd / 'Pretendard-Regular.otf'), str(_fd / 'Pretendard-Bold.otf'))

# ---------------------------------------------------------------- 색 (파랑 하나 + 회색, 나빠짐만 주황)
ACC = '3B5BDB'      # 강조 파랑
ACC_M = '748FFC'
ACC_L = '91A7FF'    # 연한 파랑 (두 번째 계열)
ACC_T = 'EDF2FF'    # 아주 연한 파랑 (띠 · 칩 바탕)
NAVY = '1C2B57'     # 진한 남색
INK = '212529'      # 제목 · 본문
BODY = '343A40'
MUTED = '868E96'    # 설명 글자
LINE = 'E9ECEF'
GRAY_F = 'F1F3F5'   # 고정된 부분 (도식)
GRAYS = 'ADB5BD'    # '실제만' 막대
BAD = 'E8590C'      # 나빠진 값
BAD_T = 'FFF4E6'
WHITE = 'FFFFFF'
# deck_lib 기본색을 v4 색으로 (부록과 차트 기준선도 따라온다)
for k_, v_ in dict(INK=INK, MUTED=MUTED, LINE=LINE, COPPER='495057', COPPER_T=BAD_T, GREEN=ACC, GREEN_M=ACC_M, TINT=ACC_T,
                   GRAY_T=GRAY_F, GRAY_L=GRAYS, GRID='EEF0F3').items():
    setattr(deck_lib, k_, v_)
deck_lib.TITLE_Y, deck_lib.TITLE_H = 0.72, 0.75
deck_lib.PAGE_NUM = None
deck_lib.LEGEND_POS = 'top'
from deck_lib import Deck, Para  # noqa: E402

X0, XW = 0.55, 12.23              # 내용 왼쪽 끝 · 폭 (오른쪽 끝 12.78)
CY, CH = 1.85, 5.0                # 카드 위 · 높이 (기본, 아래 끝 6.85)


def _badge(self, num, label):
    """v4 머리: 왼쪽 위 장 이름(파랑), 오른쪽 위 쪽 번호 네모."""
    self.text(X0, 0.3, 8, 0.3, label, size=11, color=ACC, bold=True, anchor='m')
    self.box(12.43, 0.29, 0.32, 0.32, str(self.number), fill=ACC, size=10, color=WHITE, bold=True, radius=0.07, margin=(0, 0))


_table = deck_lib.Slide.table


def _plain_table(self, *a, **k):
    """표 기본: 머리 행 채우기 없음, 아래 굵은 선."""
    k.setdefault('header_fill', WHITE)
    k.setdefault('header_color', INK)
    k.setdefault('header_line', INK)
    return _table(self, *a, **k)


deck_lib.Slide.badge = _badge
deck_lib.Slide.table = _plain_table
import deck_v4_appendix  # noqa: E402  (deck_lib 색을 바꾼 뒤에 불러야 부록도 새 색을 쓴다)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'ppt_figures'
OUT = ROOT / 'PCB_AD_presentation_v4.pptx'                  # 공개본 (git). SMT 사진 칸은 비워 둔다
OUT_LOCAL = ROOT / 'PCB_AD_presentation_v4_local.pptx'      # 발표용: SMT 사진 포함 (AIHub 약관상 git 제외)
SMT_FIG = {(c, k): FIG / 'local_only' / f'smt_{c}_{k}.jpg' for c in ('pre', 'solder') for k in ('normal', 'defect')}
HAVE_SMT = all(p.exists() for p in SMT_FIG.values())         # python ppt_build/make_smt_figures.py 로 만든다
PHOTO_CELLS = []                                             # (슬라이드, 칸): 공개본에서 SMT 사진 대신 자리 표시를 그릴 곳
APP_VISA, APP_SMT = 15, 19                                   # 데이터 부록 첫 쪽 번호 (맨 아래에서 맞는지 확인)
VISA = json.loads((FIG / 'dataset_counts.json').read_text(encoding='utf-8'))['visa']
GRADCAM = FIG / 's12_s14_gradcam_vs_seg_map.jpg'
CAM_ORIG, CAM_DEVNET, CAM_SEG = (66, 127, 300, 362), (340, 127, 575, 362), (890, 127, 1124, 362)
NEG = '1F6FA8'
FIXED = dict(fill=GRAY_F, line=None)          # 도식: 고정된 부분
TRAIN = dict(fill=ACC_T, line=None)           # 도식: 학습하는 부분
IO = dict(fill=WHITE, line='DEE2E6')          # 도식: 입력 · 출력
LOSSB = dict(fill=WHITE, line=ACC, dash=True)  # 도식: 손실


def smt_fnr(cat):
    """SMT 위치 모델 결과 (2000 step, 반복 0–2): {(K, cond): 놓친 불량 평균}. 반복 3번이 다 있는 칸만."""
    acc = defaultdict(list)
    with open(ROOT / f'AnomalyDetection/results/smt/seg_ad_{cat}.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['steps'] == '2000' and r['seed'] in ('0', '1', '2'):
                acc[(int(r['K']), r['cond'])].append(float(r['fnr']))
    return {k: sum(v) / len(v) for k, v in acc.items() if len(v) == 3}


SMT = {c: smt_fnr(c) for c in ('pre', 'solder')}
GEN = {'SD+LoRA': 'sd_lora_fix', 'cTransGAN': 'ctransgan'}


def sv(cat, k, cond):
    return SMT[cat].get((k, cond))


def pct(v, nd=1):
    return '실험 중' if v is None else f'{v * 100:.{nd}f}%'


# ---------------------------------------------------------------- 스타일 도구
d = Deck()
SECTION = ['']


def new(title, notes=None, size=26):
    return d.new(title, SECTION[0], None, notes=notes, title_size=size)


def section(name, label):
    d.section(name)
    SECTION[0] = label


def card(s, x, y, w, h):
    s.box(x, y, w, h, '', fill=WHITE, radius=0.14, shadow=True)


def ctitle(s, x, y, w, text, sub=None, color=INK, size=14):
    """카드 제목 (+ 회색 부제). 다음 내용이 시작할 y를 돌려준다."""
    s.text(x, y, w, 0.32, text, size=size, color=color, bold=True, anchor='m')
    if sub:
        s.text(x, y + 0.34, w, 0.3, sub, size=11, color=MUTED, anchor='m')
        return y + 0.72
    return y + 0.45


def item(s, x, y, w, n, head, body=None, nsize=18, hsize=15, bsize=12.5, gap=0.4, bh=0.7):
    """파랑 숫자 + 굵은 소제목 + 설명. 쓴 높이를 돌려준다."""
    s.text(x, y - 0.03, 0.45, 0.42, str(n), size=nsize, color=ACC, bold=True)
    s.text(x + 0.45, y, w - 0.45, 0.36, head, size=hsize, color=INK, bold=True, anchor='m')
    if body:
        s.text(x + 0.45, y + gap, w - 0.45, bh, body, size=bsize, color=BODY)
    return gap + (bh if body else 0)


def items(s, x, y, w, lst, step=None, **kw):
    for i, (n, h, b) in enumerate(lst):
        used = item(s, x, y, w, n, h, b, **kw)
        y += (step if step else used + 0.12)
    return y


def stat(s, x, y, w, label, value, color=INK, vsize=28, lsize=12):
    s.text(x, y, w, 0.3, label, size=lsize, color=INK, bold=True)
    s.text(x, y + 0.32, w, 0.6, value, size=vsize, color=color, bold=True)


def vline(s, x, y1, y2):
    s.line(x, y1, x, y2, color=LINE, width=1, arrow=False)


def hline(s, x1, x2, y):
    s.line(x1, y, x2, y, color=LINE, width=1, arrow=False)


def arrow(s, x1, y1, x2, y2):
    s.line(x1, y1, x2, y2, color=MUTED, width=1.5)


def bullets(s, x, y, w, h, lst, size=13, after=8, color=BODY):
    s.text(x, y, w, h, [Para(t, size, color, bullet=True, after=after) for t in lst])


def table(s, x, y, col_w, rows, size=13, row_h=0.46, colors=None, align=None, header_h=None, fills=None):
    return s.table(x, y, col_w, rows, size=size, row_h=row_h, colors=colors, align=align, header_h=header_h, fills=fills)


def pct_axis(ch):
    ch.value_axis.tick_labels.number_format = '0%'
    ch.value_axis.tick_labels.number_format_is_linked = False


def quiet(ch):
    """차트 글자를 작고 회색으로, 범례는 위."""
    for ax in (ch.value_axis, ch.category_axis):
        ax.tick_labels.font.size = Pt(11)
        ax.tick_labels.font.color.rgb = deck_lib.rgb(MUTED)
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.font.size = Pt(11)
        ch.legend.font.color.rgb = deck_lib.rgb(BODY)


def labels(ch, fmt='0.0%', size=11):
    for ser in ch.plots[0].series:
        dl = ser.data_labels
        dl.show_value = True
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size, dl.font.bold = Pt(size), True
        dl.font.color.rgb = deck_lib.rgb(INK)


def fit_image(s, path, x, y, w, h, pad=0.06):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ww = min(w - 2 * pad, (h - 2 * pad) * iw / ih)
    s.image(path, x + (w - ww) / 2, y + (h - ww * ih / iw) / 2, w=ww)


# ================================================================ 표지 · 목차
d.section('표지')
s = d.new('적은 불량 사진을 최대한 활용하는\nPCB 이상탐지', title_size=44,
          notes='안녕하십니까. 적은 불량 사진을 최대한 활용하는 PCB 이상탐지를 발표하겠습니다. '
                '검사 현장에는 정상 사진이 대부분이고 불량 사진은 몇 장씩만 쌓이는데, 이 몇 장을 어떻게 써야 결함을 제대로 보는 모델이 되는지가 주제입니다.')
t = d.slides[-1].pslide.shapes.title
t.top, t.height = int(2.05 * 914400), int(1.7 * 914400)
for p in t.text_frame.paragraphs:
    p.alignment = PP_ALIGN.CENTER
s.ops[0] = ('text', 0.6, 2.05, 12.13, 1.7, [Para(pp.text, 44, INK, True, align='c', after=0) for pp in s.ops[0][5]]) + s.ops[0][6:]
s.text(X0, 0.3, 6, 0.45, '2026-2 딥러닝\n팀 프로젝트', size=10, color=ACC, bold=True)
s.box(12.43, 0.29, 0.32, 0.32, '', fill=ACC, radius=0.07)
s.text(1.5, 3.95, 10.33, 0.4, '현장에 조금씩 쌓이는 불량 사진, 어떻게 써야 결함을 보는 모델이 될까', size=16, color=MUTED, align='c')
s.box(3.67, 4.6, 6.0, 0.42, 'VisA PCB · AIHub SMT   |   cTransGAN · SD+LoRA · 위치 모델', fill=ACC, size=13, color=WHITE,
      bold=True, radius=0.21)

section('목차', '목차')
s = new('목차', notes='순서는 이렇습니다. 먼저 왜 이 문제를 풀어야 하는지, 그리고 처음 시도와 거기서 만난 함정을 말씀드립니다. '
                    '이어서 방법을 바꾼 뒤 세 가지 질문에 차례로 답하고, 마지막으로 의의와 한계를 정리하겠습니다.')
parts = [('왜 이 문제인가', ['검사와 놓친 불량', '불량은 늘 적다', '기존 방법의 빈틈', '우리가 풀 문제']),
         ('데이터와 첫 시도', ['VisA PCB', '기존 모델', '시도', '함정: 배경을 본다']),
         ('방법 · RQ1', ['방법을 바꾸다', '결함 바꾼 시험', '생성 모델 두 가지', '위치 모델', 'RQ1 결과']),
         ('RQ2 합성은 얼마나', ['질문과 실험', '합성 장수', '실제 몇 장 값어치']),
         ('RQ3 실제 라인', ['AIHub SMT', '결과', '실제 몇 장', '합성 비율', '왜 대신 못 하나']),
         ('정리', ['의의', '현장 순서', '한계 · 다음 실험'])]
cw = (XW - 5 * 0.18) / 6
for i, (h, subs) in enumerate(parts):
    x = X0 + i * (cw + 0.18)
    card(s, x, CY, cw, 3.7)
    s.text(x + 0.22, CY + 0.28, 1, 0.5, str(i + 1), size=24, color=ACC, bold=True)
    s.text(x + 0.22, CY + 0.88, cw - 0.35, 0.7, h, size=15, color=INK, bold=True)
    bullets(s, x + 0.22, CY + 1.55, cw - 0.32, 2.0, subs, size=12, after=5)

# ================================================================ 1. 왜 이 문제인가
section('1. 왜 이 문제인가', '1  왜 이 문제인가')
s = new('배경 - PCB 검사에서 불량을 놓치면 비용이 뒤로 갈수록 커진다',
        notes='먼저 왜 이 문제를 풀려고 했는지입니다. PCB는 거의 모든 전자 제품의 바탕이라, 결함 하나가 제품 고장으로 이어집니다. '
              '그래서 라인에서 사진을 찍어 정상과 불량을 자동으로 판정합니다. 여기서 불량을 놓치면 다음 공정이나 출하 뒤에 발견되고, 늦게 발견될수록 비용이 커집니다. '
              '반대로 정상을 너무 많이 불량으로 거르면 재검 인력과 시간이 낭비됩니다. '
              '그래서 저희는 정상 100장 중 1장만 잘못 거르도록 기준을 맞추고, 그때 놓친 불량의 비율로 성능을 잽니다.')
card(s, X0, CY, 5.6, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 5.0, '라인에서 찍은 사진으로 판정한다', 'VisA PCB2. 왼쪽 정상, 오른쪽 불량 (빨간 상자 = 긁힘)')
s.image(FIG / 's03_defect_size_resolution.png', X0 + 0.3, y + 0.1, w=5.0, crop=(15, 45, 778, 328))
hline(s, X0 + 0.3, X0 + 5.3, CY + 3.55)
stat(s, X0 + 0.3, CY + 3.75, 2.4, '결함 크기', '사진의 0.2–0.5%', vsize=20)
vline(s, X0 + 2.75, CY + 3.8, CY + 4.55)
stat(s, X0 + 2.95, CY + 3.75, 2.4, '판정 기준', '오판 1% 이내', vsize=20)
c2 = X0 + 5.8
card(s, c2, CY, XW - 5.8, CH)
items(s, c2 + 0.35, CY + 0.4, XW - 6.5, [
    (1, 'PCB는 거의 모든 전자 제품의 바탕', '결함 하나가 제품 고장으로 이어짐'),
    (2, '검사는 라인에서 찍은 사진으로', '사진마다 정상 · 불량을 자동 판정'),
    (3, '놓친 불량은 뒤에서 더 비쌈', '다음 공정 · 출하 뒤에 발견될수록 수리 · 회수 비용 증가'),
    (4, '너무 많이 걸러도 손해', '정상을 불량으로 잡으면 재검 인력 · 시간 낭비 → 오판을 정상 100장 중 1장 안으로 묶고, 그때 놓친 불량을 잰다')],
    step=1.12)

s = new('현장의 현실 - 불량 사진은 늘 적고, 결함은 작고 종류가 많다',
        notes='그런데 현장의 데이터는 학습하기 어렵습니다. 불량은 드물어서 라인 사진 대부분이 정상이고, 새 제품이나 새 라인은 처음에 불량 사진이 거의 없습니다. '
              '불량 사진은 재검 단계에서 사람이 확인하면서 종류마다 몇 장씩 쌓입니다. '
              '게다가 결함은 사진의 0.2에서 0.5%로 작고, 휜 핀, 녹은 납, 긁힘, 부품 누락처럼 종류가 많습니다. '
              '즉 정상 사진은 많고, 불량은 종류마다 몇 장뿐인 상황에서 시작해야 합니다.')
card(s, X0, CY, 5.6, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 5.0, '결함 종류 (VisA PCB)', '빨간 선 = 픽셀 단위로 표시된 결함 위치')
for i, (t_, ko) in enumerate([('bent', '휜 핀'), ('melt', '녹은 납 · 부품'), ('scratch', '긁힘'), ('missing', '부품 누락')]):
    x = X0 + 0.3 + i * 1.27
    s.image(FIG / f's02_visa_type_{t_}.jpg', x, y + 0.1, w=1.17)
    s.text(x, y + 1.32, 1.17, 0.3, ko, size=11, color=BODY, bold=True, align='c')
hline(s, X0 + 0.3, X0 + 5.3, CY + 2.95)
stat(s, X0 + 0.3, CY + 3.15, 2.4, '정상 사진 (PCB마다)', '약 1,000장', vsize=24)
vline(s, X0 + 2.75, CY + 3.2, CY + 4.05)
stat(s, X0 + 2.95, CY + 3.15, 2.4, '불량 사진 (PCB마다)', '100장', color=ACC, vsize=24)
s.text(X0 + 0.3, CY + 4.25, 5.0, 0.5, '공개 데이터 VisA도 불량은 정상의 10분의 1. 학습에는 종류마다 1–8장만 쓴다', size=11, color=MUTED)
c2 = X0 + 5.8
card(s, c2, CY, XW - 5.8, CH)
items(s, c2 + 0.35, CY + 0.4, XW - 6.5, [
    (1, '불량은 드물다', '라인 사진 대부분이 정상, 불량은 일부'),
    (2, '새 제품 · 새 라인은 불량 사진이 거의 없음', '종류마다 몇 장에서 시작해 조금씩 쌓임'),
    (3, '불량 사진은 사람이 확인하며 쌓인다', '재검 단계에서 확인하고 결함 위치를 표시'),
    (4, '결함은 작고 종류가 많다', '사진의 0.2–0.5%, VisA 4종 · 실제 라인(SMT) 18종')],
    step=1.12)

s = new('기존 방법 - 쌓인 불량 사진을 버리거나, 너무 많이 요구한다',
        notes='기존 방법은 크게 세 갈래입니다. 첫째, 정상 사진만으로 학습하는 이상탐지입니다. 불량 사진이 없어도 되지만, 쌓인 불량 사진을 버리고, 정상 사진이 다양하면 약합니다. '
              '둘째, 불량까지 학습하는 지도 학습입니다. 불량을 직접 배우지만 종류마다 불량 사진이 많이 필요하고, 적으면 엉뚱한 단서를 배우기 쉽습니다. 이건 뒤에서 직접 보여 드리겠습니다. '
              '셋째, 생성 모델로 불량을 늘리는 방법입니다. 적은 불량을 수십 장으로 늘릴 수 있지만, 그 합성이 실제 불량 몇 장의 값어치인지는 알기 어렵습니다. '
              '현장에서 정말 궁금한 것은, 합성을 만드는 게 나은지 실제 불량을 더 모으는 게 나은지입니다.')
cols = [('정상만 학습', '이상탐지 · 예: PatchCore', ['불량 사진이 없어도 된다'], ['쌓인 불량 사진을 버린다', '정상 사진이 다양하면 약하다']),
        ('불량까지 학습', '지도 학습 · 예: DevNet, 분류기', ['불량의 생김새를 직접 배운다'], ['종류마다 불량 사진이 많이 필요하다', '적으면 엉뚱한 단서를 배우기 쉽다 (뒤에서)']),
        ('생성 모델로 불량 늘리기', '예: GAN, 확산 모델', ['적은 불량을 수십 장으로 늘린다'], ['합성이 실제 불량 몇 장의 값어치인지 모른다', '많이 만들수록 좋아지는지 모른다'])]
cw = (XW - 2 * 0.2) / 3
for i, (h, sub, pros, cons) in enumerate(cols):
    x = X0 + i * (cw + 0.2)
    card(s, x, CY, cw, 4.1)
    s.text(x + 0.3, CY + 0.28, 0.5, 0.42, str(i + 1), size=20, color=ACC, bold=True)
    s.text(x + 0.3, CY + 0.78, cw - 0.6, 0.36, h, size=16, color=INK, bold=True, anchor='m')
    s.text(x + 0.3, CY + 1.15, cw - 0.6, 0.3, sub, size=11, color=MUTED, anchor='m')
    hline(s, x + 0.3, x + cw - 0.3, CY + 1.6)
    s.text(x + 0.3, CY + 1.75, cw - 0.6, 0.3, '장점', size=12, color=ACC, bold=True)
    bullets(s, x + 0.3, CY + 2.07, cw - 0.6, 0.45, pros, size=13)
    s.text(x + 0.3, CY + 2.6, cw - 0.6, 0.3, '한계', size=12, color=BAD, bold=True)
    bullets(s, x + 0.3, CY + 2.92, cw - 0.6, 1.1, cons, size=13, after=6)
s.box(X0, CY + 4.3, XW, 0.7, '현장의 질문: 불량 몇 장을 어떻게 써야 하나? 합성을 만들까, 실제 불량을 더 모을까?', fill=ACC_T, size=15,
      color=ACC, bold=True, radius=0.14, align='l', margin=(0.3, 0.05))

s = new('우리가 풀 문제 - 적은 불량 사진을 어떻게 써야 결함을 보는 모델이 될까', size=24,
        notes='그래서 저희가 푼 문제는 이렇습니다. 정상 사진 수백 장과 결함 종류마다 몇 장의 불량 사진, 그리고 결함 위치가 주어졌을 때, '
              '정상을 잘못 거르는 비율을 1% 안으로 묶으면서 놓치는 불량을 최소로 만드는 것입니다. '
              '이를 세 가지 질문으로 나눴습니다. 첫째, 적은 불량 사진을 어떻게 써야 하는가. 둘째, 생성 모델로 만든 합성 불량은 얼마나 넣어야 하고 실제 몇 장의 값어치인가. '
              '셋째, 결함 위치가 상자뿐인 실제 생산 라인에서도 그런가입니다.')
card(s, X0, CY, 5.3, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 4.7, '문제 설정', color=ACC, size=16)
for i, (h, b) in enumerate([('주어진 것', '정상 사진 약 500장 + 결함 종류마다 불량 K장(1–16장)과 결함 위치'),
                            ('할 일', '새 사진마다 정상 · 불량 판정'),
                            ('목표', '정상을 잘못 거르는 비율을 1%(또는 5%) 안으로 묶고, 놓친 불량을 최소로'),
                            ('데이터', '공개 PCB(VisA)로 시작해, 실제 생산 라인(AIHub SMT)에서 확인')]):
    yy = y + 0.1 + i * 1.02
    s.text(X0 + 0.3, yy, 4.7, 0.3, h, size=12, color=MUTED, bold=True)
    s.text(X0 + 0.3, yy + 0.32, 4.7, 0.62, b, size=14, color=INK)
c2 = X0 + 5.5
card(s, c2, CY, XW - 5.5, CH)
for i, (rq, h, b) in enumerate([('RQ1', '불량 사진, 어떻게 쓸까', '사진마다 라벨 하나로 배울까, 결함 위치까지 쓸까? 몇 장이면 정상만 쓰는 방법보다 나을까?'),
                                ('RQ2', '합성은 얼마나', '생성 모델로 만든 불량을 몇 장 넣어야 하나? 실제 불량 몇 장의 값어치인가?'),
                                ('RQ3', '실제 라인에서도?', '결함 위치가 상자뿐이고 정상이 다양한 실제 생산 라인에서도 같은가?')]):
    yy = CY + 0.4 + i * 1.5
    s.box(c2 + 0.35, yy, 0.75, 0.38, rq, fill=ACC, size=12, color=WHITE, bold=True, radius=0.08, margin=(0, 0))
    s.text(c2 + 1.3, yy, XW - 7.3, 0.38, h, size=16, color=INK, bold=True, anchor='m')
    s.text(c2 + 1.3, yy + 0.48, XW - 7.3, 0.8, b, size=13, color=BODY)

# ================================================================ 2. 데이터와 첫 시도
section('2. 데이터와 첫 시도', '2  데이터와 첫 시도')
s = new('데이터 ① VisA PCB - PCB마다 정상 약 1,000장, 불량 100장',
        notes='첫 번째 데이터는 공개 데이터셋 VisA의 PCB 세 종류입니다. PCB마다 정상 약 천 장, 불량 백 장이고, 결함 위치는 픽셀 단위로 표시돼 있습니다. '
              '정상 500장으로 학습하고, 불량은 종류마다 10장을 학습용으로 떼어 둔 뒤 그중 K장만 씁니다. 나머지 불량은 시험에 씁니다.')
card(s, X0, CY, 8.75, CH)
TW = [0.8, 1.55, 1.55, 0.9, 0.8, 2.55]
rows = [['', '정상 예시', '불량 예시', '정상', '불량', '결함 종류별 불량 장수']]
for pcb in ('pcb1', 'pcb2', 'pcb3'):
    c, ty = VISA[pcb], VISA[pcb]['types']
    rows.append([pcb.upper(), '', '', f"{c['normal']:,}", str(c['defect']),
                 f"녹은 납 {ty['melt']} · 누락 {ty['missing']}\n휜 핀 {ty['bent']} · 긁힘 {ty['scratch']}\n두 종류 이상 {c['multi']}"])
RH, HH = 1.15, 0.45
table(s, X0 + 0.3, CY + 0.25, TW, rows, size=12, row_h=RH, header_h=HH, align=['c', 'c', 'c', 'c', 'c', 'l'])
for i, pcb in enumerate(('pcb1', 'pcb2', 'pcb3')):
    for j, kind in enumerate(('normal', 'defect')):
        fit_image(s, FIG / f's02_visa_{pcb}_{kind}.jpg', X0 + 0.3 + TW[0] + j * TW[1], CY + 0.25 + HH + i * RH, TW[1], RH)
s.text(X0 + 0.3, CY + 4.2, 8.1, 0.6, '불량 예시의 빨간 상자 = 결함 위치 (PCB1 휜 핀 · PCB2 부품 누락 · PCB3 녹은 납). 사진 크기 약 1400×1000. '
       f'PCB · 결함 유형별 정상 · 불량 예시: 부록 {APP_VISA}–{APP_VISA + 3}', size=11, color=MUTED)
c2 = X0 + 8.95
card(s, c2, CY, XW - 8.95, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 2.7, '실험에 쓴 장수', 'PCB마다', color=ACC)
for i, (h, b) in enumerate([('정상', '학습 500 · 검증 200\n시험 약 300'), ('불량 (학습)', '종류마다 10장을 떼어 두고\n그중 K = 1 · 2 · 4 · 8장'),
                            ('불량 (시험)', '나머지 60장 안팎')]):
    yy = y + 0.1 + i * 1.25
    s.text(c2 + 0.3, yy, 2.7, 0.3, h, size=12, color=MUTED, bold=True)
    s.text(c2 + 0.3, yy + 0.32, 2.7, 0.8, b, size=13.5, color=INK)

s = new('비교한 기존 모델 - 같은 특징을 쓰고, 불량 라벨을 쓰는 방식만 다르다', size=24,
        notes='처음 비교한 기존 모델입니다. 모두 ImageNet으로 미리 학습한 같은 WideResNet50을 고정해서, 사진을 작은 칸으로 나눈 칸별 특징을 씁니다. '
              '다른 것은 불량 라벨을 쓰는 방식뿐입니다. PatchCore는 정상 사진의 칸 특징을 모아 두고, 시험 사진의 칸이 정상과 얼마나 다른지로 점수를 냅니다. '
              '사진 단위 모델인 MLP와 DevNet은 사진 한 장을 벡터 하나로 요약하고, 사진마다 정상인지 불량인지만 알려 주고 학습합니다.')
card(s, X0, CY, XW, 1.25)
for i, (t_, st, w) in enumerate([('사진 512×512', IO, 2.3), ('WideResNet50 (ImageNet, 고정)', FIXED, 3.3),
                                  ('64×64칸, 칸마다 특징 1024차원', FIXED, 3.3)]):
    x = X0 + 0.35 + [0, 2.75, 6.5][i]
    s.box(x, CY + 0.3, w, 0.65, t_, size=13, radius=0.08, **st)
    if i < 2:
        arrow(s, x + w + 0.05, CY + 0.62, x + w + 0.4, CY + 0.62)
s.text(X0 + 10.2, CY + 0.3, 1.9, 0.65, '모든 모델이 같은 특징', size=12, color=ACC, bold=True, anchor='m')
cw = (XW - 0.2) / 2
for i, (h, sub, flow) in enumerate([
        ('PatchCore', '정상 사진만 학습', ['정상 사진의 칸 특징을 모아 둠 (대표 1%)', '시험 사진의 칸마다 가장 비슷한 정상 칸과의 거리', '가장 먼 칸의 거리 = 사진의 이상 점수']),
        ('사진 단위 모델', '정상 + 불량 K장, 사진마다 라벨 하나', ['칸 특징의 평균 · 최댓값을 이어 붙임 → 사진 하나 = 벡터 하나 (2048차원)',
                                                   'MLP: 작은 신경망 분류기 (2048 → 256 → 1)', 'DevNet: 정상 점수는 0 근처로, 불량 점수는 높게'])]):
    x = X0 + i * (cw + 0.2)
    card(s, x, CY + 1.45, cw, 3.55)
    s.text(x + 0.3, CY + 1.7, cw - 0.6, 0.36, h, size=16, color=ACC, bold=True, anchor='m')
    s.text(x + 0.3, CY + 2.08, cw - 0.6, 0.3, sub, size=12, color=MUTED, anchor='m')
    for j, f in enumerate(flow):
        yy = CY + 2.55 + j * 0.78
        s.box(x + 0.3, yy, cw - 0.6, 0.6, f, size=12.5, radius=0.08, align='l', margin=(0.18, 0.03), **(TRAIN if j == 0 else FIXED))
        if j < 2:
            arrow(s, x + cw / 2, yy + 0.61, x + cw / 2, yy + 0.77)

s = new('시도 - 불량 라벨을 쓰자 원래 시험 점수가 좋아졌다',
        notes='먼저 기존 방식대로 비교했습니다. 불량 장수를 늘릴수록 사진 단위 모델이 놓친 불량이 꾸준히 줄었고, 불량 8장이면 정상만 쓰는 PatchCore보다 덜 놓쳤습니다. '
              '정상과 불량을 가르는 점수도 0.99로 높았습니다. 불량 라벨이 도움이 된다고 보였습니다. 그런데 정말 결함을 보고 맞힌 걸까요?')
card(s, X0, CY, 7.6, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 7.0, '원래 시험셋에서 놓친 불량', '정상 100장 중 1장을 잘못 판정하는 기준, 반복 5번 × PCB 3개. x = 종류마다 학습에 쓴 불량 장수')
ch = s.line_chart(X0 + 0.15, y, 7.3, CH - (y - CY) - 0.15, ['1', '2', '4', '8', '10'],
                  [('DevNet', [0.341, 0.234, 0.153, 0.093, 0.080]), ('MLP', [0.563, 0.410, 0.313, 0.177, 0.140])],
                  [ACC, GRAYS], 0, 0.6, 0.1, ref=(0.153, 'PatchCore 15.3%'), lay=(0.1, 0.12, 0.87, 0.74))
pct_axis(ch)
quiet(ch)
c2 = X0 + 7.8
card(s, c2, CY, XW - 7.8, CH)
stat(s, c2 + 0.35, CY + 0.35, 3.8, '불량 8장 DevNet vs PatchCore', '9.3% < 15.3%', color=ACC)
stat(s, c2 + 0.35, CY + 1.4, 3.8, '정상 · 불량 구분 점수 (AUROC)', '0.99')
hline(s, c2 + 0.35, c2 + 4.08, CY + 2.5)
s.text(c2 + 0.35, CY + 2.75, 3.75, 0.4, '"불량 라벨이 도움이 된다"로 보였다', size=15, color=INK, bold=True)
s.text(c2 + 0.35, CY + 3.25, 3.75, 0.8, '그런데 정말 결함을 보고 맞힌 걸까?', size=17, color=ACC, bold=True)

s = new('함정 - 모델은 결함이 아니라 배경을 보고 있었다',
        notes='모델이 어디를 보고 판단했는지 Grad-CAM으로 찍어 봤습니다. 결함은 아래쪽 핀에 있는데, 모델은 기판 전체를 보고 있었습니다. '
              '모델이 본 곳 중 결함 자리는 4%뿐이고, 결함을 지워 정상처럼 만든 사진도 44%를 불량이라고 했습니다. '
              '원인은 데이터였습니다. VisA는 불량 사진만 따로, 더 밝게 찍혀서 사진 밝기와 기판이 놓인 자리만으로도 정상과 불량이 갈립니다. '
              '현장에서도 정상과 불량을 다른 장비로 찍으면 같은 일이 생기고, 원래 시험셋도 같은 차이가 있어서 점수로는 드러나지 않습니다.')
card(s, X0, CY, 3.9, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 3.3, '모델이 본 곳 (사진 단위 DevNet)')
s.image(GRADCAM, X0 + 0.3, y + 0.05, w=3.3, crop=(CAM_ORIG[0], CAM_ORIG[1], CAM_DEVNET[2], CAM_DEVNET[3]))
s.text(X0 + 0.3, CY + 2.45, 3.3, 0.5, 'PCB1 녹은 납. 왼쪽 원본(흰 선 = 결함), 오른쪽 모델이 본 곳', size=11, color=MUTED)
stat(s, X0 + 0.3, CY + 3.15, 1.6, '결함 자리 비율', '4%')
vline(s, X0 + 1.95, CY + 3.2, CY + 4.1)
stat(s, X0 + 2.15, CY + 3.15, 1.6, '지운 사진도 불량', '44%')
card(s, X0 + 4.1, CY, 3.9, CH)
y = ctitle(s, X0 + 4.4, CY + 0.25, 3.3, '사진 평균 밝기 (PCB2)')
s.image(FIG / 's03_capture_difference.png', X0 + 4.55, y, h=2.0, crop=(875, 503, 1405, 862))
s.text(X0 + 4.4, CY + 2.75, 3.3, 0.3, '파랑 = 정상, 주황 = 불량', size=11, color=MUTED)
stat(s, X0 + 4.4, CY + 3.15, 3.3, '밝기 · 기판 자리만으로 구분 (AUROC)', '0.93 · 0.85 · 0.76', vsize=22)
c3 = X0 + 8.2
card(s, c3, CY, XW - 8.2, CH)
items(s, c3 + 0.3, CY + 0.35, XW - 8.8, [
    (1, '결함 대신 배경을 학습', '모델이 본 곳 중 결함 자리는 4%. 결함을 지운 사진도 44%를 불량으로 판정'),
    (2, '불량 사진만 따로 촬영', 'VisA는 불량만 따로, 더 밝게 찍힘. 밝기와 기판 자리만으로 정상 · 불량 구분'),
    (3, '현장에서도 생기는 일', '정상은 검사기, 불량은 재검 장비에서 찍히면 모델은 장비 차이를 학습'),
    (4, '원래 시험 점수로는 안 보임', '원래 시험셋도 같은 차이라 점수가 높게 나옴')],
    step=1.13, nsize=17, hsize=14, bsize=12, gap=0.38)

# ================================================================ 3. 방법 · RQ1
section('3. 방법 · RQ1', '3  방법 · RQ1')
s = new('방법을 바꾸다 - 결함 위치로 배우고, 결함만 바꾼 시험으로 잰다',
        notes='그래서 방법을 네 가지 바꿨습니다. 첫째, 결함 위치를 정답으로 주고 사진의 칸마다 결함인지 맞히게 합니다. '
              '둘째, 학습 사진의 밝기와 색을 무작위로 바꿔, 찍힌 환경으로는 정상과 불량을 가를 수 없게 합니다. '
              '셋째, 시험도 바꿨습니다. 불량 사진에서 결함만 지운 사진은 정상으로, 그 결함을 정상 사진에 붙인 사진은 불량으로 맞혀야 합니다. '
              '넷째, 적은 불량은 생성 모델로 늘립니다. 아래는 전체 흐름입니다.')
steps = [('결함 위치로 배운다', '사진마다 라벨 하나로는 어디가 결함인지 모름 → 사진의 칸마다 결함인지 맞히게 함'),
         ('밝기·색을 바꿔 배운다', '학습 사진의 밝기 · 조명 · 색 · 선명도를 무작위로 → 찍힌 환경으로는 정상과 불량을 못 가르게'),
         ('결함만 바꾼 시험으로 잰다', '결함 지운 사진은 정상으로, 그 결함을 붙인 정상 사진은 불량으로 맞혀야 함'),
         ('합성으로 늘린다', '불량 K장으로 생성 모델을 학습해, 정상 사진 위에 가짜 결함을 그림')]
cw = (XW - 3 * 0.18) / 4
for i, (h, b) in enumerate(steps):
    x = X0 + i * (cw + 0.18)
    card(s, x, CY, cw, 2.85)
    s.text(x + 0.25, CY + 0.25, 0.5, 0.45, str(i + 1), size=22, color=ACC, bold=True)
    s.text(x + 0.25, CY + 0.8, cw - 0.45, 0.4, h, size=14.5, color=INK, bold=True, anchor='m')
    s.text(x + 0.25, CY + 1.3, cw - 0.45, 1.4, b, size=12.5, color=BODY)
card(s, X0, CY + 3.05, XW, 1.95)
s.text(X0 + 0.3, CY + 3.25, 4, 0.3, '전체 흐름', size=13, color=ACC, bold=True)
flow = [('실제 불량 K장\n+ 결함 위치', IO), ('생성 모델 (4)\ncTransGAN · SD+LoRA', TRAIN), ('합성 불량 + 위치\n(정상 사진 위에)', IO),
        ('위치 모델 (1 · 2)\n결함 지도를 학습', TRAIN), ('평가 (3)\n원래 + 결함 바꾼 시험', IO)]
fw = 2.05
for i, (t_, st) in enumerate(flow):
    x = X0 + 0.3 + i * (fw + 0.36)
    s.box(x, CY + 3.65, fw, 0.85, t_, size=12, radius=0.08, **st)
    if i < 4:
        arrow(s, x + fw + 0.04, CY + 4.07, x + fw + 0.32, CY + 4.07)
s.text(X0 + 0.3, CY + 4.55, 11.6, 0.3, '정상 사진 수백 장 = 합성의 바탕이자 탐지 모델의 정상 학습 데이터', size=11, color=MUTED)

s = new('결함 바꾼 시험 - 배경은 그대로 두고 결함만 바꾼 사진 짝으로 잰다',
        notes='세 번째 변화인 결함 바꾼 시험입니다. 원래 시험셋도 정상과 불량의 촬영 환경이 달라서, 배경만 봐도 점수가 잘 나옵니다. '
              '그래서 시험 불량마다 결함을 지운 사진을 만들고, 그 결함을 정상 사진에 붙인 사진도 만들었습니다. '
              '결함을 지운 사진은 정상으로, 결함을 붙인 사진은 불량으로 맞혀야 합니다. 배경만 보는 모델은 여기서 틀립니다. '
              '이 시험은 정상 쪽 사진이 적어서, 정상 100장 중 5장을 잘못 거르는 기준으로 놓친 불량을 잽니다.')
card(s, X0, CY, 7.7, CH)
FW = 6.6
FX = X0 + 0.55
sc = FW / 1890
for cx, lab, c in ((230, '시험 불량', BAD), (610, '결함 지운 사진', NEG), (990, '시험 정상', NEG),
                   (1370, '정상 + 붙인 결함', BAD), (1750, '정상 + 정상 조각', NEG)):
    s.text(FX + (cx - 45) * sc - 0.7, CY + 0.25, 1.4, 0.32, lab, size=11, bold=True, color=c, align='c')
s.image(FIG / 's13_fair_test_examples.png', FX, CY + 0.6, w=FW, crop=(45, 120, 1935, 1213))
s.text(FX, CY + 4.58, FW, 0.3, '주황 = 불량으로 맞혀야 함, 파랑 = 정상으로 맞혀야 함', size=11, color=MUTED)
c2 = X0 + 7.9
card(s, c2, CY, XW - 7.9, CH)
items(s, c2 + 0.3, CY + 0.35, XW - 8.5, [
    (1, '원래 시험셋의 문제', '정상과 불량의 촬영 환경이 달라 배경만 봐도 점수가 잘 나옴'),
    (2, '결함만 바꾼 짝', '결함 지운 사진 → 정상, 결함 붙인 정상 사진 → 불량. 배경만 보면 틀림'),
    (3, '대조군', '정상 + 정상 조각: 붙이는 작업 자체가 단서가 되지 않는지 확인'),
    (4, '지표', '정상 100장 중 5장을 잘못 판정하는 기준에서 놓친 불량 (정상 쪽 사진이 PCB마다 60장 안팎이라 1% 기준은 한 장에 흔들림)')],
    step=1.13, nsize=17, hsize=14, bsize=12, gap=0.38)

s = new('생성 모델 ① cTransGAN - 불량 K장만으로 처음부터 학습하는 GAN',
        notes='첫 번째 생성 모델은 cTransGAN입니다. Transformer로 만든 GAN을 불량 K장만으로 처음부터 학습하도록 줄였습니다. '
              'K장의 결함 부분으로 결함 종류를 구분하는 작은 분류기를 학습해 종류별 평균 특징을 조건으로 줍니다. '
              '생성기는 8×8 토큰에서 시작해 두 번 키운 뒤 256×256 사진으로 펼치고, 정상 사진의 결함 자리에 끼워 넣습니다. '
              '진짜와 가짜를 가르는 손실만 쓰면 잡음만 나와서, 결함마다 잡음 벡터를 고정하고 원본을 다시 그리도록 했습니다.')
card(s, X0, CY, XW, 3.55)
X = X0 + 0.3
s.box(X, CY + 0.3, 2.3, 0.65, 'K장의 결함 부분\n(32×32로 잘라냄)', size=11.5, radius=0.08, **IO)
arrow(s, X + 1.15, CY + 0.97, X + 1.15, CY + 1.2)
s.box(X, CY + 1.22, 2.3, 0.65, 'Transformer 분류기\n(결함 종류 구분용)', size=11.5, radius=0.08, **TRAIN)
arrow(s, X + 1.15, CY + 1.89, X + 1.15, CY + 2.12)
s.box(X, CY + 2.14, 2.3, 0.65, '조건 c\n종류별 평균 특징', size=11.5, radius=0.08, **IO)
s.text(X + 2.75, CY + 0.25, 3, 0.3, '생성기 G', size=13, color=ACC, bold=True)
s.box(X + 2.75, CY + 0.62, 1.9, 0.65, '잡음 z (128차원)\n결함마다 고정', size=11.5, radius=0.08, **IO)
arrow(s, X + 3.7, CY + 1.29, X + 3.7, CY + 1.52)
for i, gt in enumerate(['z와 c를 합쳐\n8×8 토큰', '2배 키우기\n16×16 · 2층', '2배 키우기\n32×32 · 1층']):
    gx = X + 2.75 + i * 2.15
    s.box(gx, CY + 1.55, 1.9, 0.75, gt, size=11.5, radius=0.08, **TRAIN)
    arrow(s, gx + 1.93, CY + 1.92, gx + 2.12, CY + 1.92)
arrow(s, X + 2.32, CY + 2.45, X + 2.72, CY + 2.05)
s.box(X + 9.2, CY + 1.55, 2.4, 0.75, '토큰마다 8×8 픽셀\n→ 256×256 사진', size=11.5, radius=0.08, **IO)
arrow(s, X + 10.4, CY + 2.32, X + 10.4, CY + 2.55)
s.box(X + 9.2, CY + 2.57, 2.4, 0.7, '정상 사진의 결함 자리에\n끼워 넣음', size=11.5, radius=0.08, **IO)
s.box(X + 6.85, CY + 2.57, 2.0, 0.7, '판별기 D (Transformer)\n진짜 / 가짜', size=11.5, radius=0.08, **TRAIN)
arrow(s, X + 9.18, CY + 2.92, X + 8.88, CY + 2.92)
s.box(X + 2.75, CY + 2.57, 3.85, 0.7, '손실: 진짜·가짜 + 다시 그린 부분의 픽셀 차이 + 경계 + 세밀한 무늬', size=11, radius=0.08,
      align='l', margin=(0.12, 0.03), **LOSSB)
card(s, X0, CY + 3.75, 7.4, 1.25)
s.text(X0 + 0.3, CY + 3.9, 7.0, 0.3, '실제 | 바탕 정상 사진 | 합성 | 결함 위치 (K = 1)', size=11, color=MUTED, bold=True)
s.image(FIG / 's07_ctransgan_K1_review.png', X0 + 0.3, CY + 4.2, h=0.7, crop=(3, 940, 757, 1049))
card(s, X0 + 7.6, CY + 3.75, XW - 7.6, 1.25)
bullets(s, X0 + 7.9, CY + 3.92, XW - 8.2, 1.0, ['Wang et al.(2023) Conditional TransGAN을 K장에 맞게 줄임',
                                              '판정 손실만 쓰면 무지개색 잡음 → 결함마다 잡음 고정'], size=12, after=5)

s = new('생성 모델 ② SD+LoRA - 미리 학습된 그림 모델에 작은 부분만 추가 학습', size=24,
        notes='두 번째 생성 모델은 Stable Diffusion 1.5의 인페인팅 모델에 LoRA만 붙여 학습한 것입니다. '
              '큰 데이터로 미리 학습된 확산 모델은 질감과 조명을 이미 알기 때문에, K장으로는 전체의 1% 미만인 작은 행렬만 학습해 원본을 외우는 것을 줄입니다. '
              '결함 종류는 문장으로 알려 주고, 만들 때는 정상 사진에 새 결함 위치를 주고 그 안만 다시 그려 붙입니다. 두 배를 만들어 바탕과 가장 달라진 40장을 고릅니다.')
card(s, X0, CY, XW, 3.55)
X = X0 + 0.3
s.text(X, CY + 0.2, 6, 0.3, '학습 (PCB마다 1개, 1000 step)', size=13, color=ACC, bold=True)
for i, (bx, bw, bt, st) in enumerate([(0, 2.4, 'K장의 결함 주변 512×512\n(정상 사진 10% 섞음)', IO), (2.75, 1.5, 'VAE 인코더\n(고정)', FIXED),
                                      (4.6, 2.9, 'UNet (SD1.5 인페인팅, 고정)\n+ LoRA (rank 16)만 학습', TRAIN),
                                      (7.85, 1.55, '예측한 잡음', IO), (9.75, 1.85, '손실: 잡음 예측 오차', LOSSB)]):
    s.box(X + bx, CY + 0.6, bw, 0.75, bt, size=11.5, radius=0.08, **st)
    if i < 4:
        nx = [2.75, 4.6, 7.85, 9.75][i]
        arrow(s, X + bx + bw + 0.03, CY + 0.97, X + nx - 0.03, CY + 0.97)
s.box(X, CY + 1.5, 2.4, 0.55, 'CLIP 문장 인코더 (고정)\n결함 종류를 문장으로', size=10.5, radius=0.08, **FIXED)
arrow(s, X + 2.43, CY + 1.77, X + 4.57, CY + 1.2)
s.text(X, CY + 2.2, 6, 0.3, '생성 (종류마다 40장)', size=13, color=ACC, bold=True)
gen = [(0, 2.4, '정상 사진 + 새 결함 위치', IO), (2.75, 2.3, '같은 UNet + LoRA\n잡음을 30번 걷어 냄', TRAIN),
       (5.4, 1.6, 'VAE 디코더\n(고정)', FIXED), (7.35, 2.0, '결함 위치 안만\n정상 사진에 붙임', IO), (9.7, 1.9, '2배 만들고, 바탕과\n가장 달라진 40장', IO)]
for i, (gx, gw, gt, st) in enumerate(gen):
    s.box(X + gx, CY + 2.58, gw, 0.75, gt, size=11.5, radius=0.08, **st)
    if i < 4:
        arrow(s, X + gx + gw + 0.03, CY + 2.95, X + gen[i + 1][0] - 0.03, CY + 2.95)
card(s, X0, CY + 3.75, 5.3, 1.25)
s.text(X0 + 0.3, CY + 3.9, 4.9, 0.3, '실제 결함 | 바탕 정상 사진 | 합성 (K = 1)', size=11, color=MUTED, bold=True)
s.image(FIG / 's08_sd_lora_v3_pcb1_sheet.jpg', X0 + 0.3, CY + 4.2, h=0.7, crop=(0, 599, 790, 858))
card(s, X0 + 5.5, CY + 3.75, XW - 5.5, 1.25)
bullets(s, X0 + 5.8, CY + 3.92, XW - 6.1, 1.0, ['미리 학습된 확산 모델은 질감 · 조명을 이미 앎',
                                              'K장으로는 전체의 1% 미만인 작은 행렬(LoRA)만 학습 → 원본을 외우는 것을 줄임'], size=12, after=5)

s = new('합성 시행착오 - 그럴듯함보다 흔적 없애기가 중요했다',
        notes='합성 불량을 만들면서 겪은 시행착오입니다. 처음 cTransGAN은 잡음만 냈고, SD+LoRA는 흰 얼룩을 그리거나 K장을 그대로 외웠습니다. '
              '가장 중요했던 조정은 생성 흔적을 없애는 것이었습니다. 휜 핀을 그릴 때 원래 곧은 핀이 남아 겹쳐 보여서, 원래 핀 자리까지 다시 그리고 남은 핀은 색으로 찾아 지웠습니다. '
              '그리고 합성은 늘 정상 사진 위에 그려서, 불량 사진의 촬영 환경이 따라오지 않게 했습니다.')
card(s, X0, CY, 7.0, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 6.4, '겪은 문제와 고친 방법')
rows = [['시도', '문제', '고친 방법'],
        ['cTransGAN,\n판정 손실만', '무지개색 잡음만 나옴', '결함마다 잡음을 고정하고\n원본을 다시 그리게 함'],
        ['SD+LoRA\n초기 설정', '흰 얼룩, K장을 그대로\n외움, 결함을 흐리게', '정상 사진 10% 섞기, rank 16,\n2배 만들고 많이 달라진 장만'],
        ['결함 위치 안만\n다시 그림', '원래 곧은 핀이 남아\n핀이 겹쳐 보임', '원래 핀 자리까지 다시 그리고,\n남은 핀은 색으로 찾아 지움']]
table(s, X0 + 0.3, y + 0.05, [1.55, 2.15, 2.7], rows, size=12, row_h=0.82, header_h=0.42)
s.box(X0 + 0.3, CY + 4.1, 6.4, 0.6, '원칙: 합성은 늘 정상 사진 위에 → 불량 사진의 촬영 환경이 따라오지 않는다', fill=ACC_T, size=13,
      color=ACC, bold=True, radius=0.1, align='l', margin=(0.2, 0.03))
c2 = X0 + 7.2
card(s, c2, CY, XW - 7.2, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 4.4, '남은 핀 지우기 (휜 핀)', '실제 | 지우기 전 (핀이 겹침) | 지운 후')
s.image(FIG / 's06_pinfix_before_after.jpg', c2 + 0.3, y + 0.05, w=4.43, crop=(0, 633, 790, 892))
y2 = y + 0.05 + 4.43 * 259 / 790 + 0.2
s.text(c2 + 0.3, y2, 4.4, 0.3, 'SD+LoRA (K=1): 실제 결함 | 바탕 정상 사진 | 합성', size=11, color=MUTED, bold=True)
s.image(FIG / 's08_sd_lora_v3_pcb1_sheet.jpg', c2 + 0.3, y2 + 0.32, w=4.43, crop=(0, 599, 790, 858))

s = new('위치 모델 - 결함 지도를 직접 학습한다',
        notes='탐지에 쓴 위치 모델입니다. 앞의 모델과 같은 고정된 WRN50에서 크기가 다른 특징 세 개를 뽑고, 그 위에 작은 디코더만 학습해 결함 지도를 냅니다. '
              '사진의 칸마다 정답이 있어서 작은 결함이 묻히지 않고, 사진 전체의 밝기로는 결함 지도를 맞힐 수 없습니다. '
              '사진의 점수는 결함 지도에서 가장 높은 10칸의 평균이고, 다른 모델과 같은 기준으로 판정합니다.')
card(s, X0, CY, XW, 2.3)
X = X0 + 0.35
s.box(X, CY + 0.55, 1.9, 1.2, '사진\n512×512', size=12, radius=0.08, **IO)
arrow(s, X + 1.95, CY + 1.15, X + 2.3, CY + 1.15)
s.box(X + 2.35, CY + 0.3, 3.0, 1.7, '', fill=GRAY_F, radius=0.1)
s.text(X + 2.45, CY + 0.35, 2.8, 0.3, 'WRN50 (고정)', size=12, color=BODY, bold=True, align='c')
for i, (lab, sz) in enumerate([('128×128칸', 1.0), ('64×64칸', 0.88), ('32×32칸', 0.76)]):
    s.box(X + 2.55, CY + 0.72 + i * 0.42, 2.6 * sz, 0.34, lab, size=11, fill=WHITE, radius=0.05, margin=(0.04, 0.0))
arrow(s, X + 5.4, CY + 1.15, X + 5.75, CY + 1.15)
s.box(X + 5.8, CY + 0.3, 3.0, 1.7, [Para('작은 디코더 (학습)', 13, ACC, True, align='c', after=4),
                                    Para('크기가 다른 세 특징을', 12, align='c', after=1), Para('합치며 크게 키운다', 12, align='c', after=0)],
      radius=0.1, **TRAIN)
arrow(s, X + 8.85, CY + 1.15, X + 9.2, CY + 1.15)
s.box(X + 9.25, CY + 0.55, 2.3, 1.2, '결함 지도\n128×128', size=14, bold=True, fill=ACC, color=WHITE, radius=0.1)
cw = (XW - 3 * 0.18) / 4
for i, (h, b) in enumerate([('정답', '결함 위치. VisA는 픽셀 단위, SMT는 상자 (상자 안 전체를 결함으로)'),
                            ('사진의 점수', '결함 지도에서 가장 높은 10칸의 평균 → 다른 모델과 같은 기준으로 판정'),
                            ('학습', '한 번에 정상 4장 + 불량 4장, 2000번 갱신. 합성을 쓰면 불량 4장 중 2장이 합성'),
                            ('왜 이 구조인가', '칸마다 정답이 있어 작은 결함이 묻히지 않음. 사진 전체의 밝기로는 지도를 못 맞춤')]):
    x = X0 + i * (cw + 0.18)
    card(s, x, CY + 2.5, cw, 2.5)
    s.text(x + 0.25, CY + 2.72, 0.5, 0.42, str(i + 1), size=18, color=ACC, bold=True)
    s.text(x + 0.25, CY + 3.18, cw - 0.45, 0.36, h, size=14, color=INK, bold=True, anchor='m')
    s.text(x + 0.25, CY + 3.6, cw - 0.45, 1.3, b, size=12, color=BODY)

s = new('RQ1 결과 - 위치로 배우면 결함을 보고, 실제 3장부터 PatchCore보다 낫다', size=24,
        notes='첫 번째 질문의 답입니다. 같은 불량 1장이라도 사진 단위 DevNet은 결함 바꾼 시험에서 불량의 87%를 놓치는데, 위치 모델은 24%만 놓칩니다. '
              '결함을 지운 사진을 불량이라 하는 실수도 44%에서 14%로 줄고, 정상에 붙인 결함은 74%를 잡습니다. 이제 결함을 보는 것입니다. '
              '다만 실제 불량 1장으로는 정상만 쓰는 PatchCore와 비슷합니다. 실제 3장이면 PatchCore보다 확실히 덜 놓칩니다. '
              '불량 라벨은 결함 위치까지 써야 하고, 몇 장은 모아야 값을 합니다.')
card(s, X0, CY, 6.0, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 5.4, '결함을 보는가 (실제 불량 1장)')
for i, (crop, lab) in enumerate(((CAM_ORIG, '원본'), (CAM_DEVNET, '사진 단위 DevNet'), (CAM_SEG, '위치 모델'))):
    x = X0 + 0.3 + i * 1.83
    s.image(GRADCAM, x, y + 0.05, w=1.68, crop=crop)
    s.text(x, y + 1.75, 1.68, 0.3, lab, size=11, color=ACC if i == 2 else MUTED, bold=True, align='c')
rows = [['', 'DevNet', '위치 모델'],
        ['놓친 불량 ↓', '87%', '24%'], ['본 곳 중 결함 자리 ↑', '4%', '75%'],
        ['결함 지운 사진을 불량이라 함 ↓', '44%', '14%'], ['정상에 붙인 결함을 잡음 ↑', '16%', '74%']]
table(s, X0 + 0.3, CY + 2.72, [2.9, 1.25, 1.25], rows, size=12, row_h=0.36, colors=[[None] * 3] + [[None, None, ACC]] * 4)
s.text(X0 + 0.3, CY + 4.62, 5.4, 0.3, '결함 바꾼 시험. ↑ 클수록, ↓ 작을수록 좋음', size=10.5, color=MUTED)
c2 = X0 + 6.2
card(s, c2, CY, XW - 6.2, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 5.4, '정상만 쓰는 PatchCore와 비교', '결함 바꾼 시험에서 놓친 불량, 위치 모델 + 밝기·색 바꾸기')
ch = s.bar_chart(c2 + 0.15, y - 0.05, 5.75, 2.4, ['실제 1장', '2장', '3장', '4장'], [('위치 모델', [0.217, 0.133, 0.098, 0.079])],
                 [ACC], vmax=0.25, major=0.05, legend=False, fmt='0.0%', point_colors=[GRAYS, ACC_L, ACC, ACC],
                 ref=(0.188, None), gap=90, lay=(0.11, 0.1, 0.86, 0.74))
pct_axis(ch)
quiet(ch)
labels(ch)
s.text(c2 + 3.0, y - 0.05, 2.8, 0.3, '점선 = PatchCore 18.8%', size=11, color=BODY, bold=True, align='r')
y = CY + 3.45
for n, h, b in [(1, '실제 1장', 'PatchCore와 비슷 (차이 없음)'), (2, '실제 2장', '덜 놓치지만 우연일 수 있음'),
                (3, '실제 3장부터', '확실히 덜 놓침 (−9%p, 95% 범위 −16 ~ −1%p)')]:
    s.text(c2 + 0.3, y, 0.35, 0.36, str(n), size=15, color=ACC, bold=True, anchor='m')
    s.text(c2 + 0.65, y, 1.5, 0.36, h, size=13, color=INK, bold=True, anchor='m')
    s.text(c2 + 2.1, y, 3.8, 0.36, b, size=13, color=BODY, anchor='m')
    y += 0.42
s.text(c2 + 0.3, CY + 4.68, 5.4, 0.3, '밝기·색 바꾸기를 더하면 24.4% → 21.7%. 비교는 PCB 3개 × 반복 5번', size=10.5, color=MUTED)

# ================================================================ 4. RQ2
section('4. RQ2 합성은 얼마나', '4  RQ2 합성은 얼마나')
s = new('RQ2 - 합성 불량은 얼마나 넣어야 하고, 실제 몇 장의 값어치일까',
        notes='위치 모델에 cTransGAN 합성 40장을 더하면 놓친 불량이 21.7%에서 18%로 줄었습니다. 15번 비교 중 12번 좋아져 우연은 아닙니다. '
              '실제 결함을 오려 붙이기만 한 합성은 효과가 없었으니, 생성 모델이 새 모양을 그린 덕입니다. '
              '그렇다면 합성은 몇 장이 좋고, 실제 불량 몇 장의 값어치이며, 어느 생성 모델이 나을까요? '
              '생성 모델 2개, 실제 1, 2, 4, 8장, 합성 0장에서 40장까지 바꿔 돌렸습니다. 실제 8장에 합성을 더한 조건만 반복을 한 번 했습니다.')
card(s, X0, CY, XW, 1.35)
s.text(X0 + 0.3, CY + 0.2, 6, 0.3, '출발점 (RQ1과 같은 실험, 실제 1장 + 종류마다 합성 40장)', size=13, color=ACC, bold=True)
for i, (lab, val, c) in enumerate([('cTransGAN 합성', '21.7% → 18.0%', ACC), ('SD+LoRA 합성', '21.7% → 20.5%', INK),
                                   ('실제 결함을 오려 붙이기만', '21.7% → 22.4%', BAD)]):
    x = X0 + 0.3 + i * 4.0
    s.text(x, CY + 0.55, 3.7, 0.3, lab, size=12, color=BODY, bold=True)
    s.text(x, CY + 0.82, 3.7, 0.45, val, size=20, color=c, bold=True)
    if i:
        vline(s, x - 0.2, CY + 0.6, CY + 1.2)
card(s, X0, CY + 1.55, 5.6, 3.45)
items(s, X0 + 0.3, CY + 1.8, 5.0, [('N', '합성 장수', '종류마다 1장에서 40장까지 늘리면 계속 좋아질까?'),
                                  ('K', '실제 몇 장 값어치', '합성 40장은 실제 불량을 몇 장 모은 것과 같을까?'),
                                  ('G', '생성 모델', 'cTransGAN과 SD+LoRA 중 어느 쪽이 나을까?')],
      step=1.05, nsize=17, hsize=14, bsize=12.5, gap=0.38, bh=0.55)
c2 = X0 + 5.8
card(s, c2, CY + 1.55, XW - 5.8, 3.45)
y = ctitle(s, c2 + 0.3, CY + 1.75, 5.8, '실험표 (VisA, 생성 모델마다 같은 표)', '칸 숫자 = 반복 횟수 (한 번에 PCB 3개). 합성 N장 = 같은 40장 중 앞 N장')
rows = [['실제 K \\ 합성 N', '0', '1', '5', '10', '20', '40']]
for k in (1, 2, 4, 8):
    nsp = 1 if k == 8 else 5
    rows.append([f'K = {k}', '5'] + [str(nsp)] * 5)
colors = [[None] * 7] + [[None, MUTED] + [ACC] * 5] * 3 + [[None, MUTED] + [BAD] * 5]
table(s, c2 + 0.3, y, [1.55] + [0.68] * 6, rows, size=12.5, row_h=0.37, colors=colors)
s.text(c2 + 0.3, CY + 4.55, 5.8, 0.4, '실제만은 K = 3 · 5 · 6 · 10장도 반복 5번. 불량 4장 중 2장은 늘 합성 → N을 늘리면 합성의 가짓수만 늘어남',
       size=10.5, color=MUTED)

s = new('RQ2 결과 - 실제 장수가 가장 중요하고, 합성은 20장 안팎이면 충분하다',
        notes='x축은 종류마다 더한 합성 장수, 선은 실제 불량 장수입니다. 실제 장수를 늘리는 효과가 가장 큽니다. 실제 2장이 실제 1장에 합성 40장을 더한 것보다 낫습니다. '
              'cTransGAN은 합성을 20장까지 늘리면 좋아지다가 거의 멈춥니다. SD+LoRA는 실제 1장에서는 효과가 없고, 2장부터는 cTransGAN과 비슷합니다. '
              '실제 8장은 반복을 한 번만 해서 경향으로만 봅니다. 휜 핀은 두 생성 모델 모두 합성을 더하면 오히려 나빠졌습니다.')
N = ['0', '1', '5', '10', '20', '40']
sd = [[.217, .196, .239, .229, .217, .205], [.133, .137, .122, .135, .112, .115],
      [.079, .077, .070, .064, .059, .067], [.037, .051, .037, .046, .043, .042]]
ctg = [[.217, .202, .200, .191, .177, .180], [.133, .131, .119, .112, .111, .107],
       [.079, .083, .070, .071, .067, .063], [.037, .036, .037, .026, .022, .023]]
cw = (XW - 0.2) / 2
for i, (name, data, note) in enumerate((('SD+LoRA', sd, ['실제 1장에서는 효과 없음', '실제 2장부터 cTransGAN과 비슷']),
                                         ('cTransGAN', ctg, ['합성 20장까지 좋아지고 거의 멈춤', '실제 2장: 13.3% → 11.1% (20장), 10.7% (40장)']))):
    x = X0 + i * (cw + 0.2)
    card(s, x, CY, cw, 3.85)
    y = ctitle(s, x + 0.3, CY + 0.2, cw - 0.6, name, '결함 바꾼 시험에서 놓친 불량. x = 종류마다 더한 합성 장수', color=ACC, size=15)
    ch = s.line_chart(x + 0.1, y - 0.1, cw - 0.25, 2.55, N, [(f'실제 {k}장', v) for k, v in zip((1, 2, 4, 8), data)],
                      [ACC_L, ACC_M, ACC, NAVY], 0, 0.25, 0.05, ref=(0.188, None), lay=(0.11, 0.14, 0.86, 0.72))
    pct_axis(ch)
    quiet(ch)
    bullets(s, x + 0.3, CY + 3.25, cw - 0.6, 0.6, [' · '.join(note)], size=12, after=0)
card(s, X0, CY + 4.05, XW, 0.95)
s.text(X0 + 0.3, CY + 4.15, 11.6, 0.36, '답: 실제 장수가 가장 중요하다 (실제 1 → 2 → 4장: 21.7% → 13.3% → 7.9%). 합성은 종류마다 20장 안팎이면 충분하다',
       size=14, color=ACC, bold=True, anchor='m')
s.text(X0 + 0.3, CY + 4.53, 11.6, 0.36, '점선 = PatchCore 18.8%. 실제 8장은 반복 1번 (실제만 5번 평균은 5.1%). 휜 핀은 두 생성 모델 모두 합성을 더하면 나빠짐 (부록 9)',
       size=11, color=MUTED, anchor='m')

s = new('RQ2 읽는 법 - 합성의 값어치를 "실제 몇 장"으로 바꿔 읽는다',
        notes='합성의 값어치를 실제 장수로 바꿔 읽는 방법입니다. 실제 불량만으로 1장부터 10장까지 학습해 곡선을 만들었습니다. '
              '합성을 더한 결과가 이 곡선의 어느 자리에 오는지 읽으면 됩니다. '
              '예를 들어 실제 1장에 cTransGAN 40장을 더하면 18%를 놓치는데, 실제만으로는 약 1.4장일 때 나오는 성적입니다. '
              '실제 4장에 더하면 6.3%로, 실제 약 6.6장과 같습니다.')
card(s, X0, CY, 7.6, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 7.0, '실제만 곡선 (VisA, 반복 5번)', 'x = 실제만 쓴 장수, y = 결함 바꾼 시험에서 놓친 불량')
REAL_K = ['1', '2', '3', '4', '5', '6', '8', '10']
REAL = [.217, .133, .098, .079, .069, .069, .051, .039]
CXY = (X0 + 0.15, y, 7.3, CH - (y - CY) - 0.15)
LAY = (0.1, 0.08, 0.86, 0.78)
ch = s.line_chart(*CXY, REAL_K, [('실제만', REAL)], [NAVY], 0, 0.25, 0.05, lay=LAY, legend=False,
                  ref=(0.180, '실제 1장 + cTransGAN 40장 18.0% → 1.4장'))
pct_axis(ch)
quiet(ch)
s._ref_line(*CXY, LAY, 0.25, 0.063, None)
_yy = CXY[1] + CXY[3] * (LAY[1] + LAY[3] * (1 - 0.063 / 0.25))
s.text(CXY[0] + CXY[2] * LAY[0] + 0.05, _yy + 0.03, 3.6, 0.28, '실제 4장 + cTransGAN 40장 6.3% → 6.6장', size=11, color='495057', bold=True)
c2 = X0 + 7.8
card(s, c2, CY, XW - 7.8, CH)
items(s, c2 + 0.3, CY + 0.35, XW - 8.4, [
    (1, '실제만 곡선을 만든다', '실제 불량만으로 1 · 2 · 3 · 4 · 5 · 6 · 8 · 10장 학습, 각 5번 반복'),
    (2, '합성을 더한 결과를 놓는다', '놓친 불량이 같은 높이(점선)에서 곡선과 만나는 자리'),
    (3, '그 자리의 장수를 읽는다', '예: 6.3% → 실제 약 6.6장. 곡선은 장수가 늘면 나빠지지 않게 다듬음'),
    (4, '불확실성', 'PCB 3개 × 반복 5번 = 15단위를 다시 뽑아 95% 범위')],
    step=1.13, nsize=17, hsize=14, bsize=12, gap=0.38)

s = new('RQ2 결과 - 합성 40장은 실제 1장을 1.4장, 4장을 6.6장으로 만든다',
        notes='이렇게 읽으면 cTransGAN 40장은 실제 1장을 1.4장, 2장을 2.7장, 4장을 6.6장으로 만듭니다. 실제가 많을수록 합성의 값어치도 커집니다. '
              'SD+LoRA는 실제 1장에서는 1.1장으로 효과가 없고, 실제 2장부터는 cTransGAN과 비슷합니다.')
card(s, X0, CY, 7.3, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 6.7, '합성 40장을 더하면 실제 몇 장 효과 (VisA, 반복 5번)')
ch = s.bar_chart(X0 + 0.15, y, 7.0, CH - (y - CY) - 0.15, ['실제 1장', '실제 2장', '실제 4장'],
                 [('실제만', [1, 2, 4]), ('+ SD+LoRA 40장', [1.1, 2.5, 6.1]), ('+ cTransGAN 40장', [1.4, 2.7, 6.6])],
                 [GRAYS, ACC_L, ACC], vmax=8, major=2, fmt='0.0', lay=(0.08, 0.14, 0.9, 0.74), gap=70)
ch.value_axis.tick_labels.number_format = '0'
quiet(ch)
labels(ch, fmt='0.0')
c2 = X0 + 7.5
card(s, c2, CY, XW - 7.5, CH)
s.text(c2 + 0.3, CY + 0.3, 4.2, 0.36, 'cTransGAN 40장', size=17, color=ACC, bold=True)
s.text(c2 + 0.3, CY + 0.68, 4.2, 0.3, '실제만 곡선에서 같은 성적이 나오는 장수', size=12, color=BODY)
stat(s, c2 + 0.3, CY + 1.25, 2.0, '실제 1장일 때', '1.4장')
vline(s, c2 + 2.3, CY + 1.3, CY + 2.15)
stat(s, c2 + 2.5, CY + 1.25, 2.0, '실제 4장일 때', '6.6장')
hline(s, c2 + 0.3, c2 + 4.43, CY + 2.45)
bullets(s, c2 + 0.3, CY + 2.7, 4.13, 2.2, ['실제가 많을수록 합성의 값어치도 큼 (실제 2장 → 2.7장)',
                                         'SD+LoRA는 실제 1장에서 효과 없음 (1.1장), 2장부터 cTransGAN과 비슷',
                                         '95% 범위: cTransGAN 1.2–2.0 · 2.3–3.8 · 4.5–7.8장'], size=13, after=10)

# ================================================================ 5. RQ3
section('5. RQ3 실제 라인', '5  RQ3 실제 라인')
s = new('RQ3 데이터 - 실제 생산 라인(AIHub SMT), 결함 위치는 상자뿐',
        notes='마지막 질문은 실제 생산 라인 데이터입니다. AIHub SMT 데이터에는 기판에 납을 바른 직후의 사전 공정과 부품을 납땜한 뒤의 납땜 공정이 있고, '
              '결함은 6종과 12종입니다. 결함 위치는 상자로만 표시돼 있습니다. '
              '사진은 왼쪽이 정상, 오른쪽이 불량 예시입니다. 사전 공정은 이웃한 납이 이어진 납쇼트, 납땜 공정은 부품이 빠진 미삽이고, 빨간 상자가 결함 위치입니다. '
              '여기에도 같은 문제가 있었습니다. 사진 파일 이름의 촬영 코드만으로도 정상과 불량이 갈렸고, 같은 부품 자리를 찍은 사진에서도 불량 쪽이 더 밝았습니다.')
card(s, X0, CY, XW, 3.85)
TW = [1.55, 1.75, 1.75, 4.0, 2.58]
rows = [['공정', '정상 예시', '불량 예시', '결함 종류', '장수'],
        ['사전 공정\n(납을 바른 직후)', '', '', '미납 · 납부족 · 납쇼트\n납볼 · 위치 밀림 · 모양 불량 (6종)',
         '시험: 정상 1,000 · 불량 300장\n학습 불량: 종류마다 1 · 4 · 16장'],
        ['납땜 공정\n(부품 납땜 뒤)', '', '', '미납 · 냉납 · 밀림 · 쇼트 · 오삽\n미삽(부품 빠짐) · 역삽 · 뒤집힘\n일어섬 · 납볼 · 핀홀 · 납고드름 (12종)',
         '시험: 정상 1,000 · 불량 540장\n학습 불량: 종류마다 1 · 4 · 16장']]
RH, HH = 1.5, 0.45
table(s, X0 + 0.3, CY + 0.25, TW, rows, size=12, row_h=RH, header_h=HH, align=['c', 'c', 'c', 'l', 'l'])
for i, cat in enumerate(('pre', 'solder')):          # AIHub 사진은 발표용(_local)에만, 공개본은 자리만 (맨 아래 저장 부분)
    for j, kind in enumerate(('normal', 'defect')):
        x, y = X0 + 0.3 + TW[0] + j * TW[1], CY + 0.25 + HH + i * RH
        cell = (x + 0.12, y + 0.08, TW[1] - 0.24, RH - 0.16)
        PHOTO_CELLS.append((s, cell))
        if HAVE_SMT:
            side = min(cell[2], cell[3])
            px = cell[0] + (cell[2] - side) / 2
            s.image(SMT_FIG[(cat, kind)], px, cell[1], w=side, name='SMT photo')
            if kind == 'defect':
                s.box(px + side - 0.76, cell[1] + side - 0.32, 0.7, 0.26, {'pre': '납쇼트', 'solder': '미삽'}[cat], fill=BAD, size=10,
                      color=WHITE, bold=True, radius=0.05, margin=(0, 0), name='SMT photo')
        else:
            s.box(*cell, '사진 자리\n(512×512)', fill=None, line=GRAYS, dash=True, size=11, color=MUTED, name='SMT photo placeholder')
card(s, X0, CY + 4.05, XW, 0.95)
s.text(X0 + 0.3, CY + 4.13, 1.6, 0.36, '같은 문제', size=14, color=BAD, bold=True, anchor='m')
s.text(X0 + 1.9, CY + 4.13, 10.0, 0.8, '사진 파일 이름에 붙은 촬영 코드만으로도 정상과 불량이 갈림 (AUROC 0.92). '
       '같은 부품 자리를 찍은 사진 62쌍에서도 불량 쪽이 더 밝음', size=13, color=BODY)
s.text(X0 + 1.9, CY + 4.6, 10.0, 0.3, f'18개 결함 유형별 정상 · 불량 예시와 설명: 부록 {APP_SMT}–{APP_SMT + 5}', size=11, color=MUTED)

s = new('RQ3 실험 - 같은 흐름을 SMT에, 다른 점만 맞춘다',
        notes='흐름은 VisA와 같고, 다른 점만 맞췄습니다. 위치 모델의 정답은 픽셀 단위 대신 상자입니다. '
              '생성 모델은 공정마다 K장만으로 새로 학습했고, 결함 상자가 부품 크기와 맞지 않아 원래 부품이 반쯤 남던 문제를 고쳐 비슷한 크기의 자리에 그리도록 했습니다. '
              '실제 불량은 1, 4, 16장, 반복은 세 번이고, 실제만은 32장까지 다섯 번 돌렸습니다. SMT에는 결함 바꾼 시험을 만들지 않아서 원래 시험셋으로 잽니다.')
card(s, X0, CY, XW, 1.55)
flow = [('실제 불량 K장\n+ 결함 상자', IO), ('생성 모델\ncTransGAN · SD+LoRA', TRAIN), ('합성 불량 + 상자\n(정상 사진 위에)', IO),
        ('위치 모델\n상자 = 정답', TRAIN), ('평가\n원래 시험셋', IO)]
for i, (t_, st) in enumerate(flow):
    x = X0 + 0.3 + i * (2.05 + 0.36)
    s.box(x, CY + 0.35, 2.05, 0.85, t_, size=12, radius=0.08, **st)
    if i < 4:
        arrow(s, x + 2.09, CY + 0.77, x + 2.37, CY + 0.77)
cw = (XW - 3 * 0.18) / 4
for i, (h, b) in enumerate([('결함 위치', 'VisA는 픽셀 단위, SMT는 상자뿐. 상자 안 전체를 결함으로 보고 학습'),
                            ('합성 자리 고침', '결함 상자가 부품 크기와 안 맞아 원래 부품이 반쯤 남음 → 비슷한 크기의 부품 자리에, 미납은 리드 끝, 미삽은 부품 자리 전체'),
                            ('조건', '실제 1 · 4 · 16장 × 반복 3번, 합성 0–40장. 실제만은 1–32장 × 반복 5번, 사전 공정은 합성 160장까지'),
                            ('평가', '원래 시험셋(공정마다 정상 1,000장), 정상 100장 중 1장 기준. 결함 바꾼 시험은 만들지 않음')]):
    x = X0 + i * (cw + 0.18)
    card(s, x, CY + 1.75, cw, 3.25)
    s.text(x + 0.25, CY + 1.97, 0.5, 0.42, str(i + 1), size=18, color=ACC, bold=True)
    s.text(x + 0.25, CY + 2.43, cw - 0.45, 0.36, h, size=14, color=INK, bold=True, anchor='m')
    s.text(x + 0.25, CY + 2.85, cw - 0.45, 2.05, b, size=12, color=BODY)

s = new('RQ3 결과 - 실제 라인에서도 위치 모델은 결함을 보고, 합성은 사전 공정에서 통했다', size=23,
        notes='정상 사진이 다양한 실제 라인에서는 정상만 쓰는 PatchCore가 사전 공정 불량의 91%를 놓칩니다. 위치 모델은 불량 1장만으로 20%만 놓칩니다. '
              '결함 바꾼 시험은 없지만, 결함을 지운 사진을 불량이라 한 비율이 사전 14%, 납땜 6%로 낮아서 배경이 아니라 결함을 보고 맞힌 것으로 봅니다. '
              '합성을 더하면 사전 공정에서는 모든 장수에서 좋아졌고, 어려운 납땜 공정은 실제가 적을 때만 조금 좋아지고 16장에서는 나빠졌습니다.')
card(s, X0, CY, 7.3, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 6.7, 'SMT 위치 모델의 놓친 불량', '원래 시험, 정상 100장 중 1장 기준, 종류마다 합성 40장, 반복 3번')
rows, colors = [['공정', '실제', '실제만', '+ SD+LoRA', '+ cTransGAN']], [[None] * 5]
for cat, ko in (('pre', '사전'), ('solder', '납땜')):
    for k in (1, 4, 16):
        a, b, c = sv(cat, k, 'none'), sv(cat, k, 'sd_lora_fix:40:0.5'), sv(cat, k, 'ctransgan:40:0.5')
        rows.append([ko if k == 1 else '', f'{k}장', pct(a), pct(b), pct(c)])
        colors.append([None, None, MUTED, ACC if b < a else BAD, ACC if c < a else BAD])
table(s, X0 + 0.3, y, [1.05, 0.95, 1.5, 1.6, 1.6], rows, size=14, row_h=0.48, colors=colors)
s.text(X0 + 0.3, CY + 4.6, 6.7, 0.3, '파랑 = 실제만보다 좋아짐, 주황 = 나빠짐', size=11, color=MUTED)
c2 = X0 + 7.5
card(s, c2, CY, XW - 7.5, CH)
pc1 = sv('pre', 1, 'none')
items(s, c2 + 0.3, CY + 0.35, 4.2, [
    (1, f'91% → {pc1 * 100:.0f}%', f'사전 공정. 정상만 쓰는 PatchCore는 91%를 놓침, 위치 모델은 불량 1장으로 {pct(pc1)}'),
    (2, '결함을 보고 맞힘', '결함 지운 사진을 불량이라 한 비율: 사전 14%, 납땜 6% (VisA 위치 모델 14%와 비슷)'),
    (3, '합성은 사전 공정에서 통함', '사전은 모든 장수에서 좋아짐. 납땜은 실제 1 · 4장에서만'),
    (4, '납땜 16장은 나빠짐', '불량 절반을 늘 합성으로 채운 영향일 수 있음 → 비율을 줄여 확인 (뒤에서)')],
    step=1.13, nsize=17, hsize=14, bsize=12, gap=0.38)

s = new('RQ3 결과 - 실제가 적을 때 합성 40장은 실제를 1.2–2.6배로 늘린 효과', size=24,
        notes='RQ2처럼 SMT에서도 합성의 값어치를 실제 장수로 바꿔 봤습니다. 실제만은 1장부터 32장까지 다섯 번 반복해 곡선을 만들었습니다. '
              '실제가 1장, 4장일 때는 VisA와 같은 흐름입니다. 실제 4장에 합성 40장을 더하면 두 공정, 두 생성 모델 모두 실제 5.4에서 6.3장 효과로, VisA와 거의 같습니다. '
              '실제 16장에서는 공정마다 달랐습니다. 사전 공정은 실제 32장보다도 덜 놓쳤고, 납땜 공정은 나빠졌습니다.')
card(s, X0, CY, 6.5, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 5.9, '합성 40장을 더하면 실제 몇 장 효과 (SMT)', '실제만 곡선: 1 · 2 · 4 · 8 · 16 · 32장 × 반복 5번')
eq = [('사전', '1장', '2.4장', '2.6장', ACC), ('', '4장', '6.0장', '6.3장', ACC), ('', '16장', '32장보다 좋음', '32장보다 좋음', ACC),
      ('납땜', '1장', '1.2장', '1.5장', ACC), ('', '4장', '5.9장', '5.4장', ACC), ('', '16장', '나빠짐', '크게 나빠짐', BAD)]
rows = [['공정', '실제', '+ SD+LoRA 40장', '+ cTransGAN 40장']] + [list(r[:4]) for r in eq]
colors = [[None] * 4] + [[None, None, r[4], r[4]] for r in eq]
table(s, X0 + 0.3, y, [0.9, 0.8, 2.1, 2.1], rows, size=13.5, row_h=0.46, colors=colors)
s.text(X0 + 0.3, CY + 4.55, 5.9, 0.4, '16장: 사전 3.4% → 1.6% · 1.9% (실제 32장은 2.9%), 납땜 40.9% → 49.6% · 72.8%', size=10.5, color=MUTED)
c2 = X0 + 6.7
card(s, c2, CY, XW - 6.7, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 4.9, '실제 4장 + 합성 40장: VisA와 같다', '막대 = 실제 몇 장 효과, 점선 = 실제만 4장')
ch = s.bar_chart(c2 + 0.15, y, 5.2, 2.75, ['VisA', 'SMT 사전', 'SMT 납땜'],
                 [('+ SD+LoRA 40장', [6.1, 6.0, 5.9]), ('+ cTransGAN 40장', [6.6, 6.3, 5.4])],
                 [ACC_L, ACC], vmax=8, major=2, fmt='0.0', lay=(0.1, 0.16, 0.87, 0.7), gap=80, ref=(4, None))
ch.value_axis.tick_labels.number_format = '0'
quiet(ch)
labels(ch, fmt='0.0')
bullets(s, c2 + 0.3, CY + 3.75, 4.9, 1.2, ['실제 1장 → 1.2–2.6장, 4장 → 5.4–6.3장 (두 공정 · 두 생성 모델)',
                                         '실제 16장은 공정마다 다름: 사전은 계속 좋아짐, 납땜은 나빠짐'], size=12.5, after=6)

R25 = {(cat, g): {k: sv(cat, k, f'{m}:40:0.25') for k in (1, 4, 16)} for cat in ('pre', 'solder') for g, m in GEN.items()}
R50 = {(cat, g): {k: sv(cat, k, f'{m}:40:0.5') for k in (1, 4, 16)} for cat in ('pre', 'solder') for g, m in GEN.items()}
s = new('합성 비율을 줄이면 - 납땜 16장의 해는 SD+LoRA는 비율 탓, cTransGAN은 합성 탓', size=24,
        notes='납땜 16장에서 나빠진 이유를 확인하려고, 학습 때 불량 4장 중 합성을 2장에서 1장으로 줄여 다시 돌렸습니다. '
              'SD+LoRA는 49.6%에서 42.1%로, 실제만 쓴 40.9%와 거의 같아졌습니다. 해의 대부분은 비율 탓이었습니다. '
              'cTransGAN은 72.8%에서 60.1%로 줄었지만 여전히 실제만보다 훨씬 나쁩니다. 이 세트는 결함을 실제의 8.7배로 과장해서, 합성 자체가 문제였습니다. '
              '실제가 1장, 4장일 때는 비율을 줄여도 이득이 그대로였습니다.')
card(s, X0, CY, 7.0, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 6.4, '학습 때 불량 4장 중 합성을 2장 → 1장으로', 'SMT 위치 모델의 놓친 불량, 반복 3번. 파랑 = 실제만보다 좋아짐, 주황 = 나빠짐')
rows, colors = [['', '실제', '실제만', '합성 2/4', '합성 1/4']], [[None] * 5]
for cat, ko in (('pre', '사전'), ('solder', '납땜')):
    for g in GEN:
        for k in (1, 16):
            a, b, c = sv(cat, k, 'none'), R50[(cat, g)][k], R25[(cat, g)][k]
            rows.append([f'{ko} · {g}' if k == 1 else '', f'{k}장', pct(a), pct(b), pct(c)])
            colors.append([None, None, MUTED, ACC if b < a else BAD, MUTED if c is None else (ACC if c < a else BAD)])
table(s, X0 + 0.3, y, [2.2, 0.8, 1.15, 1.15, 1.1], rows, size=12.5, row_h=0.38, colors=colors)
c2 = X0 + 7.2
card(s, c2, CY, XW - 7.2, CH)
sd16, cg16, re16 = R25[('solder', 'SD+LoRA')][16], R25[('solder', 'cTransGAN')][16], sv('solder', 16, 'none')
items(s, c2 + 0.3, CY + 0.35, 4.4, [
    (1, 'SD+LoRA: 비율 탓이 컸다', f'납땜 16장 49.6% → {pct(sd16)} (실제만 {pct(re16)}). 해가 거의 사라짐'),
    (2, 'cTransGAN: 합성 자체가 문제', f'72.8% → {pct(cg16)}, 여전히 실제만보다 나쁨. 결함을 실제의 8.7배로 과장한 세트 (부록 6)'),
    (3, '실제가 적을 때는 비율과 상관없이 도움', '실제 1 · 4장: 2/4와 1/4 모두 실제만보다 좋아짐'),
    (4, '현장에서는', '실제가 많아지면 합성 비중을 줄이고, 과장된 합성 세트는 뺀다')],
    step=1.13, nsize=17, hsize=14, bsize=12, gap=0.38)

s = new('왜 합성은 실제 불량을 대신하지 못할까 - 새 정보가 적고, 결함을 과장한다', size=24,
        notes='합성은 실제가 적을 때 도움이 되지만 실제 불량을 대신하지는 못했습니다. 이유는 세 가지입니다. '
              '첫째, 합성은 같은 실제 1장을 보고 만든 변형이라 새로운 정보가 적습니다. 160장을 만들어도 시험 결함의 75%는 비슷한 조각이 합성에 없습니다. '
              '둘째, 합성은 결함을 실제보다 훨씬 눈에 띄게 그립니다. 납쇼트는 실제 결함의 4%만 정상과 확실히 달라 보이는데 합성은 66%였고, 이 종류는 합성을 더하자 더 놓쳤습니다. '
              '셋째, 휜 핀이나 부품 빠짐처럼 생성 모델이 잘 못 그리는 종류가 있습니다.')
cw = (XW - 2 * 0.2) / 3
x = X0
card(s, x, CY, cw, 4.1)
y = ctitle(s, x + 0.3, CY + 0.2, cw - 0.6, '1  새로운 정보가 적다', '시험 결함 조각 중 합성에 비슷한 조각이 있는 비율', color=ACC)
ch = s.bar_chart(x + 0.1, y, cw - 0.2, 2.1, ['20장', '40장', '80장', '160장'], [('실제 1장', [0.085, 0.124, 0.176, 0.248]),
                                                                             ('실제 4장', [0.141, 0.212, 0.288, 0.384])],
                 [ACC_L, ACC], vmax=0.5, major=0.25, fmt='0%', lay=(0.13, 0.16, 0.84, 0.64), gap=60)
pct_axis(ch)
quiet(ch)
ch.plots[0].has_data_labels = False
s.text(x + 0.3, CY + 3.15, cw - 0.6, 0.9, '같은 1장을 보고 만든 변형. 160장을 만들어도 시험 결함의 75%는 비슷한 조각이 없음 (SMT 사전, SD+LoRA)', size=12, color=BODY)
x = X0 + cw + 0.2
card(s, x, CY, cw, 4.1)
y = ctitle(s, x + 0.3, CY + 0.2, cw - 0.6, '2  결함을 너무 눈에 띄게 그린다', '결함 상자 안 칸 중 정상과 확실히 다른 비율', color=ACC)
ch = s.bar_chart(x + 0.1, y, cw - 0.2, 2.1, ['실제 결함', '합성 (cTransGAN)'], [('비율', [0.04, 0.66])], [GRAYS], vmax=0.8, major=0.4,
                 fmt='0%', legend=False, point_colors=[GRAYS, BAD], lay=(0.13, 0.1, 0.84, 0.72), gap=90)
pct_axis(ch)
quiet(ch)
labels(ch, fmt='0%')
s.text(x + 0.3, CY + 3.15, cw - 0.6, 0.9, 'SMT 사전 공정 납쇼트. 실제는 미묘한데 합성은 과장 → 합성을 더하자 놓친 불량이 14%p 늘어남', size=12, color=BODY)
x = X0 + 2 * (cw + 0.2)
card(s, x, CY, cw, 4.1)
y = ctitle(s, x + 0.3, CY + 0.2, cw - 0.6, '3  못 그리는 종류가 있다', color=ACC)
items(s, x + 0.3, y + 0.15, cw - 0.6, [('·', '휜 핀 (VisA)', '원래 곧은 핀이 남는 문제. 지운 뒤에도 합성을 더하면 더 놓침 (14.8% → 21–23%)'),
                                      ('·', '미삽 (SMT, 부품 빠짐)', '빈 자리에 부품을 다시 그려 넣음. 놓친 비율 80% 그대로')],
      step=1.4, nsize=16, hsize=13.5, bsize=12, gap=0.36, bh=0.9)
card(s, X0, CY + 4.3, XW, 0.7)
s.text(X0 + 0.3, CY + 4.3, 11.6, 0.7, '결론: 합성은 실제 불량을 대신하지 못하고 보완한다. 실제를 더 모을 수 있으면 그게 먼저다',
       size=15, color=ACC, bold=True, anchor='m')

# ================================================================ 6. 정리
section('6. 정리', '6  정리')
s = new('의의 - 적은 불량 사진을 제대로 쓰는 법을 숫자로 보였다',
        notes='정리하겠습니다. 첫째, 같은 불량 1장이라도 결함 위치로 배우면 놓친 불량이 87%에서 24%로 줄고, 실제 3장부터는 정상만 쓰는 PatchCore보다 낫습니다. '
              '둘째, 원래 시험 점수로는 안 보이는 배경 보고 맞히기를 재는 시험을 만들었고, 공개 방법도 여기서 대부분을 놓쳤습니다. '
              '셋째, 합성의 값어치를 실제 장수로 쟀습니다. 실제가 적을 때 합성 40장은 실제를 최대 2.6배로 늘린 효과였고, 대신하지 못하는 이유도 확인했습니다. '
              '넷째, 실제 라인에서도 상자 위치만으로 학습한 위치 모델이 결함을 보고 잡았습니다.')
cw = (XW - 0.2) / 2
for i, (h, b, big) in enumerate([
        ('라벨을 쓰는 방식이 결과를 가른다', '같은 불량 1장이라도 결함 위치로 배우면 놓친 불량 87% → 24% (사진 단위 DevNet 대비). 실제 3장부터 PatchCore보다 확실히 나음', '87% → 24%'),
        ('"배경 보고 맞히기"를 재는 시험', '원래 시험 점수로는 안 보이는 문제를 드러냄. 공개 방법 DRA · DevNet도 이 시험에서 불량의 86–91%를 놓침 (부록 4)', '86–91%'),
        ('합성의 값어치를 실제 장수로', '실제 1–4장일 때 합성 40장 = 실제를 최대 2.6배로 늘린 효과. 대신하지 못하는 이유도 확인 (새 정보 부족, 과장)', '최대 2.6배'),
        ('실제 라인에서도 확인', '정상만 쓰는 방식은 91%를 놓치지만, 상자 위치만으로 배운 위치 모델은 결함을 보고 잡음', '91% → 20%')]):
    x, y = X0 + (i % 2) * (cw + 0.2), CY + (i // 2) * 2.6
    card(s, x, y, cw, 2.4)
    s.text(x + 0.3, y + 0.25, 0.5, 0.42, str(i + 1), size=20, color=ACC, bold=True)
    s.text(x + 0.8, y + 0.27, cw - 3.1, 0.4, h, size=15, color=INK, bold=True, anchor='m')
    s.text(x + cw - 2.3, y + 0.2, 2.05, 0.5, big, size=20, color=ACC, bold=True, align='r', anchor='m')
    s.text(x + 0.8, y + 0.85, cw - 1.1, 1.35, b, size=13, color=BODY)

s = new('현장에서 불량 K장을 받았다면 - 이 순서로 쓴다',
        notes='현장에서 불량 사진 몇 장을 받았을 때 따를 순서입니다. 먼저 불량 사진에 결함 위치를 상자로 표시합니다. '
              '위치 모델을 밝기와 색 바꾸기와 함께 학습합니다. 실제가 종류마다 4장 이하로 적으면 합성 20에서 40장을 더하고, 실제가 많아지면 합성 비중을 줄이고 과장된 합성 세트는 뺍니다. '
              '배포 전에는 결함을 지운 사진은 정상으로, 붙인 결함은 불량으로 맞히는지 확인합니다. 그리고 언제나 먼저, 실제 불량을 한 장이라도 더 모으는 것이 가장 효과가 큽니다.')
steps = [('결함 위치 표시', '불량 사진에 결함 위치를 상자로 표시'), ('위치 모델 학습', '밝기 · 색을 무작위로 바꾸며 학습'),
         ('합성은 적을 때만', '종류마다 4장 이하면 합성 20–40장. 많아지면 합성 비중을 줄이고, 과장된 세트는 뺀다'),
         ('배포 전 확인', '결함 지운 사진은 정상으로, 붙인 결함은 불량으로 맞히는지')]
cw = (XW - 3 * 0.3) / 4
for i, (h, b) in enumerate(steps):
    x = X0 + i * (cw + 0.3)
    card(s, x, CY, cw, 3.4)
    s.box(x + 0.25, CY + 0.3, 0.5, 0.5, str(i + 1), fill=ACC, size=16, color=WHITE, bold=True, radius=0.1, margin=(0, 0))
    s.text(x + 0.25, CY + 1.0, cw - 0.45, 0.4, h, size=15, color=INK, bold=True, anchor='m')
    s.text(x + 0.25, CY + 1.5, cw - 0.45, 1.8, b, size=13, color=BODY)
    if i < 3:
        arrow(s, x + cw + 0.03, CY + 1.7, x + cw + 0.27, CY + 1.7)
s.box(X0, CY + 3.65, XW, 0.85, '언제나 먼저: 실제 불량을 1장이라도 더 모은다 (실제 1 → 2장이 합성 40장보다 낫다)', fill=ACC_T, size=16,
      color=ACC, bold=True, radius=0.14, align='l', margin=(0.35, 0.05))

s = new('한계와 다음 실험',
        notes='한계입니다. SMT는 반복이 세 번뿐이고, 시험용 휜 핀 불량은 17장뿐이라 종류별 결론은 약합니다. '
              'SMT에는 결함 바꾼 시험을 만들지 않아서, SMT 숫자는 실제보다 좋게 나왔을 수 있습니다. '
              '실제 불량 1–2장일 때는 PatchCore와 통계적으로 차이가 없습니다. '
              '학습에 없던 결함 종류는 시험하지 않았고, VisA는 픽셀 단위로 학습해서 상자만으로 충분한지는 SMT에서만 봤습니다. 오른쪽은 이를 메울 다음 실험입니다.')
card(s, X0, CY, 7.3, CH)
y = ctitle(s, X0 + 0.3, CY + 0.25, 6.7, '한계', color=BAD, size=15)
for i, (h, b) in enumerate([('표본이 작다', 'SMT는 반복 3번, 시험용 휜 핀 불량은 17장 → 종류별 경향은 약한 근거'),
                            ('SMT 시험', '결함 바꾼 시험이 없음. 결함 지운 사진 확인은 했지만 숫자가 좋게 나왔을 수 있음'),
                            ('실제 1–2장', 'PatchCore와 통계적으로 차이 없음. PatchCore 성적은 PCB마다 6–29%로 크게 다름'),
                            ('처음 보는 결함 · 상자', '학습에 없던 결함 종류는 시험 안 함. VisA는 픽셀 단위로 학습해 상자만으로 충분한지는 SMT에서만'),
                            ('원래 시험 점수', '정상 · 불량 구분 점수는 최신 방법보다 조금 낮음 (0.971 vs 0.992, 부록 4)')]):
    yy = y + 0.05 + i * 0.8
    s.text(X0 + 0.3, yy, 1.9, 0.7, h, size=13.5, color=INK, bold=True, anchor='m')
    s.text(X0 + 2.25, yy, 4.75, 0.7, b, size=12.5, color=BODY, anchor='m')
    if i < 4:
        hline(s, X0 + 0.3, X0 + 7.0, yy + 0.76)
c2 = X0 + 7.5
card(s, c2, CY, XW - 7.5, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 4.1, '다음 실험', color=ACC, size=15)
bullets(s, c2 + 0.3, y + 0.1, 4.1, 3.9, ['SMT에도 결함 바꾼 시험 만들기 → SMT 숫자 다시 확인',
                                       'VisA를 상자로 학습 → 상자만으로 충분한지',
                                       '결함 종류 하나씩 빼고 학습 → 처음 보는 종류도 잡는지',
                                       '과장 정도로 합성 세트를 미리 거르기 → 납땜 cTransGAN 같은 세트를 학습 전에 빼기'],
        size=13.5, after=14)

MAIN_COUNT = len(d.slides)

# ================================================================ 부록
section('부록', '부록 (질의응답용)')
deck_v4_appendix.build(d, FIG, dict(DATA=FIXED, AUG=TRAIN, ADM=TRAIN, TEN=IO, LOSS=LOSSB, pct=pct, pct_axis=pct_axis))


def app(title, notes=None, size=24):
    return new(f'부록 {len(d.slides) - MAIN_COUNT + 1}. {title}', notes=notes, size=size)


def type_card(s, x, y, w, h, img_n, img_d, name, sub, desc, count, smt=False):
    """[정상 | 불량] 사진 + 유형 이름 · 설명. smt=True면 발표용에만 사진, 공개본은 자리 표시."""
    card(s, x, y, w, h)
    side = min(h - 0.62, (w - 2.9) / 2)
    for k, (img, lab, col) in enumerate(((img_n, '정상', MUTED), (img_d, '불량', BAD))):
        px, py = x + 0.18 + k * (side + 0.1), y + 0.47
        s.text(px, py - 0.32, side, 0.28, lab, size=11, color=col, bold=True, anchor='m')    # 사진 위에 (결함을 가리지 않게)
        if smt:
            PHOTO_CELLS.append((s, (px, py, side, side)))
        if (not smt) or HAVE_SMT:
            s.image(img, px, py, w=side, name='SMT photo' if smt else None)
        else:
            s.box(px, py, side, side, '사진 자리', fill=None, line=GRAYS, dash=True, size=10, color=MUTED, name='SMT photo placeholder')
    tx = x + 0.18 + 2 * side + 0.35
    tw = x + w - tx - 0.2
    s.text(tx, y + 0.15, tw, 0.34, name, size=15, color=ACC, bold=True, anchor='m')
    if sub:
        s.text(tx, y + 0.5, tw, 0.26, sub, size=10.5, color=MUTED, anchor='m')
    dy_ = y + (0.82 if sub else 0.58)
    s.text(tx, dy_, tw, y + h - 0.42 - dy_, desc, size=12, color=BODY)
    s.text(tx, y + h - 0.45, tw, 0.3, count, size=10.5, color=MUTED, anchor='m')


VTYPES = [('bent', '휜 핀', '커넥터 핀이 휘어 제자리에서 벗어남'), ('melt', '녹은 납 · 부품', '납이나 부품이 열에 녹아 뭉개짐'),
          ('scratch', '긁힘', '기판이나 부품 표면에 긁힌 자국'), ('missing', '부품 누락', '있어야 할 부품(핀 · 덮개 등)이 빠짐')]
s = app('VisA 데이터 - 공개 산업 이상탐지 데이터셋, PCB 3종을 썼다',
        notes='VisA 데이터 설명입니다. 아마존이 공개한 산업 이상탐지 데이터셋으로, 물체 12종 가운데 회로 기판 PCB1부터 PCB3까지 세 종류를 썼습니다. '
              '사진은 약 1400×1000 크기의 컬러 사진이고, 결함 위치가 픽셀 단위 마스크로 표시돼 있습니다. 다음 세 쪽에 PCB마다 결함 유형별 정상과 불량 예시가 있습니다.')
card(s, X0, CY, 6.1, CH)
items(s, X0 + 0.3, CY + 0.35, 5.5, [
    (1, '무엇인가', 'Amazon이 공개한 산업 이상탐지 데이터셋 (Zou et al., ECCV 2022). 물체 12종, CC BY 4.0'),
    (2, '우리가 쓴 것', 'PCB1–3, 서로 다른 회로 기판 3종. PCB4는 결함 종류가 다르고 여러 종류가 섞인 사진이 56%라 뺐다'),
    (3, '사진', '약 1,400×1,000 컬러 사진. 불량 사진만 따로 찍혀 더 밝다 (본편 함정)'),
    (4, '라벨', '결함마다 픽셀 단위 마스크. 마스크 값이 결함 종류(1–4)')], step=1.12, nsize=17, hsize=14, bsize=12, gap=0.38)
c2 = X0 + 6.3
card(s, c2, CY, XW - 6.3, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 5.3, 'PCB별 장수', '결함 유형 칸 = 그 유형 하나만 있는 불량 사진 수')
rows = [['', '정상', '불량', '휜 핀', '녹은 납', '긁힘', '누락', '섞임']]
for pcb in ('pcb1', 'pcb2', 'pcb3'):
    c, ty = VISA[pcb], VISA[pcb]['types']
    rows.append([pcb.upper(), f"{c['normal']:,}", str(c['defect']), str(ty['bent']), str(ty['melt']), str(ty['scratch']),
                 str(ty['missing']), str(c['multi'])])
table(s, c2 + 0.3, y, [0.8, 0.75, 0.6, 0.6, 0.7, 0.6, 0.6, 0.6], rows, size=12, row_h=0.42)
y2 = y + 4 * 0.42 + 0.3
s.text(c2 + 0.3, y2, 5.3, 0.3, '우리 분할 (PCB마다, 분할 seed 7017)', size=13, color=ACC, bold=True)
bullets(s, c2 + 0.3, y2 + 0.38, 5.3, 1.6, ['정상: 학습 500 · 검증 200 · 시험 약 300', '불량: 유형마다 10장을 학습 후보로 떼어 둠 (PCB2 휜 핀만 8장), 나머지 60장 안팎은 시험',
                                           '학습에는 후보 중 K = 1 · 2 · 4 · 8장 (실제만은 10장까지)'], size=12, after=5)

for pcb in ('pcb1', 'pcb2', 'pcb3'):
    s = app(f'VisA {pcb.upper()} 결함 유형 4가지 - 정상과 불량을 같은 자리로 확대',
            notes=f'VisA {pcb.upper()}의 결함 유형 네 가지입니다. 왼쪽은 정상 사진을 불량 사진에 맞춰 같은 자리를 자른 것이고, 오른쪽은 불량 사진이며 빨간 선이 결함 위치입니다.')
    cw = (XW - 0.2) / 2
    for i, (t_, ko, desc) in enumerate(VTYPES):
        x, y = X0 + (i % 2) * (cw + 0.2), CY + (i // 2) * 2.55
        type_card(s, x, y, cw, 2.4, FIG / f'app_visa_{pcb}_{t_}_normal.jpg', FIG / f'app_visa_{pcb}_{t_}_defect.jpg', ko, t_,
                  desc, f'불량 {VISA[pcb]["types"][t_]}장 (이 유형만)')
    s.text(X0, CY + 5.08, XW, 0.3, '정상 = 시험 정상 사진을 불량 사진에 기판 정렬해 같은 칸을 자름. 빨간 선 = 결함 마스크', size=10.5, color=MUTED)

STYPES = {'pre': [('nm', '미납', '', '패드에 납이 인쇄되지 않아 빈 패드'), ('nb', '납부족', '', '패드에 납이 덜 인쇄됨'),
                  ('ns', '납쇼트', '', '이웃한 패드의 납이 이어짐 → 납땜 뒤 합선'), ('pb', '납볼', '', '패드 밖에 작은 납 알갱이가 흩어짐'),
                  ('om', '위치 밀림', '원래 이름: 납좌표 밀림', '납이 패드에서 한쪽으로 밀려 인쇄됨'),
                  ('mf', '모양 불량', '원래 이름: 납형성 불량', '납 모양이 뭉개지거나 번짐')],
          'solder': [('nm', '미납', '', '부품 전극에 납이 붙지 않음'), ('cc', '냉납', '', '납이 덜 녹아 표면이 거칠고 접합이 약함'),
                     ('ml', '밀림', '', '부품이 패드에서 옆으로 밀려 붙음'), ('sh', '쇼트', '', '이웃한 전극 사이가 납으로 이어짐 (합선)'),
                     ('ws', '오삽', '', '다른 부품이 잘못 붙음 (겉보기로는 정상과 비슷)'), ('ms', '미삽', '', '부품이 빠져 패드만 남음'),
                     ('rs', '역삽', '', '방향이 있는 부품이 거꾸로 붙음'), ('fl', '뒤집힘', '', '부품이 뒤집혀 붙음 (글자 면이 아래로)'),
                     ('st', '일어섬', '', '칩 부품의 한쪽이 들려 세워짐'), ('pb', '납볼', '', '부품 주변에 작은 납 구슬이 튐'),
                     ('ph', '핀홀', '원래 이름: 납금감 · 핀홀', '납 표면에 작은 구멍이나 갈라짐'), ('ic', '납고드름', '', '납이 뾰족하게 늘어진 채 굳음')]}
s = app('AIHub SMT 데이터 - 실제 생산 라인의 두 공정 검사 사진',
        notes='AIHub SMT 데이터 설명입니다. 실제 SMT 생산 라인에서 모은 검사 사진입니다. SMT는 기판에 납을 인쇄하고 부품을 올린 뒤 납을 녹여 붙이는 공정입니다. '
              '사전 공정은 납을 바른 직후의 사진이고 결함 6종, 납땜 공정은 부품을 붙인 뒤의 사진이고 결함 12종입니다. '
              '사진은 부품 주변 512×512이고 결함 위치는 상자로 표시돼 있습니다. 약관상 사진은 공개본에서 뺐습니다.')
card(s, X0, CY, 6.1, CH)
items(s, X0 + 0.3, CY + 0.35, 5.5, [
    (1, '무엇인가', 'AIHub 「SMT 공정 개선 멀티모달 데이터」(71900). 실제 SMT 라인의 검사 사진 + 설비 센서 값 (센서는 안 씀)'),
    (2, 'SMT 공정', '기판에 납 인쇄 → 부품 장착 → 납을 녹여 붙임(리플로우). 단계마다 사진으로 검사'),
    (3, '두 공정', '사전 공정 = 납을 바른 직후 (결함 6종), 납땜 공정 = 부품을 붙인 뒤 (결함 12종)'),
    (4, '사진 · 라벨', '부품 주변 512×512 JPEG. 결함 위치는 상자뿐. 불량 상자가 하나라도 있으면 불량 사진'),
    (5, '쓰는 조건', 'AIHub 약관상 재배포 금지 → 사진은 발표용 파일에만, 저장소에는 목록 · 분할표만')],
    step=0.92, nsize=16, hsize=13.5, bsize=11.5, gap=0.34, bh=0.55)
c2 = X0 + 6.3
card(s, c2, CY, XW - 6.3, CH)
y = ctitle(s, c2 + 0.3, CY + 0.25, 5.3, '공정별 장수 (우리 분할, seed 7017)', '한 공정 안에서 학습 · 검증 · 시험 사진이 겹치지 않게 나눔')
rows = [['', '사전 공정', '납땜 공정'],
        ['쓴 사진 (거른 뒤)', '18,390', '6,111'], ['결함 유형', '6종', '12종'],
        ['시험 정상', '1,000', '1,000'], ['시험 불량', '300 (유형마다 40\n+ 섞인 것 60)', '540 (유형마다 40\n+ 섞인 것 60)'],
        ['학습 정상 · 검증 정상', '500 · 200', '500 · 200'], ['학습 불량 후보', '유형마다 32', '유형마다 32'],
        ['학습에 쓴 불량 K', '1 · 4 · 16장', '1 · 4 · 16장']]
table(s, c2 + 0.3, y, [2.0, 1.65, 1.65], rows, size=11.5, row_h=0.42)
s.text(c2 + 0.3, CY + 4.6, 5.3, 0.3, '실제만 곡선은 1–32장. 납땜은 촬영 코드 A · 같은 압축 품질 사진만 씀', size=10.5, color=MUTED)

pages = [('pre', f'사전 공정 결함 6종 ({i + 1}/2, 납을 바른 직후)', STYPES['pre'][i * 4:(i + 1) * 4]) for i in range(2)] + \
        [('solder', f'납땜 공정 결함 12종 ({i + 1}/3, 부품을 붙인 뒤)', STYPES['solder'][i * 4:(i + 1) * 4]) for i in range(3)]
for cat, title, lst in pages:
    s = app(f'SMT {title}', size=24,
            notes=f'SMT {title}입니다. 유형마다 왼쪽이 정상, 오른쪽이 불량이고 빨간 상자가 결함 위치입니다. 모두 시험셋 사진입니다. '
                  '정상 사진은 AIHub가 유형마다 따로 모아 둔 같은 종류 부품의 정상 사진에서 골랐습니다.')
    cw = (XW - 0.2) / 2
    for i, (code, ko, sub, desc) in enumerate(lst):
        x, y = X0 + (i % 2) * (cw + 0.2), CY + (i // 2) * 2.55
        type_card(s, x, y, cw, 2.4, FIG / f'local_only/smt_type_{cat}_{code}_normal.jpg', FIG / f'local_only/smt_type_{cat}_{code}_defect.jpg',
                  ko, sub, desc, '시험 40장 · 학습 후보 32장', smt=True)

assert d.slides[MAIN_COUNT + APP_VISA - 1].pslide.shapes.title.text.startswith(f'부록 {APP_VISA}. VisA 데이터')
assert d.slides[MAIN_COUNT + APP_SMT - 1].pslide.shapes.title.text.startswith(f'부록 {APP_SMT}. AIHub SMT 데이터')

d.save(OUT_LOCAL if HAVE_SMT else OUT)
args = sys.argv[1:]
preview_dir = ROOT / 'ppt_build' / 'preview_v4'
if '--preview' in args:
    i = args.index('--preview')
    preview_dir = Path(args[i + 1])
    del args[i:i + 2]
d.preview(preview_dir, only=[int(a) for a in args[0].split(',')] if args else None)
if HAVE_SMT:                       # 공개본: 덱 전체에서 SMT 사진을 빼고 자리만 남겨 따로 저장
    for sl in d.slides:
        for shp in list(sl.shapes):
            if shp.name == 'SMT photo':
                rid = shp._element.blipFill.blip.rEmbed if shp.shape_type == 13 else None
                shp._element.getparent().remove(shp._element)
                if rid:                        # 그림 파일 연결도 끊어야 pptx 안에 사진이 남지 않는다
                    sl.pslide.part.drop_rel(rid)
    for sl, cell in PHOTO_CELLS:
        sl.box(*cell, '사진 생략\n(AIHub 약관)', fill=None, line=GRAYS, dash=True, size=10, color=MUTED,
               name='SMT photo placeholder')
    d.prs.save(OUT)
    print('발표용 (SMT 사진 포함, git 제외):', OUT_LOCAL)
print('저장:', OUT, f'(본편 {MAIN_COUNT}장 + 부록 {len(d.slides) - MAIN_COUNT}장)', '미리보기:', preview_dir)
for w in d.warnings:
    print('경고:', w)
