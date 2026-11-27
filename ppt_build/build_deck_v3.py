"""15분 발표 pptx (v3). v2를 처음 듣는 사람 기준으로 줄이고 쉽게 고친 판 (본편 24장 → 18장).
흐름: 문제와 RQ1 → 시도 → 함정 → 방법을 바꿈 → RQ1 결과 → RQ2 합성은 얼마나 → RQ3 실제 라인 → 원인 → 의의 · 한계.

    python ppt_build/build_deck_v3.py [--preview 미리보기폴더] [슬라이드번호,...]      (아무 위치에서)
결과: PCB_AD_presentation_v3.pptx (저장소 루트), 미리보기 ppt_build/preview_v3/slide-NN.png
먼저 python ppt_build/make_dataset_figures.py 로 VisA 예시 사진과 장수(ppt_figures/dataset_counts.json)를 만든다.
부록은 deck_v3_appendix.py (v2 부록 14장) + 이 파일 끝의 본편에서 옮긴 4장.

v2에서 바꾼 것
  - 합침: 기존 모델 + 시도, 방법 + 시험셋 예시, 생성 모델 두 구조, RQ2 질문 + 실험, RQ3 데이터 + 실험.
    생성 모델 상세 구조 · 합성 시행착오 · 기존 모델 구조는 부록 15–18로.
  - 결론 고침: 실제 1장에서는 PatchCore와 비기고 3장부터 낫다는 것, SMT 큰 숫자는 위치 모델 값으로,
    실제 16장 결론은 공정마다 다르다는 것과 합성 비율 실험(run_smt_ratio.sh) 결과로.
  - 숫자 고침: cTransGAN 토큰 8×8 픽셀, 결함 크기 2.8%는 주변 여유 포함, VisA K = 8은 본편에서 뺌(반복 1번뿐).

용어 (덱 전체에서 같게 쓴다)
  놓친 불량       = 미검률. 정상 100장 중 1장(원래 시험) 또는 5장(결함 바꾼 시험)을 불량으로 잘못 판정하도록 기준을 맞췄을 때
                    놓친 불량의 비율. 낮을수록 좋다. 결함 바꾼 시험은 정상 쪽 사진이 PCB마다 60장 안팎이라 1% 기준은 한 장에 흔들려 5%를 쓴다
  결함 바꾼 시험  = 팀 문서의 '공정 시험셋'. 불량 사진에서 결함을 지운 사진(정상으로 맞혀야 함)과 그 결함을 정상 사진에 붙인 사진(불량으로 맞혀야 함)을 더한 시험
  밝기·색 바꾸기  = 학습 사진의 밝기 · 조명 · 색 · 선명도를 무작위로 바꾸기 (팀 문서의 '촬영 조건 무작위화')
  사진 단위 모델  = 사진마다 정상/불량 라벨 하나로 배우는 모델(DevNet, MLP)
  위치 모델       = 결함 위치를 정답으로 배우는 모델(seg)
  합성            = 생성 모델이 정상 사진 위에 그린 가짜 결함
  반복 N번        = 학습용 실제 불량을 다르게 뽑아 N번 되풀이 (팀 문서의 '분할 seed')

글꼴: 발표 PC에 Pretendard(무료, OFL)가 있어야 한다. 미리보기는 PRETENDARD_DIR(기본 ~/.local/share/fonts/pretendard)로 그린다.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deck_lib  # noqa: E402
from PIL import Image  # noqa: E402
from pptx.enum.chart import XL_LABEL_POSITION  # noqa: E402
from pptx.util import Pt  # noqa: E402

_fd = Path(os.environ.get('PRETENDARD_DIR', Path.home() / '.local/share/fonts/pretendard'))
deck_lib.FONT = 'Pretendard'
if (_fd / 'Pretendard-Regular.otf').exists():
    deck_lib.PREVIEW_FONTS = (str(_fd / 'Pretendard-Regular.otf'), str(_fd / 'Pretendard-Bold.otf'))

from deck_lib import (COPPER, COPPER_T, DARK, GRAY_L, GRAY_T, GREEN, GREEN_M, INK, LINE, MUTED, TINT, WHITE,  # noqa: E402,F401
                      Deck, Para)
import deck_v3_appendix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'ppt_figures'
OUT = ROOT / 'PCB_AD_presentation_v3.pptx'
VISA = json.loads((FIG / 'dataset_counts.json').read_text(encoding='utf-8'))['visa']

DATA = dict(fill=GRAY_T, line=GRAY_L)
AUG = dict(fill=COPPER_T, line=COPPER)
ADM = dict(fill=TINT, line=GREEN)
TEN = dict(fill=WHITE, line=GRAY_L)
LOSS = dict(fill=WHITE, line=COPPER, dash=True)
K_COLORS = ['9BD3B4', '4FA77F', '0B3B2A']                      # 실제 1 · 2 · 4장 (연한 → 진한 초록)
TRAIN_COLORS = ['A3ADA8', 'E3A869', '9C4F12']                  # 실제만 · +SD+LoRA · +cTransGAN
NEG = '1F6FA8'                                                 # 결함 바꾼 시험의 '정상으로 맞혀야 하는 사진' (파랑 테두리)
GRADCAM = FIG / 's12_s14_gradcam_vs_seg_map.jpg'               # PCB1 녹은 납 첫 행: 원본 · DevNet이 본 곳 · · 위치 모델 지도 · ·
CAM_ORIG, CAM_DEVNET, CAM_SEG = (66, 127, 300, 362), (340, 127, 575, 362), (890, 127, 1124, 362)
FAIR = '결함 바꾼 시험, 정상 100장 중 5장을 잘못 판정하는 기준'
ORIG = '원래 시험, 정상 100장 중 1장을 잘못 판정하는 기준'


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


def smt_v(cat, k, cond):
    return SMT[cat].get((k, cond))


def pct(v, nd=1):
    return '실험 중' if v is None else f'{v * 100:.{nd}f}%'


def header(s, x, y, w, num, text, color=GREEN, size=15):
    s.oval(x, y + 0.02, 0.4, color, str(num), size=13)
    s.text(x + 0.52, y, w - 0.52, 0.44, text, size=size, bold=True, color=color, anchor='m')


def fit_image(s, path, x, y, w, h, pad=0.07):
    """칸 (x, y, w, h) 안에 비율을 지켜 가운데 놓는다."""
    with Image.open(path) as im:
        iw, ih = im.size
    ww = min(w - 2 * pad, (h - 2 * pad) * iw / ih)
    s.image(path, x + (w - ww) / 2, y + (h - ww * ih / iw) / 2, w=ww)


def pct_axis(ch):
    ch.value_axis.tick_labels.number_format = '0%'
    ch.value_axis.tick_labels.number_format_is_linked = False


def labels_inside(ch, fmt='0.0%', white=(2,)):
    """막대 값 글자를 막대 안쪽 끝에. 진한 막대는 흰 글자."""
    for i, ser in enumerate(ch.plots[0].series):
        dl = ser.data_labels
        dl.show_value = True
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.position = XL_LABEL_POSITION.INSIDE_END
        dl.font.size, dl.font.bold = Pt(11), True
        dl.font.color.rgb = deck_lib.rgb(WHITE if i in white else INK)


def pipeline(s, y, stages, h=0.95, size=13, x0=0.6, step=2.5, w=2.1):
    for i, (t, st) in enumerate(stages):
        x = x0 + i * step
        s.box(x, y, w, h, t, size=size, **st)
        if i < len(stages) - 1:
            s.line(x + w + 0.02, y + h / 2, x + step - 0.02, y + h / 2, color=INK, width=1.75)


def stat_card(s, x, y, w, h, big, body, color, fill=GRAY_T, big_size=20, body_size=12):
    s.box(x, y, w, h, '', fill=fill, radius=0.12)
    s.text(x + 0.2, y + 0.1, w - 0.4, 0.5, big, size=big_size, bold=True, color=color)
    s.text(x + 0.2, y + 0.6, w - 0.4, h - 0.65, body, size=body_size)


d = Deck()

# ======================================================================= 표지
d.section('표지')
s = d.new('적은 불량 사진을 최대한 활용하는\nPCB 이상탐지', dark=True, title_size=36,
          notes='검사 현장에는 정상 사진이 대부분이고, 불량 사진은 종류마다 몇 장씩만 쌓입니다. '
                '이 적은 불량 사진을 어떻게 써야 결함을 제대로 보는 모델이 되는지, 세 가지 질문으로 답하겠습니다.')
d.slides[-1].pslide.shapes.title.top = int(2.0 * 914400)
d.slides[-1].pslide.shapes.title.height = int(1.6 * 914400)
s.ops[0] = ('text', 0.6, 2.0, 12.13, 1.6) + s.ops[0][5:]
s.text(0.6, 1.4, 8, 0.45, '2026-2 딥러닝 팀 프로젝트', size=16, color='B8C7BF', bold=True)
s.text(0.6, 3.85, 8.2, 0.9, '현장에 조금씩 쌓이는 불량 사진, 어떻게 써야 결함을 보는 모델이 될까', size=19, color='DCE8E1')
s.text(0.6, 5.9, 8.2, 0.8, ['데이터: VisA PCB(공개) · AIHub SMT(실제 생산 라인)',
                            '생성 모델: cTransGAN · SD+LoRA      탐지 모델: PatchCore · DevNet · 위치 모델'],
       size=14, color='B8C7BF')
for i, (n, t) in enumerate([('RQ1', '불량 사진, 어떻게 쓸까'), ('RQ2', '합성은 얼마나'), ('RQ3', '실제 라인에서도?')]):
    y = 1.6 + i * 1.35
    s.oval(9.3, y, 1.0, [GREEN_M, COPPER, GREEN][i], n, size=17)
    s.text(10.45, y + 0.2, 2.75, 0.6, t, size=18, color=WHITE, bold=True, anchor='m')

# ======================================================================= 1. 문제 · 시도 · 함정
d.section('1. 문제 · 시도 · 함정')
s = d.new('문제: 현장에 쌓이는 불량 사진, 표준 이상탐지는 버린다', '문제 · RQ1', 1, title_size=26,
          notes='검사 라인에는 정상 사진이 대부분이고, 불량 사진은 종류마다 몇 장씩 쌓입니다. '
                '그런데 표준 이상탐지는 정상 사진만으로 학습하고, 이 불량 사진은 쓰지 않습니다. '
                '그래서 첫 번째 질문입니다. 적은 불량 사진을 학습에 쓰면 더 잘 잡을 수 있을까요? 쓴다면 어떻게 써야 할까요?')
cards = [('현장', '정상 사진이 대부분이고, 불량 사진은 종류마다 몇 장씩 쌓인다. 결함 종류도 제각각이다 (휜 핀, 녹은 납, 긁힘, 부품 누락 …)',
          GRAY_T, MUTED),
         ('표준 이상탐지', '정상 사진만으로 학습한다 (PatchCore 등). 쌓인 불량 사진은 쓰지 않는다', TINT, GREEN),
         ('RQ1', '적은 불량 사진을 학습에 쓰면 더 잘 잡을까? 쓴다면 어떻게 써야 할까?', COPPER_T, COPPER)]
for i, (h, body, fill, c) in enumerate(cards):
    y = 1.65 + i * 1.78
    s.box(0.6, y, 6.9, 1.55, '', fill=fill, radius=0.12)
    s.oval(0.85, y + 0.2, 0.5, c, str(i + 1), size=15)
    s.text(1.55, y + 0.2, 5.75, 0.45, h, size=18, bold=True, color=INK if i == 0 else c)
    s.text(1.55, y + 0.68, 5.75, 0.82, body, size=15 if i == 2 else 14, bold=i == 2)
    if i < 2:
        s.line(1.1, y + 1.57, 1.1, y + 1.76, color=GRAY_L, width=2)
s.text(7.85, 1.65, 2.4, 0.4, '정상 사진', size=14, bold=True, align='c')
s.text(10.3, 1.65, 2.4, 0.4, '불량 사진 (빨간 상자 = 긁힘)', size=14, bold=True, color=COPPER, align='c')
s.image(FIG / 's03_defect_size_resolution.png', 7.85, 2.08, w=4.88, crop=(15, 45, 778, 328))
s.text(7.85, 3.95, 4.88, 0.35, 'VisA PCB2. 결함은 사진의 아주 작은 부분이다', size=11, color=MUTED, align='c')
for i, (big, lab, c) in enumerate([('약 500장', '정상 학습 사진\n(PCB마다)', GREEN), ('1–8장', '불량 학습 사진\n(결함 종류마다)', COPPER)]):
    x = 7.85 + i * 2.5
    s.box(x, 4.5, 2.38, 1.75, '', fill=GRAY_T, radius=0.12)
    s.text(x + 0.2, 4.65, 2.0, 0.7, big, size=30, bold=True, color=c)
    s.text(x + 0.2, 5.38, 2.0, 0.75, lab, size=13)
s.text(7.85, 6.45, 4.88, 0.5, '공개 PCB(VisA)로 시작해, 실제 생산 라인(AIHub SMT)에서 확인한다', size=13, bold=True)

s = d.new('데이터 ① VisA PCB: PCB마다 정상 약 1,000장, 불량 100장', '데이터', 1, title_size=26,
          notes='첫 번째 데이터는 공개 데이터셋 VisA의 PCB 세 종류입니다. PCB마다 정상 약 천 장, 불량 백 장이고, '
                '결함은 휜 핀, 녹은 납, 긁힘, 부품 누락 네 종류입니다. 결함 위치는 픽셀 단위로 표시돼 있습니다. '
                '정상 500장으로 학습하고, 불량은 종류마다 몇 장만 학습에 씁니다.')
TW = [0.85, 1.75, 1.75, 0.95, 0.95, 2.6]
rows = [['', '정상 예시', '불량 예시', '정상', '불량', '결함 종류별 불량 장수']]
for pcb in ('pcb1', 'pcb2', 'pcb3'):
    c, ty = VISA[pcb], VISA[pcb]['types']
    rows.append([pcb.upper(), '', '', f"{c['normal']:,}", str(c['defect']),
                 f"녹은 납 {ty['melt']} · 누락 {ty['missing']}\n휜 핀 {ty['bent']} · 긁힘 {ty['scratch']}\n두 종류 이상 {c['multi']}"])
RH, HH = 1.22, 0.5
s.table(0.6, 1.6, TW, rows, size=13, row_h=RH, header_h=HH, align=['c', 'c', 'c', 'c', 'c', 'l'])
for i, pcb in enumerate(('pcb1', 'pcb2', 'pcb3')):
    for j, kind in enumerate(('normal', 'defect')):
        fit_image(s, FIG / f's02_visa_{pcb}_{kind}.jpg', 0.6 + TW[0] + j * TW[1], 1.6 + HH + i * RH, TW[1], RH)
s.text(0.6, 5.83, 8.85, 0.3, '불량 예시의 빨간 상자 = 결함 위치 (PCB1 휜 핀 · PCB2 부품 누락 · PCB3 녹은 납). 사진 크기 약 1400×1000',
       size=11, color=MUTED)
s.text(9.75, 1.6, 2.98, 0.42, '결함 종류 4가지', size=15, bold=True, color=COPPER)
for i, (t_, ko) in enumerate([('bent', '휜 핀'), ('melt', '녹은 납 · 부품'), ('scratch', '긁힘'), ('missing', '부품 누락')]):
    x, y = 9.75 + (i % 2) * 1.55, 2.08 + (i // 2) * 1.85
    s.image(FIG / f's02_visa_type_{t_}.jpg', x, y, w=1.43)
    s.text(x, y + 1.46, 1.43, 0.32, [Para(f'**{ko}**', 12, align='c')])
s.text(9.75, 5.83, 2.98, 0.3, '빨간 선 = 픽셀 단위로 표시된 결함 위치', size=11, color=MUTED)
s.box(0.6, 6.25, 12.13, 0.72, [Para('실험에 쓴 장수 (PCB마다)', 13, GREEN, True, after=2),
                              Para('정상 500장으로 학습. 불량은 종류마다 1 · 2 · 4 · 8장만 학습에 쓰고, 나머지 불량(PCB마다 60장 안팎)은 시험에 쓴다', 13)],
      align='l', anchor='m', margin=(0.2, 0.05), fill=GRAY_T)

s = d.new('시도: 불량 라벨을 쓰자 원래 시험 점수가 좋아졌다', '시도', 1, title_size=26,
          notes='먼저 기존 방식대로 비교했습니다. PatchCore는 정상 사진만 학습하고, DevNet과 MLP는 사진마다 정상인지 불량인지 라벨 하나를 받아 학습합니다. '
                '불량 장수를 늘릴수록 놓친 불량이 꾸준히 줄었고, 불량 8장이면 PatchCore보다 덜 놓쳤습니다. '
                '불량 라벨이 도움이 된다고 보였습니다. 그런데 정말 결함을 보고 맞힌 걸까요?')
s.text(0.6, 1.55, 12.13, 0.4, '세 방식 모두 같은 사진 특징(ImageNet으로 미리 학습한 WRN50, 고정)을 쓰고, 불량 라벨을 쓰는 방식만 다르다',
       size=13, color=MUTED)
for i, (h, body, st) in enumerate([('PatchCore', '정상 사진만 학습\n(불량 사진 안 씀)', DATA),
                                   ('사진 단위 모델 (DevNet · MLP)', '정상 + 불량 K장\n사진마다 라벨 하나', ADM),
                                   ('위치 모델 (뒤에서)', '정상 + 불량 K장\n결함 위치까지', AUG)]):
    x = 0.6 + i * 4.15
    s.box(x, 1.98, 3.83, 0.95, [Para(h, 14, INK, True, align='c', after=2), Para(body, 12, align='c', after=0)], **st)
    if i < 2:
        s.line(x + 3.85, 2.45, x + 4.13, 2.45, color=INK, width=1.75)
s.text(0.6, 3.15, 7.3, 0.4, f'놓친 불량 ({ORIG}). x = 종류마다 학습에 쓴 불량 장수', size=12, color=MUTED)
ch = s.line_chart(0.6, 3.5, 7.3, 3.2, ['1', '2', '4', '8', '10'],
                  [('DevNet', [0.341, 0.234, 0.153, 0.093, 0.080]), ('MLP', [0.563, 0.410, 0.313, 0.177, 0.140])],
                  [GREEN, GRAY_L], 0, 0.6, 0.1, ref=(0.153, 'PatchCore 15.3%'), lay=(0.1, 0.04, 0.87, 0.74))
pct_axis(ch)
stat_card(s, 8.25, 3.2, 4.48, 1.3, '9.3% < 15.3%', '불량 8장을 쓴 DevNet이 정상만 쓰는 PatchCore보다 덜 놓친다', GREEN, big_size=24, body_size=13)
s.box(8.25, 4.7, 4.48, 1.9, [Para('"불량 라벨이 도움이 된다"로 보였다.', 15, INK, True, after=6),
                            Para('그런데 정말 결함을 보고 맞힌 걸까?', 15, COPPER, True)],
      align='l', anchor='m', margin=(0.22, 0.08), fill=COPPER_T)

s = d.new('함정: 모델은 결함이 아니라 배경을 보고 있었다', '함정', 1, title_size=26,
          notes='모델이 어디를 보고 판단했는지 Grad-CAM으로 찍어 봤습니다. 결함은 아래쪽 핀에 있는데, 모델은 기판 전체를 보고 있었습니다. '
                '모델이 본 곳 중 결함 자리는 4%뿐이고, 결함을 지워 정상처럼 만든 사진도 44%를 불량이라고 했습니다. '
                '원인은 데이터였습니다. VisA는 불량 사진만 따로, 더 밝게 찍혀서 사진 밝기와 기판이 놓인 자리만으로도 정상과 불량이 갈립니다. '
                '현장에서도 정상과 불량을 다른 장비로 찍으면 같은 일이 생깁니다. 원래 시험셋도 같은 차이가 있어서 점수로는 드러나지 않습니다.')
cols, CW = [0.6, 4.75, 8.9], 3.83
header(s, cols[0], 1.6, CW, 1, '모델이 본 곳: 결함이 아니다', COPPER)
header(s, cols[1], 1.6, CW, 2, '원인: 불량만 따로, 밝게 찍혔다', COPPER)
header(s, cols[2], 1.6, CW, 3, '현장에서도 생긴다 (예시)', GREEN)
s.text(cols[0], 2.08, CW, 0.35, 'PCB1 녹은 납. 왼쪽 원본(흰 선 = 결함), 오른쪽 모델이 본 곳', size=11, color=MUTED)
s.image(GRADCAM, cols[0], 2.65, w=CW, crop=(CAM_ORIG[0], CAM_ORIG[1], CAM_DEVNET[2], CAM_DEVNET[3]))
s.text(cols[1], 2.08, CW, 0.35, 'PCB2 사진의 평균 밝기. 파랑 = 정상, 주황 = 불량', size=11, color=MUTED)
s.image(FIG / 's03_capture_difference.png', cols[1] + 0.25, 2.42, h=2.3, crop=(875, 503, 1405, 862))
for x, (big, lab) in zip(cols[:2], [('4%', '모델이 본 곳 중 결함 자리 비율. 결함을 지워 정상처럼 만든 사진도 44%를 불량이라 함'),
                                    ('0.93 · 0.85 · 0.76', '결함과 관계없는 사진 밝기와 기판이 놓인 자리만으로 정상과 불량을 가른 점수 (AUROC, PCB1 · 2 · 3)')]):
    s.text(x, 4.85, CW, 0.55, big, size=24, bold=True, color=COPPER)
    s.text(x, 5.4, CW, 0.8, lab, size=13)
X3 = cols[2]
s.box(X3, 2.15, 2.35, 0.8, '생산 라인 검사기\n(카메라 · 조명 A)', size=12, **DATA)
s.line(X3 + 2.37, 2.55, X3 + 2.6, 2.55, color=INK, width=1.5)
s.box(X3 + 2.62, 2.15, 1.21, 0.8, '정상 사진', size=12, bold=True, **ADM)
s.box(X3, 3.15, 2.35, 0.8, '재검 · 수리 단계의\n다른 장비 (카메라 · 조명 B)', size=12, **AUG)
s.line(X3 + 2.37, 3.55, X3 + 2.6, 3.55, color=INK, width=1.5)
s.box(X3 + 2.62, 3.15, 1.21, 0.8, '불량 사진', size=12, bold=True, fill=WHITE, line=COPPER)
s.line(X3 + 3.22, 3.97, X3 + 3.22, 4.2, color=INK, width=1.5)
s.box(X3, 4.22, CW, 0.62, '함께 학습하면 모델은 A와 B의 차이를 배운다', size=12, bold=True, fill=COPPER_T, line=None)
s.text(X3, 5.0, CW, 1.2, 'VisA도 그랬다: 불량 사진만 따로 만들어 찍었다', size=13)
s.box(0.6, 6.3, 12.13, 0.68, '불량 라벨을 그냥 쓰면 모델은 결함 대신 "어떤 환경에서 찍혔나"를 배운다. 원래 시험셋도 같은 차이가 있어 점수로는 드러나지 않는다',
      size=14, fill=COPPER_T, align='l', margin=(0.2, 0.04))

# ======================================================================= 2. 방법을 바꿈 · RQ1
d.section('2. 방법을 바꿈 · RQ1')
s = d.new('방법을 바꾸다: 결함 위치로 배우고, 결함만 바꾼 시험으로 잰다', '방법', 2, title_size=26,
          notes='그래서 방법을 네 가지 바꿨습니다. 첫째, 결함 위치를 정답으로 주고 사진의 칸마다 결함인지 맞히게 합니다. '
                '둘째, 학습 사진의 밝기와 색을 무작위로 바꿔, 찍힌 환경으로는 정상과 불량을 가를 수 없게 합니다. '
                '셋째, 시험도 바꿨습니다. 오른쪽처럼 불량 사진에서 결함만 지운 사진은 정상으로, 그 결함을 정상 사진에 붙인 사진은 불량으로 맞혀야 합니다. '
                '배경만 보는 모델은 여기서 틀립니다. 넷째, 적은 불량은 생성 모델로 늘립니다.')
steps = [('1', '결함 위치로 배운다', '결함은 사진의 0.2–0.5%. 라벨 하나로는 어디가 결함인지 모른다 → 사진의 칸마다 결함인지 맞히게 한다', GREEN),
         ('2', '밝기·색을 바꿔 배운다', '학습 사진의 밝기 · 조명 · 색 · 선명도를 무작위로 바꿔, 찍힌 환경으로는 정상과 불량을 못 가르게 한다', GREEN),
         ('3', '결함만 바꾼 시험으로 잰다', '결함을 지운 사진은 정상으로, 그 결함을 정상 사진에 붙인 사진은 불량으로 맞혀야 한다 (오른쪽)', GREEN),
         ('4', '합성으로 늘린다', '불량 K장으로 생성 모델을 학습해, 정상 사진 위에 가짜 결함을 그린다. 결함 위치도 함께 얻는다', COPPER)]
for i, (n, h, body, c) in enumerate(steps):
    y = 1.6 + i * 1.27
    s.box(0.6, y, 5.3, 1.15, '', fill=COPPER_T if c == COPPER else TINT, radius=0.1)
    s.oval(0.78, y + 0.13, 0.45, c, n, size=14)
    s.text(1.35, y + 0.1, 4.45, 0.4, h, size=15, bold=True, color=c, anchor='m')
    s.text(1.35, y + 0.5, 4.45, 0.62, body, size=12)
FX, FW = 6.2, 6.53
s.text(FX, 1.6, FW, 0.4, '결함 바꾼 시험: 배경은 그대로, 결함만 바꾼 사진 짝', size=15, bold=True, color=GREEN)
sc = FW / 1890
for cx, lab, c in ((230, '시험 불량', COPPER), (610, '결함 지운 사진', NEG), (990, '시험 정상', NEG),
                   (1370, '정상 + 붙인 결함', COPPER), (1750, '정상 + 정상 조각', NEG)):
    s.text(FX + (cx - 45) * sc - 0.68, 2.03, 1.36, 0.35, lab, size=11, bold=True, color=c, align='c')
s.image(FIG / 's13_fair_test_examples.png', FX, 2.38, w=FW, crop=(45, 120, 1935, 1213))
s.text(FX, 6.18, FW, 0.8, '주황 = 불량으로 맞혀야 함, 파랑 = 정상으로 맞혀야 함. 맨 오른쪽은 "붙이는 작업" 자체가 단서가 되지 않는지 보는 대조군. '
       '놓친 불량은 정상 100장 중 5장을 잘못 판정하는 기준으로 잰다 (정상 쪽 사진이 적어 1% 기준은 한두 장에 흔들린다)',
       size=11, color=MUTED)
s.box(0.6, 6.7, 5.3, 0.4, '합성은 늘 정상 사진 위에 → 불량 사진의 촬영 환경이 따라오지 않는다', size=11, fill=GRAY_T,
      align='l', margin=(0.12, 0.02))

s = d.new('생성 모델 두 가지: 불량 K장만 보고 가짜 결함을 그린다', '생성 모델', 2, title_size=26,
          notes='합성 불량을 만든 생성 모델은 두 가지입니다. cTransGAN은 Transformer로 만든 작은 GAN을 불량 K장만으로 처음부터 학습합니다. '
                'SD+LoRA는 큰 데이터로 미리 학습된 그림 모델인 Stable Diffusion에 LoRA라는 작은 부분만 추가로 학습합니다. '
                '둘 다 정상 사진 위에 결함을 그려서 결함 위치를 함께 얻습니다. 자세한 구조는 부록에 있습니다.')
gens = [('cTransGAN', '불량 K장만으로 처음부터 학습', COPPER,
         ['Transformer로 만든 작은 GAN (Wang et al. 2023을 K장에 맞게 줄임)',
          '결함 주변을 256×256으로 잘라, 그 안의 결함만 그린다',
          '진짜·가짜 판정만으로는 잡음만 나와, 결함마다 잡음을 고정하고 원본을 다시 그리게 했다'],
         FIG / 's07_ctransgan_K1_review.png', (3, 940, 757, 1049), '실제 | 바탕 정상 사진 | 합성 | 결함 위치'),
        ('SD+LoRA', '미리 학습된 그림 모델에 작은 부분만 추가 학습', COPPER,
         ['Stable Diffusion 1.5 인페인팅(빈칸을 채워 그리는 모델)은 질감 · 조명을 이미 안다',
          'K장으로는 LoRA(전체의 1% 미만인 작은 행렬)만 학습 → 원본을 그대로 외우는 것을 줄인다',
          '결함 종류는 문장으로 알려 주고, 정상 사진의 새 자리에 결함을 채워 그린다'],
         FIG / 's08_sd_lora_v3_pcb1_sheet.jpg', (0, 599, 790, 858), '실제 결함 | 바탕 정상 사진 | 합성')]
for i, (name, sub, c, bullets, img, crop, cap) in enumerate(gens):
    x = 0.6 + i * 6.15
    s.box(x, 1.6, 5.98, 0.85, [Para(name, 18, c, True, after=1), Para(sub, 13, INK, after=0)],
          align='l', anchor='m', margin=(0.2, 0.05), **AUG)
    s.text(x + 0.05, 2.6, 5.9, 1.75, [Para(b, 13, bullet=True, after=5) for b in bullets])
    s.text(x, 4.38, 5.98, 0.32, cap + ' (K = 1)', size=11, bold=True)
    s.image(img, x, 4.72, w=5.98 if i == 0 else 4.6, crop=crop)
s.box(0.6, 6.45, 12.13, 0.55, '공통: PCB · 결함 종류마다 40장을 그린다. 휜 핀은 원래 곧은 핀이 남아 겹쳐 보여서, 남은 핀을 찾아 지웠다 (부록 18)',
      size=13, fill=GRAY_T, align='l', margin=(0.2, 0.03))

s = d.new('위치 모델: 결함 지도를 직접 학습한다', '탐지 모델', 2, title_size=26,
          notes='탐지에 쓴 위치 모델입니다. 앞의 모델과 같은 고정된 WRN50에서 크기가 다른 특징 세 개를 뽑고, 그 위에 작은 디코더만 학습해 결함 지도를 냅니다. '
                '사진의 칸마다 정답이 있어서 작은 결함이 묻히지 않고, 사진 전체의 밝기로는 결함 지도를 맞힐 수 없습니다. '
                '사진의 점수는 결함 지도에서 가장 높은 10칸의 평균입니다.')
s.box(0.6, 2.0, 1.9, 1.3, '사진\n512×512', size=12, **DATA)
s.line(2.52, 2.65, 2.88, 2.65, color=INK)
s.box(2.9, 1.65, 2.9, 2.0, '', fill=WHITE, line=GREEN)
s.text(3.0, 1.7, 2.7, 0.35, 'WRN50 (고정)', size=13, bold=True, color=GREEN, align='c')
for i, (lab, sz) in enumerate([('128×128칸', 1.0), ('64×64칸', 0.9), ('32×32칸', 0.8)]):
    s.box(3.1, 2.1 + i * 0.5, 2.5 * sz, 0.4, lab, size=11, fill=TINT, line=None, radius=0.05, margin=(0.04, 0.0))
s.line(5.82, 2.65, 6.18, 2.65, color=INK)
s.box(6.2, 1.65, 3.4, 2.0, [Para('작은 디코더 (학습)', 13, GREEN, True, align='c', after=4),
                            Para('크기가 다른 세 특징을', 12, align='c', after=1),
                            Para('합치며 크게 키운다', 12, align='c', after=0)], **ADM)
s.line(9.62, 2.65, 9.98, 2.65, color=INK)
s.box(10.0, 2.0, 2.7, 1.3, '결함 지도\n128×128', size=14, bold=True, fill=COPPER_T, line=COPPER)
for i, (h, body) in enumerate([('정답', '결함 위치. VisA는 픽셀 단위, SMT는 상자 (상자 안 전체를 결함으로 본다)'),
                               ('사진의 점수', '결함 지도에서 가장 높은 10칸의 평균 → 다른 모델과 같은 기준으로 정상 / 불량 판정'),
                               ('학습', '한 번에 정상 4장 + 불량 4장, 2000번 갱신. 합성을 쓰면 불량 4장 중 2장이 합성'),
                               ('왜 이 구조인가', '칸마다 정답이 있어 작은 결함이 묻히지 않는다. 사진 전체의 밝기로는 결함 지도를 맞힐 수 없다')]):
    x, y = 0.6 + (i % 2) * 6.15, 4.0 + (i // 2) * 1.45
    s.box(x, y, 5.95, 1.25, [Para(h, 14, COPPER if i == 3 else GREEN, True, after=3), Para(body, 13)],
          align='l', anchor='t', margin=(0.2, 0.1), fill=COPPER_T if i == 3 else GRAY_T)

s = d.new('RQ1 결과: 위치로 배우면 결함을 보고, 실제 3장부터는 PatchCore보다 낫다', 'RQ1 · 결과', 2, title_size=24,
          notes='첫 번째 질문의 답입니다. 같은 불량 1장이라도 사진 단위 DevNet은 결함 바꾼 시험에서 불량의 87%를 놓치는데, 위치 모델은 24%만 놓칩니다. '
                '결함을 지운 사진을 불량이라 하는 실수도 44%에서 14%로 줄고, 정상에 붙인 결함은 74%를 잡습니다. 이제 결함을 보는 것입니다. '
                '다만 실제 불량 1장으로는 정상만 쓰는 PatchCore와 비슷합니다. 실제 불량이 3장이면 PatchCore보다 확실히 덜 놓칩니다. '
                '불량 라벨은 결함 위치까지 써야 하고, 몇 장은 모아야 값을 합니다.')
header(s, 0.6, 1.6, 5.9, 1, '결함을 보는가 (실제 불량 1장)', GREEN)
for i, (crop, lab) in enumerate(((CAM_ORIG, '원본 (흰 선 = 결함)'), (CAM_DEVNET, '사진 단위 DevNet'), (CAM_SEG, '위치 모델'))):
    x = 0.6 + i * 2.0
    s.text(x, 2.1, 1.75, 0.3, lab, size=11, bold=True, color=COPPER if i == 1 else (GREEN if i == 2 else INK), align='c')
    s.image(GRADCAM, x + 0.1, 2.4, w=1.55, crop=crop)
rows = [['', '사진 단위 DevNet', '위치 모델'],
        ['놓친 불량 (결함 바꾼 시험) ↓', '87%', '**24%**'],
        ['모델이 본 곳 중 결함 자리 ↑', '4%', '**75%**'],
        ['결함 지운 사진을 불량이라 함 ↓', '44%', '**14%**'],
        ['정상에 붙인 결함을 잡음 ↑', '16%', '**74%**']]
s.table(0.6, 4.05, [3.05, 1.45, 1.4], rows, size=12, row_h=0.42)
s.text(0.6, 6.25, 5.9, 0.7, '↑ 클수록 좋음, ↓ 작을수록 좋음. 위치 모델에 밝기·색 바꾸기를 더하면 놓친 불량 24.4% → 21.7%', size=11, color=MUTED)
header(s, 6.85, 1.6, 5.88, 2, '정상만 쓰는 PatchCore와 비교', GREEN)
s.text(6.85, 2.08, 5.88, 0.35, '위치 모델 + 밝기·색 바꾸기, 실제 불량만. 놓친 불량 (결함 바꾼 시험)', size=11, color=MUTED)
ch = s.bar_chart(6.85, 2.42, 5.88, 2.65, ['실제 1장', '2장', '3장', '4장'], [('위치 모델', [0.217, 0.133, 0.098, 0.079])],
                 [GREEN], vmax=0.25, major=0.05, legend=False, fmt='0.0%', point_colors=[GRAY_L, '9BD3B4', GREEN, GREEN],
                 ref=(0.188, None), gap=70, lay=(0.11, 0.06, 0.86, 0.78))
pct_axis(ch)
labels_inside(ch, white=())
s.text(9.4, 2.4, 3.3, 0.3, '점선 = PatchCore 18.8%', size=11, bold=True, color=COPPER, align='r')
s.box(6.85, 5.2, 5.88, 1.75, [Para('실제 1장: PatchCore와 비슷하다 (차이 없음)', 13, after=3),
                             Para('실제 2장: 덜 놓치지만 우연일 수 있다', 13, after=3),
                             Para('실제 3장부터: 확실히 덜 놓친다 (−9%p, 95% 범위 −16 ~ −1%p)', 13, GREEN, True, after=3),
                             Para('15번 비교 = PCB 3개 × 반복 5번', 10, MUTED)],
      align='l', anchor='t', margin=(0.18, 0.1), fill=TINT)

# ======================================================================= 3. RQ2
d.section('3. RQ2 합성은 얼마나')
s = d.new('RQ2. 합성 불량은 얼마나 넣어야 하고, 실제 몇 장의 값어치일까?', 'RQ2', 3, title_size=26,
          notes='위치 모델에 cTransGAN 합성 40장을 더하면 놓친 불량이 21.7%에서 18%로 줄었습니다. 15번 비교 중 12번 좋아져 우연은 아닙니다. '
                '단순히 실제 결함을 오려 붙인 합성은 효과가 없었으니, 생성 모델이 새 모양을 그린 덕입니다. '
                '그렇다면 합성은 몇 장이 좋고, 실제 불량 몇 장의 값어치일까요? 생성 모델 2개, 실제 1·2·4장, 합성 0장에서 40장까지 바꿔 돌렸습니다.')
s.box(0.6, 1.6, 12.13, 1.2, [Para('출발점 (RQ1과 같은 실험, 실제 불량 1장 + 종류마다 합성 40장)', 13, GREEN, True, after=4),
                             Para('cTransGAN 21.7% → **18.0%** (15번 중 12번 좋아짐)   ·   SD+LoRA 21.7% → 20.5% (15번 중 9번, 우연일 수 있음)   ·   '
                                  '실제 결함을 오려 붙이기만 한 합성 21.7% → 22.4% (효과 없음)', 13, after=0)],
      align='l', anchor='m', margin=(0.25, 0.08), fill=TINT)
for i, (n, h, body, c) in enumerate([('N', '합성 장수', '종류마다 1장에서 40장까지 늘리면 계속 좋아질까, 어디서 멈출까?', COPPER),
                                     ('K', '실제 몇 장 값어치', '합성 40장을 더한 효과는 실제 불량을 몇 장 모은 것과 같을까?', GREEN),
                                     ('G', '생성 모델', '처음부터 학습하는 cTransGAN과 미리 학습된 SD+LoRA 중 어느 쪽이 나을까?', COPPER)]):
    x = 0.6 + i * 4.15
    s.box(x, 3.0, 3.83, 2.15, '', fill=GRAY_T, radius=0.12)
    s.oval(x + 0.25, 3.18, 0.7, c, n, size=22)
    s.text(x + 1.1, 3.28, 2.6, 0.5, h, size=18, bold=True, color=c)
    s.text(x + 0.25, 4.0, 3.35, 1.1, body, size=14)
s.box(0.6, 5.35, 12.13, 1.6, [Para('실험', 14, GREEN, True, after=4),
                              Para('VisA, 생성 모델 2개 × 실제 불량 1 · 2 · 4장 × 합성 0 · 1 · 5 · 10 · 20 · 40장, 반복 5번 (PCB 3개씩). '
                                   '실제만은 1장부터 10장까지 더 돌려 "실제 몇 장" 곡선을 만들었다', 13, bullet=True, after=4),
                              Para('합성 N장 = 같은 40장 중 앞 N장. 학습 때 불량 4장 중 2장은 늘 합성이라, N을 늘리면 합성이 차지하는 비중은 같고 '
                                   '서로 다른 합성의 가짓수만 늘어난다', 13, bullet=True, after=0)],
      align='l', anchor='t', margin=(0.25, 0.12), fill=WHITE, line=GRAY_L)

s = d.new('RQ2 결과: 실제 장수가 가장 중요하고, 합성은 20장 안팎이면 충분하다', 'RQ2 · 결과', 3, title_size=26,
          notes='x축은 종류마다 더한 합성 장수, 선은 실제 불량 장수입니다. 실제 장수를 늘리는 효과가 가장 큽니다. '
                '실제 2장이 실제 1장에 합성 40장을 더한 것보다 낫습니다. '
                'cTransGAN은 합성을 20장까지 늘리면 좋아지다가 거의 멈춥니다. SD+LoRA는 실제 1장에서는 효과가 없고, 2장부터는 cTransGAN과 비슷합니다.')
s.text(0.6, 1.55, 12.13, 0.4, f'놓친 불량 ({FAIR}). x = 종류마다 더한 합성 장수, 선 = 실제 불량 장수, 점선 = PatchCore 18.8%. 반복 5번 평균',
       size=12, color=MUTED)
N = ['0', '1', '5', '10', '20', '40']
sd = [[.217, .196, .239, .229, .217, .205], [.133, .137, .122, .135, .112, .115], [.079, .077, .070, .064, .059, .067]]
ctg = [[.217, .202, .200, .191, .177, .180], [.133, .131, .119, .112, .111, .107], [.079, .083, .070, .071, .067, .063]]
for i, (name, data) in enumerate((('SD+LoRA', sd), ('cTransGAN', ctg))):
    x = 0.6 + i * 6.15
    s.text(x, 1.95, 5.9, 0.4, name, size=16, bold=True, color=COPPER)
    ch = s.line_chart(x, 2.35, 5.98, 3.2, N, [(f'실제 {k}장', v) for k, v in zip((1, 2, 4), data)], K_COLORS,
                      0, 0.25, 0.05, ref=(0.188, None), lay=(0.11, 0.04, 0.86, 0.7))
    pct_axis(ch)
s.box(0.6, 5.7, 5.98, 0.8, '실제 장수가 가장 중요하다: 실제 1 → 2 → 4장에서 21.7% → 13.3% → 7.9%. 실제 2장이 실제 1장 + 합성 40장(18.0%)보다 낫다',
      size=12, fill=TINT, align='l', margin=(0.15, 0.05))
s.box(6.75, 5.7, 5.98, 0.8, 'cTransGAN은 합성 20장까지 좋아지고 거의 멈춘다 (실제 2장: 13.3% → 11.1%, 40장 10.7%). '
      'SD+LoRA는 실제 1장에서는 효과가 없다', size=12, fill=COPPER_T, align='l', margin=(0.15, 0.05))
s.text(0.6, 6.6, 12.13, 0.4, '답: 합성은 종류마다 20장 안팎이면 충분하다. 실제 1장이면 cTransGAN, 2장부터는 둘이 비슷하다',
       size=15, bold=True, color=GREEN)

s = d.new('RQ2 결과: 합성 40장 = 실제 1장 → 1.4장, 4장 → 6.6장의 효과', 'RQ2 · 실제 몇 장', 3, title_size=26,
          notes='합성의 값어치를 실제 장수로 바꿔 봤습니다. 오른쪽은 실제 불량만으로 1장부터 10장까지 학습한 곡선입니다. '
                '예를 들어 실제 4장에 cTransGAN 40장을 더하면 6.3%를 놓치는데, 실제만으로는 약 6.6장일 때 나오는 성적입니다. '
                '이렇게 읽으면 cTransGAN 40장은 실제 1장을 1.4장, 2장을 2.7장, 4장을 6.6장으로 만듭니다. 실제가 많을수록 합성의 값어치도 커집니다.')
s.text(0.6, 1.55, 12.13, 0.4, '읽는 법: 합성을 더한 결과와 놓친 불량이 같아지는 자리를 실제만 곡선(오른쪽)에서 찾는다. VisA, 결함 바꾼 시험, 반복 5번',
       size=12, color=MUTED)
header(s, 0.6, 2.0, 6.0, 1, '합성 40장을 더하면 실제 몇 장 효과', GREEN, size=14)
ch = s.bar_chart(0.6, 2.45, 6.0, 2.95, ['실제 1장', '실제 2장', '실제 4장'],
                 [('실제만', [1, 2, 4]), ('+ SD+LoRA 40장', [1.1, 2.5, 6.1]), ('+ cTransGAN 40장', [1.4, 2.7, 6.6])],
                 TRAIN_COLORS, vmax=8, major=2, fmt='0.0', lay=(0.09, 0.08, 0.89, 0.66), gap=70)
ch.value_axis.tick_labels.number_format = '0'
s.text(0.6, 5.45, 6.0, 0.45, '95% 범위: cTransGAN 1.2–2.0 · 2.3–3.8 · 4.5–7.8장, SD+LoRA 1장 미만–1.4 · 2.0–3.2 · 4.0–7.5장',
       size=11, color=MUTED)
header(s, 6.85, 2.0, 5.88, 2, '실제만 곡선에서 읽기', GREEN, size=14)
REAL_K = ['1', '2', '3', '4', '5', '6', '8', '10']
REAL = [.217, .133, .098, .079, .069, .069, .051, .039]
LAY = (0.12, 0.06, 0.85, 0.7)
ch = s.line_chart(6.85, 2.45, 5.88, 2.95, REAL_K, [('실제만', REAL)], [GRAY_L], 0, 0.25, 0.05, lay=LAY, legend=False,
                  ref=(0.180, '실제 1장 + cTransGAN 40장 18.0% → 1.4장'))
pct_axis(ch)
s._ref_line(6.85, 2.45, 5.88, 2.95, LAY, 0.25, 0.063, None)       # 글자는 곡선과 겹치지 않게 선 아래 왼쪽에
_yy = 2.45 + 2.95 * (LAY[1] + LAY[3] * (1 - 0.063 / 0.25))
s.text(6.85 + 5.88 * LAY[0] + 0.05, _yy + 0.03, 3.6, 0.28, '실제 4장 + cTransGAN 40장 6.3% → 6.6장', size=11, color=COPPER, bold=True)
s.text(6.85, 5.45, 5.88, 0.45, 'x = 실제만 쓴 장수, y = 놓친 불량. 점선이 곡선과 만나는 자리가 실제 몇 장 효과', size=11, color=MUTED)
s.box(0.6, 6.0, 6.0, 0.9, [Para('실제가 많을수록 합성의 값어치도 크다', 13, GREEN, True, after=2),
                           Para('cTransGAN 40장: 실제 1장 → 1.4장, 2장 → 2.7장, 4장 → 6.6장', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=TINT)
s.box(6.85, 6.0, 5.88, 0.9, [Para('SD+LoRA는 실제 1장에서는 효과가 없다', 13, COPPER, True, after=2),
                             Para('1.1장 (95% 범위 1장 미만–1.4). 실제 2장부터는 cTransGAN과 비슷하다', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=COPPER_T)

# ======================================================================= 4. RQ3
d.section('4. RQ3 실제 라인')
s = d.new('RQ3. 실제 생산 라인(AIHub SMT)에서도 위치 모델과 합성이 통할까?', 'RQ3 · 데이터', 4, title_size=24,
          notes='마지막 질문은 실제 생산 라인 데이터입니다. AIHub SMT 데이터에는 기판에 납을 바른 직후의 사전 공정과 부품을 납땜한 뒤의 납땜 공정이 있고, '
                '결함은 6종과 12종입니다. 결함 위치는 상자로만 표시돼 있습니다. '
                '여기에도 같은 문제가 있었습니다. 사진 파일 이름의 촬영 코드만으로도 정상과 불량이 갈렸습니다. '
                '흐름은 VisA와 같고, 결함 위치가 상자뿐이라는 점과 시험 방식만 다릅니다.')
TW = [1.6, 1.9, 1.9, 3.5, 3.23]
rows = [['공정', '정상 예시', '불량 예시', '결함 종류', '장수'],
        ['사전 공정\n(납을 바른 직후)', '', '', '미납 · 납부족 · 납쇼트\n납볼 · 위치 밀림 · 모양 불량 (6종)',
         '시험: 정상 1,000 · 불량 300장\n학습 불량: 종류마다 1 · 4 · 16장'],
        ['납땜 공정\n(부품 납땜 뒤)', '', '', '미납 · 냉납 · 밀림 · 쇼트 · 오삽\n미삽(부품 빠짐) · 역삽 · 뒤집힘\n일어섬 · 납볼 · 핀홀 · 납고드름 (12종)',
         '시험: 정상 1,000 · 불량 540장\n학습 불량: 종류마다 1 · 4 · 16장']]
RH, HH = 1.45, 0.45
s.table(0.6, 1.6, TW, rows, size=12, row_h=RH, header_h=HH, align=['c', 'c', 'c', 'l', 'l'])
for i in range(2):                                   # AIHub 사진은 git에 올릴 수 없어 자리만 둔다 (직접 넣기)
    for j in range(2):
        x, y = 0.6 + TW[0] + j * TW[1], 1.6 + HH + i * RH
        s.box(x + 0.15, y + 0.1, TW[1] - 0.3, RH - 0.2, '사진 자리\n(512×512)', fill=None, line=GRAY_L, dash=True,
              size=11, color=MUTED, name='SMT photo placeholder')
s.box(0.6, 5.1, 12.13, 0.62, '같은 문제가 있다: 사진 파일 이름에 붙은 촬영 코드만으로도 정상과 불량이 갈린다 (AUROC 0.92). '
      '같은 부품 자리를 찍은 사진 62쌍에서도 불량 쪽이 더 밝다', size=13, fill=COPPER_T, align='l', margin=(0.2, 0.03))
for i, (h, body) in enumerate([('결함 위치', '상자뿐이다. 위치 모델은 상자 안 전체를 결함으로 보고 배운다'),
                               ('반복', '실제 1 · 4 · 16장 × 3번 (실제만은 1–32장 × 5번)'),
                               ('시험', '원래 시험셋, 정상 100장 중 1장 기준. 결함 바꾼 시험은 만들지 않았다')]):
    x = 0.6 + i * 4.1
    s.box(x, 5.88, 3.93, 1.1, [Para(f'VisA와 다른 점: {h}', 13, GREEN, True, after=2), Para(body, 12, after=0)],
          align='l', anchor='t', margin=(0.16, 0.08), fill=TINT)

s = d.new('RQ3 결과: 실제 라인에서도 위치 모델은 결함을 보고, 합성은 사전 공정에서 통했다', 'RQ3 · 결과', 4, title_size=24,
          notes='정상 사진이 다양한 실제 라인에서는 정상만 쓰는 PatchCore가 사전 공정 불량의 91%를 놓칩니다. 위치 모델은 불량 1장만으로 20%만 놓칩니다. '
                '결함 바꾼 시험은 없지만, 결함을 지운 사진을 불량이라 한 비율이 사전 14%, 납땜 6%로 낮아서 배경이 아니라 결함을 보고 맞힌 것으로 봅니다. '
                '합성을 더하면 사전 공정에서는 모든 장수에서 좋아졌고, 어려운 납땜 공정은 실제가 적을 때만 조금 좋아지고 16장에서는 나빠졌습니다.')
s.text(0.6, 1.55, 7.6, 0.4, 'SMT 위치 모델의 놓친 불량 (원래 시험, 정상 100장 중 1장 기준, 반복 3번)', size=12, color=MUTED)
rows, fills = [['공정', '실제', '실제만', '+ SD+LoRA', '+ cTransGAN']], [[None] * 5]
for cat, ko in (('pre', '사전'), ('solder', '납땜')):
    for k in (1, 4, 16):
        a, b, c = smt_v(cat, k, 'none'), smt_v(cat, k, 'sd_lora_fix:40:0.5'), smt_v(cat, k, 'ctransgan:40:0.5')
        rows.append([ko if k == 1 else '', f'{k}장', pct(a), pct(b), pct(c)])
        fills.append([None, None, None, TINT if b < a else COPPER_T, TINT if c < a else COPPER_T])
s.table(0.6, 1.95, [1.2, 1.0, 1.7, 1.85, 1.85], rows, size=15, row_h=0.55, fills=fills)
s.text(0.6, 5.85, 7.6, 0.4, '초록 = 실제만보다 좋아짐, 주황 = 나빠짐', size=12, color=MUTED)
s.box(0.6, 6.3, 7.6, 0.68, '답: 실제 라인에서도 위치 모델이 통한다. 합성은 사전 공정에서는 늘 도움이 되고, 어려운 납땜 공정은 실제가 적을 때만 돕는다',
      size=13, bold=True, fill=TINT, align='l', margin=(0.2, 0.04))
pc1 = smt_v('pre', 1, 'none')
stat_card(s, 8.5, 1.6, 4.23, 1.65, f'91% → {pc1 * 100:.0f}%',
          f'사전 공정: 정상만 쓰는 PatchCore는 불량의 91%를 놓친다. 위치 모델은 불량 1장만으로 {pct(pc1)}', GREEN)
stat_card(s, 8.5, 3.4, 4.23, 1.65, '결함을 보고 맞힌다',
          '결함을 지운 사진을 불량이라 한 비율: 사전 14% · 납땜 6% (VisA 위치 모델 14%와 비슷). 배경을 보는 모델이면 이 비율이 높다', GREEN)
stat_card(s, 8.5, 5.2, 4.23, 1.78, '납땜 16장: 나빠짐',
          '불량의 절반을 늘 합성으로 채워, 실제가 많아도 합성이 같은 비중이었다 → 다음 장에서 합성 비율을 줄여 봤다', COPPER)

# ---- RQ3 실제 몇 장 + 합성 비율
R25 = {(cat, g): {k: smt_v(cat, k, f'{m}:40:0.25') for k in (1, 4, 16)} for cat in ('pre', 'solder') for g, m in GEN.items()}
R50 = {(cat, g): {k: smt_v(cat, k, f'{m}:40:0.5') for k in (1, 4, 16)} for cat in ('pre', 'solder') for g, m in GEN.items()}
RATIO_DONE = all(v is not None for d_ in R25.values() for v in d_.values())
RATIO_NOTE = ('합성 비율을 줄이면: 실험 중 (run_smt_ratio.sh)' if not RATIO_DONE else
              [Para('합성 비율을 1/4로 줄이면 (납땜 16장)', 13, COPPER, True, after=3),
               Para('SD+LoRA 49.6% → ' + pct(R25[('solder', 'SD+LoRA')][16]) + ': 실제만(40.9%)과 비슷 → 비율 탓', 12, after=2),
               Para('cTransGAN 72.8% → ' + pct(R25[('solder', 'cTransGAN')][16]) + ': 여전히 나쁨 → 과장된 합성 탓 (부록 6)', 12, after=2),
               Para('실제 1 · 4장에서는 1/4로 줄여도 이득 유지', 12)])
s = d.new('RQ3 결과: SMT에서도 실제가 적을 때 합성 40장은 실제 1.2–2.6배 효과', 'RQ3 · 실제 몇 장', 4, title_size=24,
          notes='RQ2처럼 SMT에서도 합성의 값어치를 실제 장수로 바꿔 봤습니다. 실제만은 1장부터 32장까지 다섯 번 반복해 곡선을 만들었습니다. '
                '실제가 1장, 4장일 때는 VisA와 같은 흐름입니다. 실제 16장에서는 공정마다 달랐습니다. 사전 공정은 실제 32장보다도 덜 놓쳤고, 납땜 공정은 나빠졌습니다. '
                '납땜에서 나빠진 이유를 보려고 합성 비율을 4장 중 2장에서 1장으로 줄였더니, SD+LoRA는 실제만과 비슷해져 비율 탓이 컸고, '
                'cTransGAN은 여전히 나빠 결함을 과장한 합성 자체가 문제였습니다.')
s.text(0.6, 1.55, 12.13, 0.4, '읽는 법은 RQ2와 같다. 실제만 곡선은 1 · 2 · 4 · 8 · 16 · 32장 × 반복 5번. ' + ORIG, size=12, color=MUTED)
header(s, 0.6, 2.0, 6.0, 1, '합성 40장을 더하면 실제 몇 장 효과', COPPER, size=14)
smt_eq = [('사전 공정', '1장', '2.4장', '2.6장', TINT), ('', '4장', '6.0장', '6.3장', TINT),
          ('', '16장', '32장보다 좋음', '32장보다 좋음', TINT),
          ('납땜 공정', '1장', '1.2장', '1.5장', TINT), ('', '4장', '5.9장', '5.4장', TINT), ('', '16장', '나빠짐', '크게 나빠짐', COPPER_T)]
rows = [['공정', '실제', '+ SD+LoRA 40장', '+ cTransGAN 40장']] + [list(r[:4]) for r in smt_eq]
fills = [[None] * 4] + [[None, None, r[4], r[4]] for r in smt_eq]
s.table(0.6, 2.45, [1.3, 0.7, 2.0, 2.0], rows, size=14, row_h=0.44, fills=fills)
s.text(0.6, 5.6, 6.0, 0.4, '16장: 사전 3.4% → 1.6% · 1.9% (실제 32장은 2.9%), 납땜 40.9% → 49.6% · 72.8%', size=11, color=MUTED)
header(s, 6.85, 2.0, 5.88, 2, '실제 16장에서 합성 비율을 줄이면? (반복 3번)', GREEN, size=14)
rows, fills = [['실제 16장', '실제만', '합성 2/4', '합성 1/4']], [[None] * 4]
for cat, ko in (('pre', '사전'), ('solder', '납땜')):
    for g in GEN:
        a, b, c = smt_v(cat, 16, 'none'), R50[(cat, g)][16], R25[(cat, g)][16]
        rows.append([f'{ko} · {g}', pct(a), pct(b), pct(c)])
        fills.append([None, None, TINT if b < a else COPPER_T, None if c is None else (TINT if c < a else COPPER_T)])
s.table(6.85, 2.45, [2.2, 1.2, 1.24, 1.24], rows, size=13, row_h=0.44, fills=fills)
s.text(6.85, 4.75, 5.88, 0.6, '학습 때 불량 4장 중 합성이 몇 장인가. 지금까지는 2장(2/4), 줄이면 1장(1/4)', size=11, color=MUTED)
s.box(0.6, 6.05, 6.0, 0.9, [Para('실제가 적을 때는 VisA와 같은 흐름', 13, GREEN, True, after=2),
                           Para('실제 1장 → 1.2–2.6장, 4장 → 5.4–6.3장 (두 공정 · 두 생성 모델)', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=TINT)
s.box(6.85, 5.35, 5.88, 1.6, RATIO_NOTE, size=12, fill=COPPER_T, align='l', anchor='m', margin=(0.18, 0.06))

s = d.new('왜 합성은 실제 불량을 대신하지 못할까', 'RQ2 · RQ3 원인', 4, title_size=26,
          notes='합성은 실제가 적을 때 도움이 되지만 실제 불량을 대신하지는 못했습니다. 이유는 세 가지입니다. '
                '첫째, 합성은 같은 실제 1장을 보고 만든 변형이라 새로운 정보가 적습니다. 160장을 만들어도 실제 시험 결함의 75%는 비슷한 조각이 합성에 없습니다. '
                '둘째, 합성은 결함을 실제보다 훨씬 눈에 띄게 그립니다. 납쇼트는 실제 결함의 4%만 정상과 확실히 달라 보이는데 합성은 66%였고, 이 종류는 합성을 더하자 더 놓쳤습니다. '
                '셋째, 휜 핀이나 부품 빠짐처럼 생성 모델이 잘 못 그리는 종류가 있습니다.')
cols, CW = [0.6, 4.75, 8.9], 3.83
header(s, cols[0], 1.6, CW, 1, '새로운 정보가 적다', COPPER)
header(s, cols[1], 1.6, CW, 2, '결함을 너무 눈에 띄게 그린다', COPPER)
header(s, cols[2], 1.6, CW, 3, '못 그리는 종류가 있다', COPPER)
s.text(cols[0], 2.08, CW, 0.62, '시험 결함 조각 중, 합성 세트에 비슷한 조각이 있는 비율 (SMT 사전 공정, SD+LoRA)', size=11, color=MUTED)
ch = s.bar_chart(cols[0], 2.7, CW, 2.3, ['20장', '40장', '80장', '160장'], [('실제 1장', [0.085, 0.124, 0.176, 0.248]),
                                                                          ('실제 4장', [0.141, 0.212, 0.288, 0.384])],
                 [COPPER, GREEN], vmax=0.5, major=0.25, fmt='0%', lay=(0.13, 0.08, 0.84, 0.6), gap=60)
pct_axis(ch)
ch.plots[0].has_data_labels = False
s.text(cols[0], 5.1, CW, 1.0, '합성은 같은 1장을 보고 만든 변형이다. 160장을 만들어도 시험 결함의 75%는 비슷한 조각이 없다', size=13)
s.text(cols[1], 2.08, CW, 0.62, '결함 상자 안에서 정상과 확실히 달라 보이는 칸의 비율 (SMT 사전 공정, 납쇼트)', size=11, color=MUTED)
ch = s.bar_chart(cols[1], 2.7, CW, 2.3, ['실제 결함', '합성 (cTransGAN)'], [('비율', [0.04, 0.66])], [GRAY_L], vmax=0.8, major=0.4,
                 fmt='0%', legend=False, point_colors=[GREEN, COPPER], lay=(0.13, 0.08, 0.84, 0.74), gap=90)
pct_axis(ch)
s.text(cols[1], 5.1, CW, 1.0, '실제 결함은 미묘한데 합성은 과장되게 그린다 → 이 종류는 합성을 더하자 놓친 불량이 14%p 늘었다', size=13)
for i, (h, body) in enumerate([('휜 핀 (VisA)', '원래 곧은 핀이 남는 문제. 지운 뒤에도 합성을 더하면 더 놓친다 (14.8% → 21–23%)'),
                               ('미삽 (SMT, 부품 빠짐)', '빈 자리에 부품을 다시 그려 넣는 경우가 많았다. 놓친 비율 80% 그대로')]):
    s.box(cols[2], 2.15 + i * 1.5, CW, 1.35, [Para(h, 13, COPPER, True, after=2), Para(body, 12)],
          align='l', anchor='t', margin=(0.16, 0.08), fill=GRAY_T)
s.text(cols[2], 5.2, CW, 0.9, '"없음"이나 "모양이 살짝 다름"처럼 미묘한 결함일수록 흉내 내기 어렵다', size=13)
s.box(0.6, 6.3, 12.13, 0.68, '결론: 합성은 실제 불량을 대신하지 못하고 보완한다. 실제를 더 모을 수 있으면 그게 먼저다',
      size=14, bold=True, fill=TINT, align='l', margin=(0.2, 0.04))

# ======================================================================= 5. 의의 · 한계
d.section('5. 의의 · 한계')
s = d.new('의의: 적은 불량 사진을 제대로 쓰는 법', '의의', 5,
          notes='정리하겠습니다. 첫째, 같은 불량 1장이라도 결함 위치로 배우면 놓친 불량이 87%에서 24%로 줄고, 실제 3장부터는 정상만 쓰는 PatchCore보다 낫습니다. '
                '둘째, 원래 시험 점수로는 안 보이는 배경 보고 맞히기를 재는 시험을 만들었고, 공개 방법도 여기서 대부분을 놓쳤습니다. '
                '셋째, 합성의 값어치를 실제 장수로 쟀습니다. 실제가 적을 때 합성 40장은 실제를 최대 2.6배로 늘린 효과였습니다. '
                '넷째, 실제 라인에서도 상자 위치만으로 학습한 위치 모델이 결함을 보고 잡았습니다. 오른쪽은 현장에서 따를 순서입니다.')
for i, t in enumerate(['같은 불량 1장이라도 결함 위치로 배우면 놓친 불량이 87% → 24% (사진 단위 DevNet 대비). 실제 3장부터는 정상만 쓰는 PatchCore보다 확실히 낫다',
                       '원래 시험 점수로는 안 보이는 "배경 보고 맞히기"를 재는 시험을 만들었다. 공개 방법 DRA · DevNet도 이 시험에서 불량의 86–91%를 놓친다 (부록 4)',
                       '합성의 값어치를 실제 장수로 쟀다: 실제 1–4장일 때 합성 40장은 실제를 최대 2.6배로 늘린 효과. 대신하지 못하는 이유도 확인했다',
                       '실제 라인(SMT)에서 정상만 쓰는 방식은 91%를 놓치지만, 상자 위치만으로 배운 위치 모델은 결함을 보고 잡는다']):
    y = 1.65 + i * 1.3
    s.oval(0.6, y + 0.08, 0.5, GREEN, str(i + 1), size=14)
    s.text(1.3, y, 5.2, 1.25, t, size=14)
s.box(6.85, 1.6, 5.88, 5.35, '', fill=GRAY_T, radius=0.12)
s.text(7.1, 1.75, 5.4, 0.45, '현장에서 불량 K장을 받았다면', size=18, bold=True, color=COPPER)
for i, t in enumerate(['불량 사진에 결함 위치를 상자로 표시한다', '위치 모델을 밝기·색 바꾸기와 함께 학습한다',
                       '실제가 적으면(종류마다 4장 이하) 합성 20–40장을 더하고, 많아지면 합성 비중을 줄이고 과장된 세트는 뺀다',
                       '배포 전에, 결함을 지운 사진은 정상으로 · 붙인 결함은 불량으로 맞히는지 확인한다']):
    y = 2.35 + i * 0.95
    s.oval(7.1, y + 0.05, 0.42, GREEN, str(i + 1), size=13)
    s.text(7.7, y, 4.85, 0.9, t, size=14)
s.box(7.1, 6.15, 5.4, 0.6, '언제나 먼저: 실제 불량을 1장이라도 더 모은다', size=13, bold=True, fill=WHITE, line=COPPER,
      align='l', margin=(0.15, 0.04))

s = d.new('한계와 다음 실험', '한계', 5,
          notes='한계입니다. SMT는 반복이 세 번뿐이고, 시험용 휜 핀 불량은 17장뿐이라 종류별 결론은 약합니다. '
                'SMT에는 결함 바꾼 시험을 만들지 않아서, SMT 숫자는 실제보다 좋게 나왔을 수 있습니다. '
                '실제 불량 1–2장일 때는 PatchCore와 통계적으로 차이가 없습니다. '
                '학습에 없던 결함 종류는 시험하지 않았고, VisA는 픽셀 단위로 학습해서 상자만으로 충분한지는 SMT에서만 봤습니다. 오른쪽은 이를 메울 다음 실험입니다.')
for i, (h, body) in enumerate([
        ('표본이 작다', 'SMT는 반복 3번, 시험용 휜 핀 불량은 17장 → 종류별 경향은 약한 근거'),
        ('SMT 시험', 'SMT에는 결함 바꾼 시험을 만들지 않았다. 결함 지운 사진 확인은 했지만, 숫자가 실제보다 좋게 나왔을 수 있다'),
        ('실제 1–2장', '실제 불량 1–2장일 때는 PatchCore와 통계적으로 차이가 없다. PatchCore 성적은 PCB마다 6–29%로 크게 다르다'),
        ('처음 보는 결함 · 상자', '학습에 없던 결함 종류는 시험하지 않았다. VisA는 픽셀 단위로 학습해서, 상자만으로 충분한지는 SMT에서만 봤다'),
        ('원래 시험 점수', '정상 · 불량 구분 점수는 최신 방법보다 조금 낮다 (0.971 vs 0.992, 부록 4)')]):
    y = 1.6 + i * 1.07
    s.box(0.6, y, 7.1, 0.95, '', fill=GRAY_T, radius=0.1)
    s.text(0.8, y + 0.08, 2.0, 0.8, h, size=14, bold=True, color=COPPER, anchor='m')
    s.text(2.85, y + 0.08, 4.7, 0.8, body, size=13, anchor='m')
s.box(8.0, 1.6, 4.73, 5.3, '', fill=TINT, radius=0.12)
s.text(8.25, 1.75, 4.3, 0.45, '다음 실험', size=18, bold=True, color=GREEN)
s.text(8.25, 2.35, 4.3, 4.5, [Para(t, 15, bullet=True, after=16) for t in
                              ['SMT에도 결함 바꾼 시험 만들기 → SMT 숫자 다시 확인',
                               'VisA를 상자로 학습 → 상자만으로 충분한지',
                               '결함 종류 하나씩 빼고 학습 → 처음 보는 종류도 잡는지',
                               '과장 정도로 합성 세트를 미리 거르기']])

MAIN_COUNT = len(d.slides)

# ======================================================================= 6. 부록
d.section('6. 부록')
deck_v3_appendix.build(d, FIG, dict(DATA=DATA, AUG=AUG, ADM=ADM, TEN=TEN, LOSS=LOSS, pct=pct, pct_axis=pct_axis))
N_APP = len(d.slides) - MAIN_COUNT


def app(title, title_size=26, **kw):
    return d.new(f'부록 {len(d.slides) - MAIN_COUNT + 1}. {title}', '부록 (질의응답용)', 'A', title_size=title_size, **kw)


# ---- 본편에서 옮긴 것: 기존 모델 구조
s = app('비교한 기존 모델: 같은 WRN50을 쓰고, 불량 라벨을 쓰는 방식만 다르다', title_size=24)
s.box(0.6, 1.65, 2.3, 0.85, '사진\n(512×512)', size=13, **DATA)
s.line(2.92, 2.07, 3.28, 2.07, color=INK, width=1.75)
s.box(3.3, 1.65, 3.1, 0.85, 'WideResNet50\n(ImageNet으로 미리 학습, 고정)', size=13, **ADM)
s.line(6.42, 2.07, 6.78, 2.07, color=INK, width=1.75)
s.box(6.8, 1.65, 3.2, 0.85, '사진을 64×64칸으로 나눠\n칸마다 특징 벡터 (1024차원)', size=13, **TEN)
s.text(10.25, 1.65, 2.48, 0.85, '위치 모델은 같은 WRN50의 세 층을 쓴다', size=12, bold=True, color=GREEN, anchor='m')
s.line(8.4, 2.52, 3.5, 2.93, color=GRAY_L, width=1.5)
s.line(8.4, 2.52, 9.7, 2.93, color=GRAY_L, width=1.5)
s.box(0.6, 2.95, 5.8, 0.48, 'PatchCore: 정상 사진만 학습', size=14, bold=True, fill=GREEN, color=WHITE)
for i, (txt, st) in enumerate([('정상 사진의 칸별 특징을 모아 둔다\n(대표하는 1%만 남김)', ADM),
                               ('시험 사진의 칸마다, 가장 비슷한\n정상 칸과의 거리를 잰다', TEN),
                               ('가장 먼 칸의 거리 = 사진의 이상 점수', TEN)]):
    y = 3.6 + i * 0.92
    s.box(1.1, y, 4.8, 0.72, txt, size=13, **st)
    if i < 2:
        s.line(3.5, y + 0.74, 3.5, y + 0.9, color=INK, width=1.5)
s.box(6.75, 2.95, 5.98, 0.48, '사진 단위 모델: 정상 + 불량 K장, 사진마다 라벨 하나', size=14, bold=True, fill=GREEN, color=WHITE)
s.box(6.75, 3.6, 5.98, 0.72, '칸 특징의 평균과 최댓값을 이어 붙여\n사진 한 장 = 벡터 하나 (2048차원)', size=13, **ADM)
for i, (h, body) in enumerate([('kNN (불량 0장)', ['가장 비슷한 정상', '사진과의 거리']), ('MLP', ['작은 신경망 분류기', '2048 → 256 → 1']),
                               ('DevNet', ['정상 점수는 0 근처로,', '불량 점수는 높게'])]):
    x = 6.75 + i * 2.04
    s.line(9.74, 4.34, x + 0.95, 4.5, color=GRAY_L, width=1.5)
    s.box(x, 4.52, 1.9, 1.32, [Para(h, 13, GREEN, True, align='c', after=3)] + [Para(t, 12, align='c', after=0) for t in body], **TEN)
s.box(0.6, 6.25, 12.13, 0.7, '차이는 불량 라벨을 쓰는 방식뿐: 안 씀(PatchCore) → 사진마다 라벨 하나(MLP · DevNet) → 결함 위치(위치 모델)',
      size=14, fill=GRAY_T, align='l', margin=(0.2, 0.04))

# ---- 본편에서 옮긴 것: cTransGAN 구조
s = app('생성 모델 cTransGAN 구조: 불량 K장만으로 처음부터 학습')
s.box(0.6, 1.7, 2.45, 0.75, 'K장의 결함 부분\n(32×32로 잘라냄)', size=13, **DATA)
s.line(1.825, 2.47, 1.825, 2.78, color=INK)
s.box(0.6, 2.8, 2.45, 0.95, 'Transformer 분류기\n(결함 종류 구분용)', size=13, **AUG)
s.line(1.825, 3.77, 1.825, 4.08, color=INK)
s.box(0.6, 4.1, 2.45, 0.8, '조건 c\n종류별 평균 특징', size=13, **TEN)
s.text(3.45, 2.3, 4.5, 0.4, '생성기 G', size=14, bold=True, color=COPPER)
s.box(3.45, 2.8, 1.95, 0.8, '잡음 z (128차원)\n결함마다 고정, 작은 변화만', size=12, **TEN)
s.line(4.425, 3.62, 4.425, 3.98, color=INK)
for gx, gt in [(3.45, 'z와 c를 합쳐\n8×8 토큰으로'), (5.75, '2배 키우기\n16×16 · Transformer 2층'), (8.05, '2배 키우기\n32×32 · Transformer 1층')]:
    s.box(gx, 4.0, 1.95, 1.0, gt, size=12, **AUG)
for a, b in ((5.4, 5.75), (7.7, 8.05), (10.0, 10.3)):
    s.line(a + 0.02, 4.5, b - 0.02, 4.5, color=INK)
s.line(3.07, 4.5, 3.43, 4.5, color=INK)
s.box(10.3, 4.0, 2.4, 1.0, '토큰마다 8×8 픽셀로\n펼쳐 256×256 사진', size=12, **TEN)
s.line(11.5, 5.02, 11.5, 5.43, color=INK)
s.box(10.3, 5.45, 2.4, 0.95, '정상 사진의 결함 자리에\n끼워 넣음', size=12, **DATA)
s.line(10.28, 5.92, 10.02, 5.92, color=INK)
s.box(8.05, 5.45, 1.95, 0.95, '판별기 D\nTransformer\n+ 조건 c', size=12, **AUG)
s.line(8.03, 5.92, 7.72, 5.92, color=INK)
s.box(5.75, 5.45, 1.95, 0.95, '진짜 / 가짜 판정', size=13, **TEN)
s.box(0.6, 5.2, 4.85, 1.45, [Para('손실', 13, COPPER, True, after=2),
                             Para('진짜·가짜 판정 + 다시 그린 부분의 픽셀 차이 + 경계 + 세밀한 무늬', 12),
                             Para('판정 손실만 쓰면 무지개색 잡음만 나왔다 → 결함마다 잡음 벡터를 고정하고 원본을 다시 그리게 함', 12, MUTED)],
      align='l', anchor='t', margin=(0.15, 0.1), **LOSS)
s.text(0.6, 6.8, 12.13, 0.35, 'Wang et al.(2023) Conditional TransGAN을 K장에 맞게 줄임: 생성기 블록 5 / 4 / 2 → 0 / 2 / 1, 판별기 깊이 7 → 2',
       size=11, color=MUTED)

# ---- 본편에서 옮긴 것: SD+LoRA 구조
s = app('생성 모델 SD+LoRA 구조: 미리 학습된 확산 모델에 작은 부분만 추가 학습', title_size=24)
s.text(0.6, 1.55, 6, 0.35, '학습 (PCB마다 1개, 1000 step)', size=14, bold=True, color=COPPER)
s.box(0.6, 1.95, 2.35, 0.95, 'K장의 결함 주변 512×512\n(정상 사진 10% 섞음)', size=12, **DATA)
s.line(2.97, 2.42, 3.23, 2.42, color=INK)
s.box(3.25, 1.95, 1.5, 0.95, 'VAE 인코더\n(고정)', size=12, **TEN)
s.line(4.77, 2.42, 5.03, 2.42, color=INK)
s.box(5.05, 1.95, 3.0, 0.95, '입력 (64×64, 9채널)\n잡음 4 + 결함 위치 1\n+ 결함을 가린 사진 4', size=12, **TEN)
s.line(6.55, 2.92, 6.55, 3.28, color=INK)
s.box(5.05, 3.3, 3.0, 1.25, [Para('UNet (SD1.5 인페인팅, 고정)', 13, INK, True, align='c', after=2),
                             Para('+ LoRA (rank 16)', 13, COPPER, True, align='c', after=2),
                             Para('어텐션 가중치에 작은 행렬만 더해 학습', 12, INK, align='c')], **AUG)
s.box(0.6, 3.45, 3.3, 0.95, 'CLIP 텍스트 인코더 (고정)\n결함 종류를 문장으로 입력', size=12, **TEN)
s.line(3.92, 3.92, 5.03, 3.92, color=INK)
s.line(8.07, 3.92, 8.43, 3.92, color=INK)
s.box(8.45, 3.5, 1.6, 0.85, '예측한 잡음', size=13, **TEN)
s.line(10.07, 3.92, 10.33, 3.92, color=INK)
s.box(10.35, 3.4, 2.35, 1.05, '손실: 잡음 예측 오차\nLoRA만 갱신', size=13, **LOSS)
s.text(0.6, 4.75, 6, 0.35, '생성 (종류마다 40장)', size=14, bold=True, color=COPPER)
gen = [(0.6, 2.35, '정상 사진 + 새 결함 위치\n(모양 · 위치를 바꿈)', DATA), (3.25, 2.3, '같은 UNet + LoRA\n잡음을 30번 걷어 냄', AUG),
       (5.85, 1.75, 'VAE 디코더\n(고정)', TEN), (7.9, 2.4, '결함 위치 안만\n정상 사진에 붙임', TEN),
       (10.6, 2.1, '2배 만들고, 바탕과\n가장 달라진 40장', TEN)]
for i, (gx, gw, gt, st) in enumerate(gen):
    s.box(gx, 5.15, gw, 0.95, gt, size=12, **st)
    if i < 4:
        s.line(gx + gw + 0.02, 5.62, gen[i + 1][0] - 0.02, 5.62, color=INK)
s.text(0.6, 6.4, 12.13, 0.6, '왜 SD+LoRA인가: 큰 데이터로 미리 학습된 확산 모델은 질감과 조명을 이미 안다. K장으로는 전체 파라미터의 1% 미만인 작은 행렬만 '
       '학습해, 원본을 그대로 외우는 것을 줄인다', size=12, color=MUTED)

# ---- 본편에서 옮긴 것: 합성 시행착오
s = app('합성 시행착오: 잡음, 외우기, 남은 핀')
rows = [['시도', '문제', '고친 방법'],
        ['cTransGAN, 진짜·가짜 판정 손실만', '무지개색 잡음만 나옴', '결함마다 잡음 벡터를 고정하고 원본을 다시 그리게 함'],
        ['SD+LoRA 초기 설정', '흰 얼룩, K장을 그대로 외움, 결함을 흐리게 그림', '정상 사진 10% 섞기, LoRA rank 16, 2배 만들고 바탕과 가장 달라진 장만 고름'],
        ['결함 위치 안만 다시 그림', '원래 곧은 핀이 남아 핀이 겹쳐 보임', '원래 핀 자리까지 다시 그리고, 남은 핀은 색으로 찾아 지움']]
s.table(0.6, 1.65, [1.95, 2.05, 2.75], rows, size=12, row_h=0.6)
s.box(0.6, 4.6, 6.75, 1.0, [Para('원칙', 14, GREEN, True, after=2),
                            Para('합성은 늘 정상 사진 위에 그린다 → 불량 사진의 촬영 환경이 합성에 따라오지 않는다', 13)],
      align='l', anchor='t', margin=(0.2, 0.08), fill=TINT)
s.text(7.65, 1.6, 5.08, 0.35, '남은 핀 지우기 (휜 핀): 실제 | 지우기 전 (핀이 겹침) | 지운 후', size=12, bold=True)
s.image(FIG / 's06_pinfix_before_after.jpg', 7.65, 1.95, w=5.08, crop=(0, 633, 790, 892))
s.text(7.65, 3.85, 5.08, 0.35, 'SD+LoRA (K=1): 실제 결함 | 바탕 정상 사진 | 합성', size=12, bold=True)
s.image(FIG / 's08_sd_lora_v3_pcb1_sheet.jpg', 7.65, 4.2, w=5.08, crop=(0, 599, 790, 858))

d.save(OUT)
args = sys.argv[1:]
preview_dir = ROOT / 'ppt_build' / 'preview_v3'
if '--preview' in args:
    i = args.index('--preview')
    preview_dir = Path(args[i + 1])
    del args[i:i + 2]
d.preview(preview_dir, only=[int(a) for a in args[0].split(',')] if args else None)
print('저장:', OUT, f'(본편 {MAIN_COUNT}장 + 부록 {len(d.slides) - MAIN_COUNT}장)', '미리보기:', preview_dir)
print('합성 비율 실험:', '끝남' if RATIO_DONE else '아직')
if not deck_lib.PREVIEW_FONTS:
    print('주의: Pretendard 글꼴 파일이 없어 미리보기를 NanumGothic으로 그렸다 (PRETENDARD_DIR)')
for w in d.warnings:
    print('경고:', w)
