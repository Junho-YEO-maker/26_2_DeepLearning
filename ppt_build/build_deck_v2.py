"""15분 발표 pptx (v2). 흐름: 문제 → 시도 → 함정 → 방법을 바꿈 → 결과(RQ1) → 합성은 얼마나(RQ2) → 현장 데이터(RQ3) → 의의 · 한계.

    python ppt_build/build_deck_v2.py [--preview 미리보기폴더] [슬라이드번호,...]      (아무 위치에서)
결과: PCB_AD_presentation_v2.pptx (저장소 루트), 미리보기 ppt_build/preview_v2/slide-NN.png
먼저 python ppt_build/make_dataset_figures.py 로 VisA 예시 사진과 장수(ppt_figures/dataset_counts.json)를 만든다.
부록은 deck_v2_appendix.py.

용어 (덱 전체에서 같게 쓴다)
  놓친 불량 비율   = 미검률. "정상의 5%(또는 1%)를 잘못 거르도록 기준을 정했을 때" 놓친 불량의 비율. 낮을수록 좋다
  배경 섞은 시험셋 = 팀 문서의 '공정 시험셋'(결함 지운 사진 · 정상에 결함 붙인 사진을 더한 시험셋)
  사진 단위 모델   = 사진마다 정상/불량 라벨 하나로 배우는 모델(DevNet, KNN+MLP)
  위치 모델        = 결함 위치를 정답으로 배우는 모델(seg)
  촬영 조건 섞기   = 학습 사진의 밝기 · 조명 · 색 · 선명도를 무작위로 바꾸기
  합성 불량        = 생성 모델이 정상 사진 위에 그린 가짜 결함
숫자 출처: build_deck.py와 같다. 추가로 results.csv(원래 시험셋 K 곡선), fair_ad_K1_seed*.csv 짝지은 부트스트랩,
smt/gen_amount_diversity.csv(합성이 실제 결함과 닮은 비율), smt/gen_quality_smt*.csv(결함을 눈에 띄게 그린 정도).

글꼴: 발표 PC에 Pretendard(무료, OFL)가 있어야 한다. 미리보기는 PRETENDARD_DIR(기본 ~/.local/share/fonts/pretendard)로 그린다.
"""
import json
import os
import sys
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
import deck_v2_appendix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'ppt_figures'
OUT = ROOT / 'PCB_AD_presentation_v2.pptx'
VISA = json.loads((FIG / 'dataset_counts.json').read_text(encoding='utf-8'))['visa']

DATA = dict(fill=GRAY_T, line=GRAY_L)
AUG = dict(fill=COPPER_T, line=COPPER)
ADM = dict(fill=TINT, line=GREEN)
TEN = dict(fill=WHITE, line=GRAY_L)
LOSS = dict(fill=WHITE, line=COPPER, dash=True)
K_COLORS = ['9BD3B4', '4FA77F', '1E6B4E', '0B3B2A']           # K 1·2·4·8 (연한 → 진한 초록)
TRAIN_COLORS = ['A3ADA8', 'E3A869', '9C4F12']                  # 실제만 · +SD+LoRA · +cTransGAN
NEG = '1F6FA8'                                                 # 배경 섞은 시험셋의 '정상으로 쳐야 하는 사진' (파랑 테두리)
GRADCAM = FIG / 's12_s14_gradcam_vs_seg_map.jpg'               # PCB1 녹은 납 첫 행: 원본 · DevNet이 본 곳 · · 위치 모델 지도 · ·
CAM_ORIG, CAM_DEVNET, CAM_SEG = (66, 127, 300, 362), (340, 127, 575, 362), (890, 127, 1124, 362)


def pct(v, nd=1):
    return f'{v * 100:.{nd}f}%'


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


def labels_inside(ch, fmt='0.0%'):
    """막대 값 글자를 막대 안쪽 끝에 (점선 기준선과 겹치지 않게). 진한 막대는 흰 글자."""
    for i, ser in enumerate(ch.plots[0].series):
        dl = ser.data_labels
        dl.show_value = True
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.position = XL_LABEL_POSITION.INSIDE_END
        dl.font.size, dl.font.bold = Pt(11), True
        dl.font.color.rgb = deck_lib.rgb(WHITE if i == 2 else INK)


def pipeline(s, y, stages, h=0.95, size=13):
    for i, (t, st) in enumerate(stages):
        x = 0.6 + i * 2.5
        s.box(x, y, 2.1, h, t, size=size, **st)
        if i < len(stages) - 1:
            s.line(x + 2.12, y + h / 2, x + 2.48, y + h / 2, color=INK, width=1.75)


d = Deck()

# ======================================================================= 표지
d.section('표지')
s = d.new('적은 불량 사진을 최대한 활용하는\nPCB 이상탐지', dark=True, title_size=36,
          notes='검사 현장에는 정상 사진이 대부분이고 불량 사진은 종류마다 몇 장씩만 쌓입니다. 이 적은 불량 사진을 어떻게 써야 '
                '결함을 제대로 보는 모델이 되는지를 다룹니다. 세 가지 질문으로 답합니다. 결함 위치로 배우면 나아지는지, 합성 불량은 얼마나 도움이 되는지, '
                '실제 라인 데이터에서도 그런지입니다.')
d.slides[-1].pslide.shapes.title.top = int(2.0 * 914400)
d.slides[-1].pslide.shapes.title.height = int(1.6 * 914400)
s.ops[0] = ('text', 0.6, 2.0, 12.13, 1.6) + s.ops[0][5:]
s.text(0.6, 1.4, 8, 0.45, '2026-2 딥러닝 팀 프로젝트', size=16, color='B8C7BF', bold=True)
s.text(0.6, 3.85, 8.2, 0.9, '현장에 조금씩 쌓이는 불량 사진, 어떻게 써야 결함을 보는 모델이 될까', size=19, color='DCE8E1')
s.text(0.6, 5.9, 8.2, 0.8, ['데이터: VisA PCB(공개) · AIHub SMT(실제 생산 라인)',
                            '생성 모델: cTransGAN · SD+LoRA      탐지 모델: PatchCore · DevNet · 위치 모델'],
       size=14, color='B8C7BF')
for i, (n, t) in enumerate([('RQ1', '위치로 배우기'), ('RQ2', '합성은 얼마나'), ('RQ3', '현장 데이터')]):
    y = 1.6 + i * 1.35
    s.oval(9.5, y, 1.0, [GREEN_M, COPPER, GREEN][i], n, size=17)
    s.text(10.7, y + 0.2, 2.4, 0.6, t, size=21, color=WHITE, bold=True, anchor='m')

# ======================================================================= 1. 문제 정의
d.section('1. 문제 · 시도 · 함정')
s = d.new('문제: 현장에 쌓이는 불량 사진, 표준 이상탐지는 버린다', '문제', 1, title_size=26,
          notes='검사 라인에는 정상 사진이 대부분이고, 불량 사진은 종류마다 몇 장씩 쌓입니다. 결함 종류도 휜 핀, 녹은 납, 긁힘, 부품 누락처럼 제각각입니다. '
                '그런데 표준 이상탐지는 정상 사진만으로 학습하고, 이렇게 쌓인 불량 사진은 쓰지 않습니다. '
                '저희는 적은 불량 사진과 결함 위치까지 학습에 쓰면 결함을 더 잘 잡을 수 있는지 보기로 했습니다.')
cards = [('현장', '검사 라인에는 정상 사진이 대부분이고, 불량 사진은 종류마다 몇 장씩 쌓인다. 결함 종류도 제각각이다 (휜 핀, 녹은 납, 긁힘, 부품 누락 …)',
          GRAY_T, MUTED),
         ('표준 이상탐지', '정상 사진만으로 학습한다 (PatchCore 등). 쌓인 불량 사진은 학습에 쓰지 않는다', TINT, GREEN),
         ('우리의 질문', '적은 불량 사진과 결함 위치까지 학습에 쓰면, 결함을 더 잘 잡을 수 있을까?', COPPER_T, COPPER)]
for i, (h, body, fill, c) in enumerate(cards):
    y = 1.65 + i * 1.78
    s.box(0.6, y, 6.9, 1.55, '', fill=fill, radius=0.12)
    s.oval(0.85, y + 0.2, 0.5, c, str(i + 1), size=15)
    s.text(1.55, y + 0.2, 5.75, 0.45, h, size=18, bold=True, color=INK if i == 0 else c)
    s.text(1.55, y + 0.68, 5.75, 0.82, body, size=14)
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
s.text(7.85, 6.45, 4.88, 0.5, '데이터: 공개 PCB(VisA)로 시작해, 실제 생산 라인(AIHub SMT)에서 확인', size=13, bold=True)

s = d.new('데이터 ① VisA PCB: PCB마다 정상 약 1,000장, 불량 100장', '문제 · 데이터', 1, title_size=26,
          notes='첫 번째 데이터는 공개 데이터셋 VisA의 PCB 세 종류입니다. PCB마다 정상 사진이 약 천 장, 불량 사진이 백 장이고, '
                '결함은 휜 핀, 녹은 납이나 부품, 긁힘, 부품 누락 네 종류입니다. 결함 위치는 픽셀 단위로 표시돼 있습니다. '
                '정상 500장으로 학습하고, 불량은 종류마다 10장을 학습용으로 떼어 둔 뒤 그중 K장만 학습에 씁니다. 나머지 불량은 시험에 씁니다.')
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
                              Para('정상: 학습 500 · 검증 200 · 시험 약 300장.   불량: 종류마다 10장을 학습용으로 떼어 두고 그중 K = 1 · 2 · 4 · 8장만 학습, '
                                   '나머지 60–62장은 시험', 13)],
      align='l', anchor='m', margin=(0.2, 0.05), fill=GRAY_T)

s = d.new('비교한 기존 모델: 같은 특징을 쓰고, 불량 라벨을 쓰는 방식만 다르다', '문제 · 기존 모델', 1, title_size=24,
          notes='처음 비교한 기존 모델들입니다. 모두 ImageNet으로 미리 학습한 같은 WideResNet50을 고정해서, 사진을 작은 칸으로 나눈 칸별 특징을 뽑습니다. '
                '다른 것은 불량 라벨을 쓰는 방식뿐입니다. PatchCore는 정상 사진의 칸별 특징을 모아 두고, 시험 사진의 칸이 정상 칸과 얼마나 다른지로 점수를 냅니다. '
                '불량 사진은 쓰지 않습니다. 사진 단위 모델은 사진 한 장을 벡터 하나로 요약하고, 사진마다 정상인지 불량인지만 알려 주고 학습합니다. '
                'DevNet은 정상 사진의 점수는 0 근처로, 불량 사진의 점수는 높게 밀어내도록 학습합니다.')
s.box(0.6, 1.65, 2.3, 0.85, '사진\n(512×512)', size=13, **DATA)
s.line(2.92, 2.07, 3.28, 2.07, color=INK, width=1.75)
s.box(3.3, 1.65, 3.1, 0.85, 'WideResNet50\n(ImageNet으로 미리 학습, 고정)', size=13, **ADM)
s.line(6.42, 2.07, 6.78, 2.07, color=INK, width=1.75)
s.box(6.8, 1.65, 3.2, 0.85, '사진을 64×64칸으로 나눠\n칸마다 특징 벡터 (1024차원)', size=13, **TEN)
s.text(10.25, 1.65, 2.48, 0.85, '모든 모델이 같은 특징을 쓴다', size=13, bold=True, color=GREEN, anchor='m')
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
for i, (h, body) in enumerate([('kNN (K = 0)', ['가장 비슷한 정상', '사진과의 거리']), ('MLP (K > 0)', ['작은 신경망 분류기', '2048 → 256 → 1']),
                               ('DevNet (K > 0)', ['정상 점수는 0 근처로,', '불량 점수는 높게'])]):
    x = 6.75 + i * 2.04
    s.line(9.74, 4.34, x + 0.95, 4.5, color=GRAY_L, width=1.5)
    s.box(x, 4.52, 1.9, 1.32, [Para(h, 13, GREEN, True, align='c', after=3)] + [Para(t, 12, align='c', after=0) for t in body], **TEN)
s.box(0.6, 6.25, 12.13, 0.7, '차이는 불량 라벨을 쓰는 방식뿐: 안 씀(PatchCore) → 사진마다 라벨 하나(KNN+MLP · DevNet) → 결함 위치(위치 모델, 뒤에서)',
      size=14, fill=GRAY_T, align='l', margin=(0.2, 0.04))

s = d.new('시도: 불량 라벨을 쓰자 원래 시험 점수가 좋아졌다', '시도', 1, title_size=26,
          notes='먼저 불량 라벨을 쓰는 사진 단위 모델을 시험했습니다. 종류마다 쓰는 불량 장수 K를 늘릴수록 놓친 불량이 꾸준히 줄었고, '
                '불량 8장이면 정상만 쓰는 PatchCore보다 덜 놓쳤습니다. 정상과 불량을 가르는 점수도 0.99로 높았습니다. '
                '불량 라벨이 도움이 된다고 보였습니다. 그런데 모델이 정말 결함을 보고 맞힌 걸까요?')
s.text(0.6, 1.55, 7.3, 0.62, '원래 시험셋에서 놓친 불량 비율 (낮을수록 좋음. 정상의 1%만 잘못 거르는 기준). '
       'x = 종류마다 학습에 쓴 실제 불량 장수 K, 점선 = PatchCore(정상만) 15.3%', size=12, color=MUTED)
ch = s.line_chart(0.6, 2.2, 7.3, 3.95, ['1', '2', '4', '8', '10'],
                  [('DevNet', [0.341, 0.234, 0.153, 0.093, 0.080]), ('KNN+MLP', [0.563, 0.410, 0.313, 0.177, 0.140])],
                  [GREEN, GRAY_L], 0, 0.6, 0.1, ref=(0.153, None), lay=(0.1, 0.04, 0.87, 0.76))
pct_axis(ch)
for i, (big, lab, c) in enumerate([('9.3% < 15.3%', '불량 8장을 쓴 DevNet이 정상만 쓰는 PatchCore보다 덜 놓친다', GREEN),
                                   ('0.99', '정상과 불량을 가르는 점수 (AUROC, 1이면 완벽). 불량 8장 DevNet', GREEN)]):
    y = 1.6 + i * 1.65
    s.box(8.25, y, 4.48, 1.48, '', fill=GRAY_T, radius=0.12)
    s.text(8.47, y + 0.12, 4.1, 0.6, big, size=26, bold=True, color=c)
    s.text(8.47, y + 0.74, 4.1, 0.7, lab, size=13)
s.box(8.25, 4.9, 4.48, 1.25, [Para('"불량 라벨이 도움이 된다"로 보였다.', 14, INK, True, after=4),
                             Para('그런데 정말 결함을 보고 맞힌 걸까?', 14, COPPER, True)],
      align='l', anchor='m', margin=(0.22, 0.08), fill=COPPER_T)
s.text(0.6, 6.4, 12.13, 0.5, 'PatchCore = 정상만 학습, DevNet · KNN+MLP = 사진마다 정상/불량 라벨로 학습 (앞 장). VisA PCB 3개 × 학습·시험 나누기 5번 평균',
       size=12, color=MUTED)

s = d.new('함정: 모델은 결함이 아니라 배경을 보고 있었다', '함정', 1, title_size=26,
          notes='모델이 어디를 보고 판단했는지 Grad-CAM으로 찍어 봤습니다. 결함은 아래쪽 핀에 있는데, 모델은 센서와 기판 전체를 보고 있었습니다. '
                '모델이 본 곳 중 결함 안에 든 비율은 4%로, 아무 데나 볼 때와 같은 수준입니다. 결함을 지운 불량 사진도 44%를 여전히 불량이라고 했습니다. '
                '원인은 데이터였습니다. 불량 사진만 따로, 더 밝게 찍혀서 결함과 관계없는 밝기와 위치 정보만으로도 정상과 불량이 갈립니다. '
                '현장에서도 이런 일이 생길 수 있습니다. 예를 들어 정상 사진은 생산 라인의 검사기에서, 불량 사진은 재검이나 수리 단계의 다른 장비에서 찍혀 모이면, '
                '모델은 결함 대신 장비의 차이를 배웁니다. 원래 시험셋도 같은 차이를 갖고 있어서 점수로는 드러나지 않습니다.')
cols, CW = [0.6, 4.75, 8.9], 3.83
header(s, cols[0], 1.6, CW, 1, '모델이 본 곳: 결함이 아니다', COPPER)
header(s, cols[1], 1.6, CW, 2, '원인: 불량만 따로, 밝게 찍혔다', COPPER)
header(s, cols[2], 1.6, CW, 3, '현장에서도 생긴다 (예시)', GREEN)
s.text(cols[0], 2.08, CW, 0.35, 'PCB1 녹은 납. 왼쪽 원본(흰 선 = 결함), 오른쪽 모델이 본 곳', size=11, color=MUTED)
s.image(GRADCAM, cols[0], 2.65, w=CW, crop=(CAM_ORIG[0], CAM_ORIG[1], CAM_DEVNET[2], CAM_DEVNET[3]))
s.text(cols[1], 2.08, CW, 0.35, 'PCB2 사진의 평균 밝기. 파랑 = 정상, 주황 = 불량', size=11, color=MUTED)
s.image(FIG / 's03_capture_difference.png', cols[1] + 0.25, 2.42, h=2.3, crop=(875, 503, 1405, 862))
for x, (big, lab) in zip(cols[:2], [('4%', '모델이 본 곳 중 결함 안에 든 비율 (아무 데나 봐도 3%). 결함을 지운 사진도 44%를 불량이라 함'),
                                    ('0.93 · 0.85 · 0.76', '결함과 관계없는 밝기 · 위치 정보만으로 정상과 불량을 가른 점수 (AUROC, PCB1 · 2 · 3)')]):
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
s.text(X3, 5.0, CW, 1.2, '우리 데이터도 그랬다: VisA는 불량만 따로 만들어 찍었고, SMT도 같은 자리를 전후로 찍은 사진에서 불량 쪽이 더 밝았다',
       size=13)
s.box(0.6, 6.3, 12.13, 0.68, '불량 라벨을 그냥 쓰면 모델은 결함 대신 "어떤 환경에서 찍혔나"를 배운다. 원래 시험셋도 같은 차이가 있어 점수로는 드러나지 않는다',
      size=14, fill=COPPER_T, align='l', margin=(0.2, 0.04))

# ======================================================================= 2. 방법을 바꿈 · RQ1
d.section('2. 방법을 바꿈 · RQ1')
s = d.new('방법을 바꾸다: 위치로 배우고, 촬영 조건을 섞고, 배경을 섞어 잰다', '방법', 2, title_size=24,
          notes='그래서 방법을 네 가지 바꿨습니다. 첫째, 결함은 사진의 0.2에서 0.5%로 작아서 사진 라벨 하나로는 어디가 결함인지 알 수 없습니다. 결함 위치를 정답으로 주고 칸마다 결함인지 맞히게 합니다. '
                '둘째, 학습 사진의 밝기와 조명, 색을 무작위로 바꿔 촬영 환경으로는 정상과 불량을 가를 수 없게 합니다. '
                '셋째, 결함을 지운 사진과 정상 사진에 결함을 붙인 사진을 시험에 더해, 결함을 봐야만 맞힐 수 있게 잽니다. '
                '넷째, 적은 불량은 생성 모델로 늘립니다. 정상 사진 위에 그리기 때문에 결함 위치도 함께 얻습니다.')
steps = [('1', '위치로 배운다', '결함은 사진의 0.2–0.5%로 작다. 사진 라벨 하나로는 어디가 결함인지 모른다 → 결함 위치를 정답으로 주고, 칸마다 결함인지 맞히게 한다', GREEN),
         ('2', '촬영 조건을 섞는다', '학습 사진의 밝기 · 조명 · 색 · 선명도를 무작위로 바꾼다 → 촬영 환경으로는 정상과 불량을 가를 수 없게 한다', GREEN),
         ('3', '배경을 섞어 잰다', '결함을 지운 사진(정상)과 정상 사진에 결함을 붙인 사진(불량)을 시험에 더한다 → 결함을 봐야만 맞힐 수 있다', GREEN),
         ('4', '합성으로 늘린다', '종류마다 K장으로 생성 모델을 학습해, 정상 사진 위에 가짜 결함을 그린다. 결함 위치도 함께 얻는다', COPPER)]
for i, (n, h, body, c) in enumerate(steps):
    x = 0.6 + i * 3.08
    s.box(x, 1.6, 2.9, 3.0, '', fill=COPPER_T if c == COPPER else TINT, radius=0.12)
    s.oval(x + 0.2, 1.78, 0.5, c, n, size=15)
    s.text(x + 0.82, 1.78, 2.0, 0.5, h, size=15, bold=True, color=c, anchor='m')
    s.text(x + 0.2, 2.45, 2.55, 2.1, body, size=13)
s.text(0.6, 4.82, 8, 0.4, '전체 흐름', size=15, bold=True)
pipeline(s, 5.22, [('실제 불량 K장\n+ 결함 위치', DATA), ('생성 모델 (4)\ncTransGAN · SD+LoRA', AUG),
                   ('합성 불량 + 위치\n(정상 사진 위에)', dict(fill=WHITE, line=COPPER)), ('위치 모델 (1 · 2)\n결함 지도를 학습', ADM),
                   ('평가 (3)\n원래 + 배경 섞은 시험', dict(fill=WHITE, line=GREEN))], h=0.9)
s.box(3.1, 6.45, 7.1, 0.45, '정상 사진 수백 장: 합성의 바탕이자 탐지 모델의 정상 학습 데이터', size=13, **DATA)
s.line(6.65, 6.43, 6.65, 6.14, color=GRAY_L, width=1.5)
s.line(9.15, 6.43, 9.15, 6.14, color=GRAY_L, width=1.5)

s = d.new('모델 구조 ① 생성 모델 cTransGAN: 불량 K장만으로 처음부터 학습', '모델 구조', 2, title_size=26,
          notes='첫 번째 생성 모델은 cTransGAN입니다. Transformer로 만든 GAN을 불량 K장만으로 처음부터 학습하도록 줄였습니다. '
                'K장의 결함 부분으로 결함 종류를 구분하는 작은 분류기를 학습하고, 종류별 평균 특징을 조건으로 줍니다. '
                '생성기는 작은 토큰 격자에서 시작해 두 번 키운 뒤 256×256 사진으로 펼치고, 그 결과를 정상 사진의 결함 자리에 끼워 넣습니다. '
                '진짜와 가짜를 가르는 손실만 쓰면 잡음만 나와서, 결함마다 잡음 벡터를 고정하고 원본을 다시 그리도록 했습니다.')
s.box(0.6, 1.7, 2.45, 0.75, 'K장의 결함 부분\n(32×32로 잘라냄)', size=13, **DATA)
s.line(1.825, 2.47, 1.825, 2.78, color=INK)
s.box(0.6, 2.8, 2.45, 0.95, 'Transformer 분류기\n(결함 종류 구분용)', size=13, **AUG)
s.line(1.825, 3.77, 1.825, 4.08, color=INK)
s.box(0.6, 4.1, 2.45, 0.8, '조건 c\n종류별 평균 특징', size=13, **TEN)
s.text(3.45, 2.3, 4.5, 0.4, '생성기 G', size=14, bold=True, color=COPPER)
s.box(3.45, 2.8, 1.95, 0.8, '잡음 z (128차원)\n결함마다 고정 + 작게 흔듦', size=12, **TEN)
s.line(4.425, 3.62, 4.425, 3.98, color=INK)
for gx, gt in [(3.45, 'z와 c를 합쳐\n8×8 토큰으로'), (5.75, '2배 키우기\n16×16 · Transformer 2층'), (8.05, '2배 키우기\n32×32 · Transformer 1층')]:
    s.box(gx, 4.0, 1.95, 1.0, gt, size=12, **AUG)
for a, b in ((5.4, 5.75), (7.7, 8.05), (10.0, 10.3)):
    s.line(a + 0.02, 4.5, b - 0.02, 4.5, color=INK)
s.line(3.07, 4.5, 3.43, 4.5, color=INK)
s.box(10.3, 4.0, 2.4, 1.0, '토큰마다 16×16 픽셀로\n펼쳐 256×256 사진', size=12, **TEN)
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

s = d.new('모델 구조 ② 생성 모델 SD+LoRA: 미리 학습된 확산 모델에 작은 부분만 추가 학습', '모델 구조', 2, title_size=24,
          notes='두 번째 생성 모델은 Stable Diffusion 1.5의 인페인팅 모델에 LoRA만 붙여 학습한 것입니다. '
                '큰 데이터로 미리 학습된 확산 모델은 질감과 조명을 이미 알기 때문에, K장으로는 어텐션 층에 붙인 작은 행렬만 학습합니다. '
                '결함 종류는 문장으로 알려 주고, 만들 때는 정상 사진에 새 결함 위치를 주고 그 안만 다시 그려 붙입니다.')
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
       (10.6, 2.1, '2배 만들고\n변화 큰 40장 고름', TEN)]
for i, (gx, gw, gt, st) in enumerate(gen):
    s.box(gx, 5.15, gw, 0.95, gt, size=12, **st)
    if i < 4:
        s.line(gx + gw + 0.02, 5.62, gen[i + 1][0] - 0.02, 5.62, color=INK)
s.text(0.6, 6.4, 12.13, 0.6, '왜 SD+LoRA인가: 큰 데이터로 미리 학습된 확산 모델은 질감과 조명을 이미 안다. K장으로는 전체 파라미터의 1% 미만인 작은 행렬만 '
       '학습해, 원본을 그대로 외우는 것을 줄인다', size=12, color=MUTED)

s = d.new('모델 구조 ③ 위치 모델(seg): 결함 지도를 직접 학습', '모델 구조', 2, title_size=26,
          notes='탐지에 쓴 위치 모델입니다. 앞의 모델들과 같은 고정된 WRN50에서 세 층의 특징을 뽑고, 그 위에 작은 디코더만 학습해 128×128 결함 지도를 냅니다. '
                '결함 칸마다 직접 정답이 있어서 작은 결함이 묻히지 않고, 사진 전체의 밝기 차이로는 결함 지도를 맞힐 수 없습니다. '
                '사진의 이상 점수는 결함 지도에서 가장 높은 10칸의 평균이고, 다른 모델과 같은 기준으로 정상과 불량을 판정합니다.')
s.box(0.6, 2.0, 1.9, 1.3, '사진\n512×512', size=12, **DATA)
s.line(2.52, 2.65, 2.88, 2.65, color=INK)
s.box(2.9, 1.65, 2.9, 2.0, '', fill=WHITE, line=GREEN)
s.text(3.0, 1.7, 2.7, 0.35, 'WRN50 (고정)', size=13, bold=True, color=GREEN, align='c')
for i, (lab, sz) in enumerate([('layer1: 128×128칸', 1.0), ('layer2: 64×64칸', 0.9), ('layer3: 32×32칸', 0.8)]):
    s.box(3.1, 2.1 + i * 0.5, 2.5 * sz, 0.4, lab, size=11, fill=TINT, line=None, radius=0.05, margin=(0.04, 0.0))
s.line(5.82, 2.65, 6.18, 2.65, color=INK)
s.box(6.2, 1.65, 3.4, 2.0, [Para('FPN 디코더 (학습)', 13, GREEN, True, align='c', after=4),
                            Para('세 층의 특징을 큰 해상도로', 12, align='c', after=1),
                            Para('키우며 합친다', 12, align='c', after=4),
                            Para('작은 합성곱 층 몇 개만 학습', 12, MUTED, align='c')], **ADM)
s.line(9.62, 2.65, 9.98, 2.65, color=INK)
s.box(10.0, 2.0, 2.7, 1.3, '결함 지도\n128×128', size=14, bold=True, fill=COPPER_T, line=COPPER)
for i, (h, body) in enumerate([('손실', 'focal + dice: 결함 칸이 아주 적어도 잘 배우도록 만든 손실. 정답 = 결함 위치 (VisA는 픽셀 단위, SMT는 상자)'),
                               ('사진의 이상 점수', '결함 지도에서 가장 높은 10칸의 평균 → 다른 모델과 같은 기준으로 정상 / 불량 판정'),
                               ('학습', '한 번에 8장 = 정상 4 + 불량 4 (합성이 있으면 불량 중 절반은 합성). 2000 step'),
                               ('왜 이 구조인가', '결함 칸마다 직접 정답이 있어 작은 결함이 묻히지 않는다. 사진 전체의 밝기 차이로는 결함 지도를 맞힐 수 없다')]):
    x, y = 0.6 + (i % 2) * 6.15, 4.0 + (i // 2) * 1.45
    s.box(x, y, 5.95, 1.25, [Para(h, 14, COPPER if i == 3 else GREEN, True, after=3), Para(body, 13)],
          align='l', anchor='t', margin=(0.2, 0.1), fill=COPPER_T if i == 3 else GRAY_T)

s = d.new('실험 조정 ①: 합성은 그럴듯함보다 흔적 없애기가 중요했다', '실험 조정', 2, title_size=26,
          notes='합성 불량을 만들면서 겪은 시행착오입니다. 처음 cTransGAN은 잡음만 냈고, SD+LoRA는 흰 얼룩을 그리거나 K장을 그대로 외웠습니다. '
                '가장 중요했던 조정은 생성 흔적을 없애는 것이었습니다. 휜 핀을 그릴 때 원래 곧은 핀이 남아 겹쳐 보여서, 원래 핀 자리까지 다시 그리고 남은 핀은 색으로 찾아 지웠습니다. '
                '그리고 합성은 늘 정상 사진 위에 그려서, 불량 사진의 촬영 환경이 따라오지 않게 했습니다.')
for i, (h, body) in enumerate([('cTransGAN', 'K장의 결함 부분으로 처음부터 학습하는 Transformer GAN. 결함마다 잡음 벡터를 고정해 다시 그린다'),
                               ('SD+LoRA', '미리 학습된 확산 모델(SD1.5 인페인팅)에 작은 행렬(LoRA)만 학습. 결함 종류는 문장으로 알려 준다')]):
    s.box(0.6 + i * 3.45, 1.6, 3.3, 1.35, [Para(h, 15, COPPER, True, after=3), Para(body, 12)], align='l', anchor='t',
          margin=(0.18, 0.1), **AUG)
rows = [['시도', '문제', '고친 방법'],
        ['cTransGAN, 진짜·가짜 판정 손실만', '무지개색 잡음만 나옴', '결함마다 잡음 벡터를 고정하고 원본을 다시 그리게 함'],
        ['SD+LoRA 첫 두 버전', '흰 얼룩, K장을 그대로 외움, 결함을 흐리게 그림', '정상 사진 10% 섞기, LoRA 크기 16, 2배 만들고 변화 큰 장만 고름'],
        ['결함 위치 안만 다시 그림', '원래 곧은 핀이 남아 핀이 겹쳐 보임', '원래 핀 자리까지 다시 그리고, 남은 핀은 색으로 찾아 지움']]
s.table(0.6, 3.15, [1.95, 2.05, 2.75], rows, size=12, row_h=0.6)
s.box(0.6, 5.9, 6.75, 1.0, [Para('원칙', 14, GREEN, True, after=2),
                            Para('합성은 늘 정상 사진 위에 그린다 → 불량 사진의 촬영 환경이 합성에 따라오지 않는다', 13)],
      align='l', anchor='t', margin=(0.2, 0.08), fill=TINT)
s.text(7.65, 1.6, 5.08, 0.35, 'SD+LoRA (K=1): 실제 결함 | 바탕 정상 사진 | 합성', size=12, bold=True)
s.image(FIG / 's08_sd_lora_v3_pcb1_sheet.jpg', 7.65, 1.95, w=5.08, crop=(0, 599, 790, 858))
s.text(7.65, 3.72, 5.08, 0.35, 'cTransGAN (K=1): 실제 | 바탕 정상 사진 | 합성 | 결함 위치', size=12, bold=True)
s.image(FIG / 's07_ctransgan_K1_review.png', 7.65, 4.07, w=5.08, crop=(3, 940, 757, 1049))
s.text(7.65, 4.93, 5.08, 0.35, '남은 핀 지우기 (휜 핀): 실제 | 지우기 전 (핀이 겹침) | 지운 후', size=12, bold=True)
s.image(FIG / 's06_pinfix_before_after.jpg', 7.65, 5.28, w=5.08, crop=(0, 633, 790, 892))

s = d.new('실험 조정 ②: 학습은 촬영 조건 섞기, 평가는 배경 섞은 시험셋', '실험 조정', 2, title_size=26,
          notes='원래 시험셋도 정상과 불량의 촬영 환경이 달라서, 배경만 봐도 점수가 잘 나옵니다. 그래서 결함만 바꾼 사진 짝을 만들었습니다. '
                '시험 불량마다 결함을 지운 사진을 만들고, 그 결함을 정상 사진에 붙인 사진도 만듭니다. 결함을 지운 사진은 정상으로, 결함을 붙인 사진은 불량으로 맞혀야 합니다. '
                '배경만 보는 모델은 여기서 틀립니다. 지표는 정상의 5%를 잘못 거르도록 기준을 정했을 때 놓친 불량 비율입니다. '
                '학습 때는 밝기, 조명, 색, 선명도, 잡음을 무작위로 바꿔 촬영 환경 차이가 단서가 되지 않게 했습니다.')
s.box(0.6, 1.6, 4.9, 2.6, [Para('원래 시험셋으로는 안 되는 이유', 16, COPPER, True, after=6),
                           Para('원래 시험셋도 정상과 불량의 촬영 환경이 다르다', 14, bullet=True, after=5),
                           Para('배경만 봐도 점수가 잘 나와, 결함을 보는지 알 수 없다', 14, bullet=True, after=5),
                           Para('→ 결함만 바꾼 사진 짝으로 잰다 (오른쪽)', 14, GREEN, True)],
      align='l', anchor='t', margin=(0.22, 0.14), fill=COPPER_T)
s.box(0.6, 4.35, 4.9, 2.1, [Para('학습: 촬영 조건 섞기', 16, GREEN, True, after=6),
                            Para('밝기 · 대비, 조명 기울기, 채도, 흐림 ↔ 선명, 잡음을 무작위로', 14, bullet=True, after=5),
                            Para('정상 · 실제 불량 · 합성 모두에 똑같이', 14, bullet=True, after=5),
                            Para('→ 촬영 환경 차이가 불량의 단서가 되지 않게', 14, COPPER, True)],
      align='l', anchor='t', margin=(0.22, 0.14), fill=GRAY_T)
FX, FW = 5.85, 6.88
s.text(FX, 1.6, FW, 0.4, '평가: 배경 섞은 시험셋 (결함만 바꾼 사진 짝)', size=16, bold=True, color=GREEN)
sc = FW / 1890
for cx, lab, c in ((230, '시험 불량', COPPER), (610, '결함 지운 사진', NEG), (990, '시험 정상', NEG),
                   (1370, '정상 + 붙인 결함', COPPER), (1750, '정상 + 정상 조각', NEG)):
    s.text(FX + (cx - 45) * sc - 0.68, 2.05, 1.36, 0.35, lab, size=12, bold=True, color=c, align='c')
s.image(FIG / 's13_fair_test_examples.png', FX, 2.42, w=FW, crop=(45, 120, 1935, 1213))
s.text(FX, 6.45, FW, 0.55, '주황 테두리 = 불량으로 맞혀야 함, 파랑 = 정상으로 맞혀야 함. 맨 오른쪽은 "붙이는 작업" 자체가 단서가 되지 않는지 보는 대조군. '
       '지표: 정상의 5%를 잘못 거르는 기준에서 놓친 불량 비율', size=11, color=MUTED)

s = d.new('결과: 위치로 배우면 결함을 보고, 합성은 작은 추가 이득', 'RQ1 · 결과', 2, title_size=26,
          notes='결과입니다. 가장 큰 변화는 결함 위치로 배운 것입니다. 같은 불량 1장이라도 사진 단위 DevNet은 배경 섞은 시험에서 불량의 87%를 놓치는데, 위치 모델은 24%만 놓칩니다. '
                '위치 모델은 결함을 지운 사진을 불량이라 하는 실수도 44%에서 14%로 줄이고, 정상에 붙인 결함은 74%를 잡습니다. '
                '합성은 작은 추가 이득입니다. 촬영 조건 섞기를 쓴 위치 모델에 cTransGAN 합성을 더하면 21.7%에서 18.0%로 줄고, 15번 비교 중 12번 좋아져 우연으로 보기 어렵습니다. '
                'SD+LoRA의 이득은 우연일 수 있는 수준이었고, 정상만 쓰는 PatchCore와는 차이가 없다고 봐야 합니다.')
for i, (h, big, lab, fill, c) in enumerate([
        ('위치로 배우기', '87% → 24%', '놓친 불량, 사진 단위 DevNet → 위치 모델 (배경 섞은 시험, 실제 불량 1장)', TINT, GREEN),
        ('합성 40장 더하기', '21.7% → 18.0%', '놓친 불량, 위치 모델 + 촬영 조건 섞기에 cTransGAN 합성을 더했을 때', COPPER_T, COPPER)]):
    x = 0.6 + i * 6.15
    s.box(x, 1.55, 5.98, 0.95, '', fill=fill, radius=0.1)
    s.text(x + 0.2, 1.6, 1.7, 0.85, h, size=15, bold=True, color=c, anchor='m')
    s.text(x + 1.85, 1.6, 2.55, 0.85, big, size=21, bold=True, color=c, anchor='m')
    s.text(x + 4.4, 1.6, 1.5, 0.85, lab, size=10, color=INK, anchor='m')
header(s, 0.6, 2.68, 5.9, 1, '결함을 보는가 (실제 불량 1장만)', GREEN)
for i, (crop, lab) in enumerate(((CAM_ORIG, '원본 (흰 선 = 결함)'), (CAM_DEVNET, '사진 단위 DevNet'), (CAM_SEG, '위치 모델'))):
    x = 0.6 + i * 2.0
    s.text(x, 3.15, 1.75, 0.3, lab, size=11, bold=True, color=COPPER if i == 1 else (GREEN if i == 2 else INK), align='c')
    s.image(GRADCAM, x + 0.1, 3.45, w=1.55, crop=crop)
rows = [['', '사진 단위 DevNet', '위치 모델'],
        ['모델이 본 곳 중 결함 안 ↑', '4%', '**75%**'],
        ['결함 지운 사진을 불량이라 함 ↓', '44%', '**14%**'],
        ['정상에 붙인 결함을 잡음 ↑', '16%', '**74%**']]
s.table(0.6, 5.1, [3.0, 1.45, 1.45], rows, size=12, row_h=0.42)
header(s, 6.85, 2.68, 5.88, 2, '합성을 더하면 (배경 섞은 시험, 실제 1장 + 합성 40장)', COPPER, size=14)
ch = s.bar_chart(6.85, 3.12, 5.88, 2.25, ['위치 모델', '+ 촬영 조건 섞기'],
                 [('실제만', [0.244, 0.217]), ('+ SD+LoRA', [0.195, 0.205]), ('+ cTransGAN', [0.188, 0.180])],
                 TRAIN_COLORS, vmax=0.30, major=0.1, lay=(0.11, 0.05, 0.86, 0.62), ref=(0.188, None))
pct_axis(ch)
labels_inside(ch)
s.text(9.8, 3.08, 2.9, 0.3, '점선 = PatchCore(정상만) 18.8%', size=11, bold=True, color=COPPER, align='r')
s.box(6.85, 5.45, 5.88, 1.5, [Para('**cTransGAN** 21.7% → 18.0%: 15번 비교 중 12번 좋아짐 (우연으로 보기 어려움)', 12, after=2),
                             Para('**SD+LoRA** 21.7% → 20.5%: 15번 중 9번 (우연일 수 있음)', 12, after=2),
                             Para('**PatchCore**(정상만) 18.8%와는 차이가 없다고 봐야 한다', 12, after=3),
                             Para('15번 = PCB 3개 × 학습·시험 나누기 5번. 95% 신뢰구간: cTransGAN −6.6 ~ −1.1%p, PatchCore 대비 −8.6 ~ +7.8%p', 10, MUTED)],
      align='l', anchor='t', margin=(0.16, 0.08), fill=GRAY_T)

# ======================================================================= 3. RQ2
d.section('3. RQ2 합성은 얼마나')
s = d.new('RQ2. 합성이 도움이 된다면, 얼마나 늘려야 할까?', 'RQ2', 3,
          notes='RQ1에서 위치 모델에 합성을 더하면 조금 도움이 된다는 것을 봤습니다. 그렇다면 얼마나 늘려야 할까요. '
                '세 가지로 나눴습니다. 합성 장수를 늘리면 계속 좋아지는지, 합성을 더한 효과가 실제 불량 몇 장을 더 모은 것과 같은지, 어느 생성 모델을 늘릴 가치가 있는지입니다. '
                '쉬운 데이터인 VisA에서 먼저 보고, 어려운 SMT는 RQ3에서 봅니다.')
s.box(0.6, 1.6, 12.13, 0.75, 'RQ1에서: 위치 모델에 종류마다 합성 40장을 더하자 놓친 불량이 줄었다 (cTransGAN 21.7% → 18.0%).  그렇다면 몇 장을, 어떤 생성 모델로?',
      size=14, fill=TINT, align='l', margin=(0.25, 0.04))
for i, (n, h, body, c) in enumerate([('N', '합성 장수', '종류마다 1장에서 40장까지 늘리면 계속 좋아질까, 어디서 멈출까?', COPPER),
                                     ('K', '실제 장수', '합성을 더한 효과는 실제 불량을 몇 장 더 모은 것과 같을까?', GREEN),
                                     ('G', '생성 모델', '처음부터 학습하는 cTransGAN과 미리 학습된 SD+LoRA 중 어느 쪽을 늘릴 가치가 있을까?', COPPER)]):
    x = 0.6 + i * 4.15
    s.box(x, 2.65, 3.83, 2.85, '', fill=GRAY_T, radius=0.12)
    s.oval(x + 0.3, 2.9, 0.85, c, n, size=26)
    s.text(x + 1.35, 3.05, 2.3, 0.55, h, size=20, bold=True, color=c)
    s.text(x + 0.3, 3.95, 3.25, 1.45, body, size=15)
s.box(0.6, 5.85, 5.6, 0.9, [Para('쉬운 데이터: VisA PCB', 15, GREEN, True, after=2), Para('결함 4종, 픽셀 단위 결함 위치 → RQ2', 13)],
      align='l', anchor='m', margin=(0.25, 0.06), fill=TINT)
s.line(6.3, 6.3, 7.0, 6.3, color=INK, width=2)
s.box(7.13, 5.85, 5.6, 0.9, [Para('어려운 데이터: AIHub SMT', 15, COPPER, True, after=2), Para('실제 생산 라인, 결함 18종, 상자 위치만 → RQ3', 13)],
      align='l', anchor='m', margin=(0.25, 0.06), fill=COPPER_T)

s = d.new('RQ2 실험: 생성 모델 2종 × 실제 장수 4단계 × 합성 장수 6단계', 'RQ2 · 실험', 3, title_size=26,
          notes='VisA에서 생성 모델마다 실제 불량 K를 1, 2, 4, 8장으로, 종류마다 더하는 합성 N을 0에서 40장까지 바꿔 48가지 조건을 돌렸습니다. '
                '생성 모델은 K마다 그 K장으로 새로 학습했고, N장은 같은 40장 중 앞에서부터 잘라 넣었습니다. '
                '탐지 모델은 RQ1에서 정한 위치 모델과 촬영 조건 섞기이고, 평가는 배경 섞은 시험셋입니다. '
                '실제 1, 2, 4장은 학습·시험 나누기를 다섯 번 했고, 실제 8장에 합성을 더한 조건만 한 번이라 경향으로만 봅니다. '
                '실제만 쓴 조건은 3, 5, 6, 10장도 더 돌려서, 합성의 효과를 실제 몇 장으로 읽을 수 있게 했습니다.')
s.text(0.6, 1.6, 6.4, 0.4, 'VisA 실험표 (생성 모델마다 같은 표)', size=15, bold=True)
rows = [['실제 K \\ 합성 N', '0', '1', '5', '10', '20', '40']]
fills = [[None] * 7]
for k in (1, 2, 4, 8):
    nsp = 1 if k == 8 else 5
    rows.append([f'K = {k}', '5'] + [str(nsp)] * 5)
    fills.append([None, GRAY_T] + [TINT if nsp == 5 else None] * 5)
s.table(0.6, 2.05, [1.5] + [0.82] * 6, rows, size=14, row_h=0.6, fills=fills)
s.text(0.6, 5.15, 6.4, 0.7, '칸 숫자 = 학습·시험 나누기를 몇 번 했나 (한 번에 PCB 3개). N = 0은 합성 없이 실제만 (회색). '
       '실제만은 K = 3 · 5 · 6 · 10장도 나누기 5번', size=12, color=MUTED)
s.box(7.3, 1.6, 5.43, 4.15, [Para('설정', 16, GREEN, True, after=6),
                             Para('생성 모델은 K마다 그 K장으로 새로 학습하고, 종류마다 40장을 만든다', 14, bullet=True, after=6),
                             Para('합성 N장 = 같은 40장 중 앞의 N장 (큰 N은 작은 N을 포함)', 14, bullet=True, after=6),
                             Para('탐지 모델 = 위치 모델 + 촬영 조건 섞기 (RQ1의 최종)', 14, bullet=True, after=6),
                             Para('학습할 때 한 번에 정상 4 + 불량 4장. 합성이 있으면 불량 4장 중 2장은 합성', 14, bullet=True, after=6),
                             Para('평가 = 배경 섞은 시험셋에서 놓친 불량 비율. 비교 기준: PatchCore 18.8%', 14, bullet=True)],
      align='l', anchor='t', margin=(0.22, 0.14), fill=GRAY_T)
s.box(0.6, 6.0, 12.13, 0.95, [Para('읽을 때 주의', 14, COPPER, True, after=2),
                              Para('K = 8에 합성을 더한 조건은 나누기를 1번만 했다 → 경향으로만 본다. N과 상관없이 불량의 절반이 늘 합성이라, '
                                   'N을 늘리면 합성의 "양"보다 "다양함"이 바뀐다', 13)],
      align='l', anchor='t', margin=(0.22, 0.08), fill=COPPER_T)

s = d.new('RQ2 결과: 실제 장수가 가장 중요하고, 합성은 20장 안팎에서 멈춘다', 'RQ2 · 결과', 3, title_size=26,
          notes='x축은 종류마다 더한 합성 장수, 선은 실제 불량 장수입니다. 실제 장수를 늘리는 효과가 가장 큽니다. '
                '실제 2장(13.3%)이 실제 1장에 합성 40장을 더한 것(18.0%)보다 낫습니다. '
                'cTransGAN은 합성을 20장까지 늘리면 좋아지다가 거의 멈춥니다. SD+LoRA는 실제 1장에서는 효과가 없고, 실제 2장부터는 cTransGAN과 비슷하게 좋아집니다. '
                '휜 핀은 두 생성 모델 모두 합성을 더하면 오히려 나빠졌습니다.')
s.text(0.6, 1.55, 12.13, 0.4, '배경 섞은 시험셋에서 놓친 불량 비율 (낮을수록 좋음). x = 종류마다 더한 합성 장수, 선 = 실제 장수 K, 점선 = PatchCore 18.8%. '
       'K = 8만 나누기 1번', size=12, color=MUTED)
N = ['0', '1', '5', '10', '20', '40']
sd = [[.217, .196, .239, .229, .217, .205], [.133, .137, .122, .135, .112, .115],
      [.079, .077, .070, .064, .059, .067], [.037, .051, .037, .046, .043, .042]]
ctg = [[.217, .202, .200, .191, .177, .180], [.133, .131, .119, .112, .111, .107],
       [.079, .083, .070, .071, .067, .063], [.037, .036, .037, .026, .022, .023]]
for i, (name, data) in enumerate((('SD+LoRA', sd), ('cTransGAN', ctg))):
    x = 0.6 + i * 6.15
    s.text(x, 1.95, 5.9, 0.4, name, size=16, bold=True, color=COPPER)
    ch = s.line_chart(x, 2.35, 5.98, 3.2, N, [(f'K={k}', v) for k, v in zip((1, 2, 4, 8), data)], K_COLORS,
                      0, 0.25, 0.05, ref=(0.188, None), lay=(0.11, 0.04, 0.86, 0.7))
    pct_axis(ch)
s.box(0.6, 5.7, 5.98, 0.8, '실제 장수가 가장 중요하다: K = 1 → 2 → 4장에서 21.7% → 13.3% → 7.9%. 실제 2장(13.3%)이 실제 1장 + 합성 40장(18.0%)보다 낫다',
      size=12, fill=TINT, align='l', margin=(0.15, 0.05))
s.box(6.75, 5.7, 5.98, 0.8, 'cTransGAN은 합성 20장까지 좋아지고 거의 멈춘다 (K=2: 13.3% → 11.1%, 40장 10.7%). SD+LoRA는 실제 1장에서는 효과가 없고 '
      '2장부터 cTransGAN과 비슷하다. 휜 핀은 둘 다 오히려 나빠진다', size=12, fill=COPPER_T, align='l', margin=(0.15, 0.05))
s.text(0.6, 6.6, 12.13, 0.4, '답: 합성은 종류마다 20장 안팎이면 충분하다. 그럼 합성을 더한 효과는 실제 불량 몇 장만큼일까?',
       size=15, bold=True, color=GREEN)

s = d.new('RQ2 결과: 합성 40장은 실제 불량을 몇 장 더 모은 효과일까', 'RQ2 · 실제 몇 장', 3, title_size=26,
          notes='합성을 더한 효과를 실제 장수로 바꿔 봤습니다. 오른쪽은 실제만으로 1장부터 10장까지 학습한 곡선입니다. '
                '합성을 더한 결과가 이 곡선에서 실제 몇 장 자리에 오는지 읽으면 됩니다. 예를 들어 실제 4장에 cTransGAN 40장을 더하면 6.3%인데, '
                '실제만으로는 6장과 8장 사이, 약 6.6장에서 나오는 성적입니다. '
                '이렇게 읽으면 cTransGAN 40장은 실제 1장을 1.4장, 2장을 2.7장, 4장을 6.6장으로 만듭니다. 실제가 많을수록 합성의 값어치도 커집니다. '
                'SD+LoRA는 실제 1장에서는 효과가 없고, 2장부터는 cTransGAN과 비슷합니다.')
s.text(0.6, 1.55, 12.13, 0.4, '읽는 법: 실제만으로 학습한 곡선(오른쪽)에서, 합성을 더한 결과와 놓친 불량 비율이 같아지는 실제 장수를 읽는다. '
       'VisA 배경 섞은 시험셋, 나누기 5번', size=12, color=MUTED)
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
s.text(6.85, 5.45, 5.88, 0.45, 'x = 실제만 쓴 장수, y = 놓친 불량 비율. 점선이 곡선과 만나는 자리가 실제 몇 장 효과',
       size=11, color=MUTED)
s.box(0.6, 6.0, 6.0, 0.9, [Para('실제가 많을수록 합성의 값어치도 크다', 13, GREEN, True, after=2),
                           Para('cTransGAN 40장: 실제 1장 → 1.4장, 2장 → 2.7장, 4장 → 6.6장', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=TINT)
s.box(6.85, 6.0, 5.88, 0.9, [Para('SD+LoRA는 실제 1장에서는 효과가 없다', 13, COPPER, True, after=2),
                             Para('1.1장 (범위 1장 미만–1.4). 실제 2장부터는 cTransGAN과 비슷하다', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=COPPER_T)

# ======================================================================= 4. RQ3
d.section('4. RQ3 현장 데이터')
s = d.new('RQ3. 현장 데이터에도 같은 문제가 있을까? 데이터 ② AIHub SMT', 'RQ3 · 데이터', 4, title_size=26,
          notes='마지막 질문은 실제 생산 라인 데이터입니다. AIHub의 SMT 검사 데이터에는 기판에 납을 바른 직후의 사전 공정과, 부품을 납땜한 뒤의 납땜 공정이 있고, '
                '결함은 각각 6종과 12종입니다. 사진은 부품 주변 512×512이고, 결함 위치는 상자로만 표시돼 있습니다. '
                '여기에도 같은 문제가 있었습니다. 파일명 코드만으로 정상과 불량이 0.92 수준으로 갈렸고, 같은 자리를 전후로 찍은 사진에서 불량 쪽이 더 밝았습니다. '
                '게다가 정상 사진이 다양해서, 정상만 쓰는 PatchCore는 사전 공정 불량의 91%를 놓칩니다.')
TW = [1.6, 1.9, 1.9, 3.5, 3.23]
rows = [['공정', '정상 예시', '불량 예시', '결함 종류', '장수'],
        ['사전 공정\n(기판에 납을\n바른 직후)', '', '', '미납 · 납부족 · 납쇼트\n납볼 · 위치 밀림 · 모양 불량\n(6종)',
         '시험: 정상 1,000 · 불량 300장\n(종류마다 40장 + 여러 종류 섞임)\n학습 불량: 종류마다 K = 1 · 4 · 16장'],
        ['납땜 공정\n(부품을\n납땜한 뒤)', '', '', '미납 · 냉납 · 밀림 · 쇼트\n오삽 · 미삽 · 역삽 · 뒤집힘\n일어섬 · 납볼 · 핀홀 · 납고드름\n(12종)',
         '시험: 정상 1,000 · 불량 540장\n(종류마다 40장 + 여러 종류 섞임)\n학습 불량: 종류마다 K = 1 · 4 · 16장']]
RH, HH = 1.75, 0.5
s.table(0.6, 1.6, TW, rows, size=13, row_h=RH, header_h=HH, align=['c', 'c', 'c', 'l', 'l'])
for i in range(2):                                   # AIHub 사진은 git에 올릴 수 없어 자리만 둔다 (직접 넣기)
    for j in range(2):
        x, y = 0.6 + TW[0] + j * TW[1], 1.6 + HH + i * RH
        s.box(x + 0.15, y + 0.1, TW[1] - 0.3, RH - 0.2, '사진 자리\n(512×512)', fill=None, line=GRAY_L, dash=True,
              size=11, color=MUTED, name='SMT photo placeholder')
s.box(0.6, 5.72, 12.13, 0.66, [Para('같은 문제가 있다: 파일명 코드만으로 정상 · 불량 구분 점수 0.92, 같은 자리 전후 사진 62쌍에서 불량 쪽이 더 밝다.', 13, after=1),
                              Para('정상 사진도 다양해서, 정상만 쓰는 PatchCore는 사전 공정 불량의 91%를 놓친다', 13)],
      align='l', anchor='m', margin=(0.2, 0.03), fill=COPPER_T)
s.box(0.6, 6.48, 12.13, 0.52, 'RQ3: 결함 위치가 상자뿐이고 정상이 다양한 현장 데이터에서도, 위치 모델과 합성이 효과가 있을까?',
      size=15, bold=True, fill=GRAY_T, align='l', margin=(0.25, 0.04))

s = d.new('RQ3 실험: 같은 흐름을 SMT에, 다른 점만 맞춘다', 'RQ3 · 실험', 4, title_size=26,
          notes='흐름은 VisA와 같고, 다른 점만 맞췄습니다. 위치 모델의 정답은 픽셀 단위 대신 상자입니다. '
                '생성 모델은 공정마다 K장만으로 새로 학습했고, 결함 상자가 부품 크기와 맞지 않아 원래 부품이 반쯤 남던 문제를 고쳐 비슷한 크기의 자리에 그리도록 했습니다. '
                '실제 불량은 1, 4, 16장, 학습·시험 나누기는 세 번이고, 사전 공정에서는 합성을 160장까지 늘려 봤습니다. '
                'SMT에는 배경 섞은 시험셋을 만들지 않아서 원래 시험셋으로 잽니다.')
pipeline(s, 1.65, [('실제 불량 K장\n+ 결함 상자', DATA), ('생성 모델\ncTransGAN · SD+LoRA', AUG),
                   ('합성 불량 + 상자\n(정상 사진 위에)', dict(fill=WHITE, line=COPPER)), ('위치 모델\n상자 = 정답', ADM),
                   ('평가\n원래 시험셋', dict(fill=WHITE, line=GREEN))])
for i, (h, body, fill, c) in enumerate([
        ('결함 위치', 'VisA는 픽셀 단위, SMT는 상자뿐이다. 위치 모델은 상자 안 전체를 결함으로 보고 학습한다', GRAY_T, INK),
        ('합성', '공정마다 그 K장만으로 두 생성 모델을 새로 학습해 종류마다 40장. 결함을 그릴 자리를 고쳤다: 비슷한 크기의 부품 자리에, 미납은 리드 끝, 미삽은 부품 자리 전체',
         COPPER_T, COPPER),
        ('조건', '실제 불량 K = 1 · 4 · 16장 × 학습·시험 나누기 3번, 합성 N = 0–40장. 사전 공정 K = 1 · 4는 160장까지', TINT, GREEN),
        ('평가', '원래 시험셋(공정마다 정상 1,000장)에서 정상의 1%를 잘못 거르는 기준으로 놓친 불량 비율. 배경 섞은 시험셋은 SMT에 만들지 않았다', TINT, GREEN)]):
    x, y = 0.6 + (i % 2) * 6.15, 3.0 + (i // 2) * 1.65
    s.box(x, y, 5.98, 1.45, [Para(h, 16, c, True, after=5), Para(body, 14)], align='l', anchor='t', margin=(0.22, 0.14), fill=fill)
s.text(0.6, 6.4, 12.13, 0.4, '결함 자리를 고친 이유: 결함 상자가 부품 크기와 맞지 않아, 합성 사진에 원래 부품이 반쯤 남았다 (납땜 공정)', size=12, color=MUTED)

s = d.new('RQ3 결과: 현장에서는 불량 라벨이 꼭 필요하고, 합성은 사전 공정에서 확인됐다', 'RQ3 · 결과', 4, title_size=24,
          notes='먼저 정상만 쓰는 PatchCore는 사전 공정에서 불량의 91%를 놓치고, 종류마다 불량 1장만 써도 21%로 줄어듭니다. 현장 데이터에서는 불량 라벨이 꼭 필요합니다. '
                '위치 모델에 합성을 더하면 사전 공정에서는 두 생성 모델 모두 모든 K에서 좋아졌습니다. VisA의 결과가 다시 확인된 것입니다. '
                '어려운 납땜 공정은 실제 불량이 1장, 4장일 때 조금 좋아지고, 16장이면 오히려 나빠졌습니다. '
                '다만 불량의 절반을 늘 합성으로 채웠기 때문에, 실제가 많을 때도 합성이 같은 비중을 차지했습니다. 이 영향은 아직 확인하지 못했습니다.')
s.text(0.6, 1.55, 7.6, 0.4, 'SMT 위치 모델이 놓친 불량 비율 (정상의 1%를 잘못 거르는 기준, 3번 평균, 종류마다 합성 40장)', size=12, color=MUTED)
smt = [('사전', 1, .199, .136, .163), ('사전', 4, .106, .063, .063), ('사전', 16, .034, .016, .019),
       ('납땜', 1, .802, .772, .725), ('납땜', 4, .564, .494, .509), ('납땜', 16, .409, .496, .728)]
rows = [['공정', '실제 K', '실제만', '+ SD+LoRA', '+ cTransGAN']] + [[p, str(k), pct(a), pct(b), pct(c)] for p, k, a, b, c in smt]
fills = [[None] * 5] + [[None, None, None, TINT if b < a else COPPER_T, TINT if c < a else COPPER_T] for _, _, a, b, c in smt]
s.table(0.6, 1.95, [1.2, 1.0, 1.7, 1.85, 1.85], rows, size=15, row_h=0.55, fills=fills)
s.text(0.6, 5.95, 7.6, 0.4, '초록 = 실제만보다 좋아짐, 주황 = 나빠짐', size=12, color=MUTED)
s.box(0.6, 6.35, 7.6, 0.65, '답: 사전 공정에서는 VisA처럼 위치 모델 + 합성이 통한다. 어려운 납땜 공정은 실제가 적을 때만 조금 돕는다',
      size=13, bold=True, fill=TINT, align='l', margin=(0.2, 0.04))
for i, (big, body, c) in enumerate([
        ('91% → 21%', '사전 공정: 정상만 쓰는 PatchCore는 불량의 91%를 놓친다. 종류마다 불량 1장만 써도(DevNet) 21%로 준다 → 불량 라벨이 꼭 필요하다', GREEN),
        ('사전: 모든 K에서 좋아짐', '두 생성 모델 모두 (K=1: 19.9% → 13.6% / 16.3%). 합성을 160장까지 늘려도 K=1은 12–19% 사이에서 오르내린다', GREEN),
        ('납땜 K=16: 나빠짐', '40.9% → 49.6% / 72.8%. 단, 불량의 절반을 늘 합성으로 채워, 실제가 192장이어도 합성이 같은 비중이었다 → 비율을 바꿔 다시 볼 필요', COPPER)]):
    y = 1.6 + i * 1.8
    s.box(8.5, y, 4.23, 1.65, '', fill=GRAY_T, radius=0.12)
    s.text(8.72, y + 0.1, 3.85, 0.5, big, size=19, bold=True, color=c)
    s.text(8.72, y + 0.58, 3.85, 1.05, body, size=12)

s = d.new('RQ3 결과: 합성 40장은 SMT에서도 실제 몇 장 효과일까', 'RQ3 · 실제 몇 장', 4, title_size=26,
          notes='RQ2에서 VisA로 했던 것처럼, SMT에서도 합성을 더한 효과를 실제 장수로 바꿔 봤습니다. '
                '이를 위해 실제만 쓴 조건을 1, 2, 4, 8, 16, 32장으로 늘리고 나누기도 다섯 번으로 늘려 곡선을 만들었습니다. '
                '실제가 적을 때는 VisA와 같은 흐름입니다. 사전 공정은 실제 1장이 2.4에서 2.6장, 4장이 6장 남짓이 되고, 납땜 공정도 실제 4장이 5.4에서 5.9장이 됩니다. '
                '그런데 실제가 16장쯤 모이면 사전 공정은 1%p 남짓 좋아지는 데 그치고, 납땜 공정은 오히려 나빠졌습니다. '
                '그래서 합성은 실제 불량이 쌓일 때까지 쓰고, 충분히 모이면 빼는 것이 맞습니다.')
s.text(0.6, 1.55, 12.13, 0.4, '읽는 법은 RQ2와 같다. 실제만은 1 · 2 · 4 · 8 · 16 · 32장 × 나누기 5번으로 곡선을 만들었다. '
       '원래 시험셋, 정상의 1%를 잘못 거르는 기준', size=12, color=MUTED)
header(s, 0.6, 2.0, 6.6, 1, 'SMT: 합성 40장을 더하면 실제 몇 장 효과', COPPER, size=14)
smt_eq = [('사전 공정', '1장', '2.4장', '2.6장', TINT), ('', '4장', '6.0장', '6.3장', TINT), ('', '16장', '1%p 남짓', '1%p 남짓', GRAY_T),
          ('납땜 공정', '1장', '1.2장', '1.5장', TINT), ('', '4장', '5.9장', '5.4장', TINT), ('', '16장', '나빠짐', '크게 나빠짐', COPPER_T)]
rows = [['공정', '실제', '+ SD+LoRA 40장', '+ cTransGAN 40장']] + [list(r[:4]) for r in smt_eq]
fills = [[None] * 4] + [[None, None, r[4], r[4]] for r in smt_eq]
s.table(0.6, 2.45, [1.5, 1.0, 2.05, 2.05], rows, size=14, row_h=0.43, fills=fills)
s.text(0.6, 5.5, 6.6, 0.45, '16장: 사전 3.4% → 1.6% · 1.9%, 납땜 40.9% → 49.6% · 72.8% (RQ3 결과와 같은 값). '
       '사전 공정 SD+LoRA는 나누기 5번, 나머지는 3번', size=11, color=MUTED)
header(s, 7.45, 2.0, 5.28, 2, '실제 4장이면 VisA와 같다', GREEN, size=14)
ch = s.bar_chart(7.45, 2.45, 5.28, 3.0, ['VisA', 'SMT 사전', 'SMT 납땜'],
                 [('+ SD+LoRA 40장', [6.1, 6.0, 5.9]), ('+ cTransGAN 40장', [6.6, 6.3, 5.4])],
                 TRAIN_COLORS[1:], vmax=8, major=2, fmt='0.0', lay=(0.1, 0.08, 0.88, 0.66), gap=80, ref=(4, None))
ch.value_axis.tick_labels.number_format = '0'
s.text(7.45, 5.5, 5.28, 0.45, '막대 = 실제 4장 + 합성 40장이 실제 몇 장 효과인가. 점선 = 실제만 4장', size=11, color=MUTED)
s.box(0.6, 6.0, 6.6, 0.9, [Para('실제가 적을 때는 VisA와 같은 흐름', 13, GREEN, True, after=2),
                           Para('실제 1장 → 1.2–2.6장, 4장 → 5.4–6.3장 (두 공정 · 두 생성 모델)', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=TINT)
s.box(7.45, 6.0, 5.28, 0.9, [Para('실제가 16장쯤 모이면 효과가 없거나 해가 된다', 13, COPPER, True, after=2),
                             Para('→ 합성은 실제 불량이 쌓일 때까지 쓰고, 모이면 뺀다', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=COPPER_T)

s = d.new('왜 합성은 실제 불량을 대신하지 못할까', 'RQ2 · RQ3 원인', 4, title_size=26,
          notes='VisA와 SMT 모두에서 합성은 실제가 적을 때만 도움이 되고, 실제 불량을 대신하지는 못했습니다. 이유는 세 가지입니다. 첫째, 합성은 같은 실제 1장을 보고 만든 변형이라 새로운 정보가 적습니다. 사진을 작은 칸으로 나눠 비교하면, '
                '실제 시험 결함과 닮은 칸이 합성 안에 있는 비율이 합성 20장에서 8.5%, 160장으로 늘려도 25%입니다. '
                '실제가 4장이면 같은 장수의 합성도 더 다양해지고, 그래서 실제 4장일 때 합성의 효과가 더 컸습니다. '
                '둘째, 합성은 결함을 실제보다 훨씬 눈에 띄게 그립니다. 납쇼트의 경우 결함 상자 안에서 정상과 확실히 달라 보이는 칸이 실제는 4%인데 합성은 66%였고, '
                '이 종류는 합성을 더하자 오히려 더 놓쳤습니다. 셋째, 생성 모델이 못 그리는 종류가 있습니다. '
                '그리고 학습할 때 불량의 절반을 늘 합성으로 채운 영향은 아직 확인하지 못했습니다. 결론적으로 합성은 실제 불량을 대신하지 못하고 보완합니다.')
cols, CW = [0.6, 4.75, 8.9], 3.83
header(s, cols[0], 1.6, CW, 1, '새로운 정보가 적다', COPPER)
header(s, cols[1], 1.6, CW, 2, '결함을 너무 눈에 띄게 그린다', COPPER)
header(s, cols[2], 1.6, CW, 3, '못 그리는 종류가 있다', COPPER)
s.text(cols[0], 2.08, CW, 0.62, '실제 시험 결함과 닮은 부분이 합성 안에 있는 비율 (사진을 32×32칸으로 나눠 칸끼리 비교)', size=11, color=MUTED)
ch = s.bar_chart(cols[0], 2.7, CW, 2.2, ['20장', '40장', '80장', '160장'], [('실제 1장', [0.085, 0.124, 0.176, 0.248]),
                                                                          ('실제 4장', [0.141, 0.212, 0.288, 0.384])],
                 [COPPER, GREEN], vmax=0.5, major=0.25, fmt='0%', lay=(0.13, 0.08, 0.84, 0.6), gap=60)
pct_axis(ch)
ch.plots[0].has_data_labels = False
s.text(cols[0], 4.98, CW, 1.2, '합성은 같은 1장을 보고 만든 변형이다. 160장을 만들어도 실제 결함의 75%는 닮은 부분이 없다 (SMT 사전 공정, SD+LoRA)',
       size=12)
s.text(cols[1], 2.08, CW, 0.62, '결함 상자 안에서 정상과 확실히 달라 보이는 칸의 비율 (SMT 사전 공정, 납쇼트)', size=11, color=MUTED)
ch = s.bar_chart(cols[1], 2.7, CW, 2.2, ['실제 결함', '합성 (cTransGAN)'], [('비율', [0.04, 0.66])], [GRAY_L], vmax=0.8, major=0.4,
                 fmt='0%', legend=False, point_colors=[GREEN, COPPER], lay=(0.13, 0.08, 0.84, 0.74), gap=90)
pct_axis(ch)
s.text(cols[1], 4.98, CW, 1.2, '실제 결함은 미묘한데 합성은 과장되게 그린다 → 이 종류는 합성을 더하자 놓친 불량이 14%p 늘었다. '
       '108개 조건 전체에서도 과장된 합성일수록 효과가 작았다', size=12)
for i, (h, body) in enumerate([('휜 핀 (VisA)', '원래 곧은 핀이 남는 문제. 지운 뒤에도 합성을 더하면 더 놓친다 (14.8% → 21–23%)'),
                               ('미삽 (SMT)', '부품이 빠진 자리에 부품을 다시 그려 넣는 경우가 많았다. 놓친 비율 80% 그대로'),
                               ('뜻', '"없음"이나 "모양이 살짝 다름"처럼 미묘한 결함일수록 합성이 실제를 흉내 내기 어렵다')]):
    s.box(cols[2], 2.15 + i * 1.32, CW, 1.2, [Para(h, 13, COPPER if i < 2 else GREEN, True, after=2), Para(body, 12)],
          align='l', anchor='t', margin=(0.16, 0.08), fill=GRAY_T if i < 2 else TINT)
s.box(0.6, 6.2, 5.98, 0.78, [Para('아직 확인 못 한 것', 12, COPPER, True, after=1),
                             Para('학습할 때 불량의 절반을 늘 합성으로 채웠다. 합성 비율을 줄이면 달라질 수 있다 (다음 실험)', 12)],
      align='l', anchor='m', margin=(0.18, 0.04), fill=COPPER_T)
s.box(6.75, 6.2, 5.98, 0.78, '결론: 합성은 실제 불량을 대신하지 못하고 보완한다. 실제를 더 모을 수 있으면 그게 먼저, 구하기 어려운 결함에 합성을 쓴다',
      size=13, bold=True, fill=TINT, align='l', margin=(0.18, 0.04))

# ======================================================================= 5. 의의 · 한계
d.section('5. 의의 · 한계')
s = d.new('의의: 적은 불량 사진을 제대로 쓰는 법', '의의', 5,
          notes='의의는 네 가지입니다. 같은 불량 한 장이라도 결함 위치를 정답으로 주면 배경 섞은 시험에서 놓친 불량이 87%에서 24%로 줄었습니다. 불량 라벨을 어떻게 쓰느냐가 결과를 가릅니다. '
                '원래 시험 점수로는 보이지 않는 배경 보고 맞히기를 재는 시험셋을 만들었고, 공개 방법도 여기서 대부분을 놓친다는 것을 보였습니다. '
                '합성의 값어치를 실제 장수로 재서, 실제가 적을 때는 실제를 1.1에서 2.6배 모은 효과가 있고 16장쯤 모이면 효과가 없거나 해가 된다는 것을 보였습니다. '
                '그 이유가 다양성 부족과 과장된 결함이라는 것도 확인했습니다. '
                '그리고 실제 라인 데이터에서 정상만 쓰는 방식이 무너지고, 상자 위치만으로도 위치 모델을 학습할 수 있음을 보였습니다.')
for i, t in enumerate(['같은 불량 1장이라도 결함 위치를 정답으로 주면, 배경 섞은 시험에서 놓친 불량이 87% → 24%로 준다. 불량 라벨을 "어떻게" 쓰느냐가 결과를 가른다',
                       '원래 시험 점수로는 안 보이는 "배경 보고 맞히기"를 재는 시험셋을 만들었다. 공개 방법 DRA · DevNet도 이 시험에서 불량의 86–91%를 놓친다 (우리 17%)',
                       '합성의 값어치를 실제 장수로 쟀다: 실제 1–4장이면 합성 40장 = 실제를 1.1–2.6배 모은 효과, 16장이면 효과가 없거나 해. 이유도 확인 (새 정보 부족, 과장된 결함)',
                       '실제 라인 데이터(SMT)에서 정상만 쓰는 방식이 무너지고(91% 놓침), 상자 위치만으로도 위치 모델을 학습할 수 있음을 보였다']):
    y = 1.65 + i * 1.3
    s.oval(0.6, y + 0.08, 0.5, GREEN, str(i + 1), size=14)
    s.text(1.3, y, 5.2, 1.25, t, size=14)
s.box(6.85, 1.6, 5.88, 5.35, '', fill=GRAY_T, radius=0.12)
s.text(7.1, 1.75, 5.4, 0.45, '현장에서 불량 K장을 받았다면', size=18, bold=True, color=COPPER)
for i, t in enumerate(['불량 사진에 결함 위치를 상자로 표시한다', '위치 모델을 촬영 조건 섞기와 함께 학습한다',
                       '실제가 적으면(종류마다 4장 이하) 합성 20–40장을 더하고, 16장쯤 모이면 뺀다',
                       '배포 전에, 결함을 지운 사진은 정상으로 · 붙인 결함은 불량으로 맞히는지 확인한다']):
    y = 2.35 + i * 0.95
    s.oval(7.1, y + 0.05, 0.42, GREEN, str(i + 1), size=13)
    s.text(7.7, y, 4.85, 0.9, t, size=14)
s.box(7.1, 6.15, 5.4, 0.6, '언제나 먼저: 실제 불량을 1장이라도 더 모은다', size=13, bold=True, fill=WHITE, line=COPPER,
      align='l', margin=(0.15, 0.04))

s = d.new('한계와 다음 실험', '한계', 5,
          notes='한계입니다. 표본이 작습니다. VisA에서 실제 8장에 합성을 더한 조건은 나누기를 한 번만 했고, SMT 납땜 공정과 cTransGAN은 세 번입니다. '
                '시험용 휜 핀 불량은 17장뿐이라 종류별 결론은 약합니다. '
                '실제 불량 1장일 때는 정상만 쓰는 PatchCore와 통계적으로 차이가 없습니다. 불량의 절반을 늘 합성으로 채워서, 납땜에서 나빠진 이유가 합성 자체인지 비율인지 모릅니다. '
                '학습에 없던 결함 종류는 시험하지 않았고, VisA는 픽셀 단위로 학습해서 상자만으로 충분한지는 SMT에서만 봤습니다. 오른쪽은 이 한계를 메울 다음 실험입니다.')
for i, (h, body) in enumerate([
        ('표본이 작다', 'VisA 8장 + 합성은 나누기 1번, SMT 납땜은 3번뿐이고, 시험용 휜 핀 불량은 17장 → 종류별 경향은 약한 근거'),
        ('PatchCore와 차이 없음', '실제 불량 1장일 때는 정상만 쓰는 PatchCore와 통계적으로 차이가 없다. PatchCore 성적은 PCB마다 6–29%로 크게 다르다'),
        ('합성 비율 고정', '불량의 절반을 늘 합성으로 채웠다 → 납땜 K=16에서 나빠진 이유가 합성 자체인지 비율인지 모른다'),
        ('처음 보는 결함 · 상자', '학습에 없던 결함 종류는 시험하지 않았다. VisA는 픽셀 단위로 학습해서, 상자만으로 충분한지는 SMT에서만 봤다'),
        ('원래 시험 점수', '정상 · 불량 구분 점수는 최신 방법보다 조금 낮다 (0.971 vs EfficientAD + SegAD 0.992)')]):
    y = 1.6 + i * 1.07
    s.box(0.6, y, 7.1, 0.95, '', fill=GRAY_T, radius=0.1)
    s.text(0.8, y + 0.08, 2.0, 0.8, h, size=14, bold=True, color=COPPER, anchor='m')
    s.text(2.85, y + 0.08, 4.7, 0.8, body, size=13, anchor='m')
s.box(8.0, 1.6, 4.73, 5.3, '', fill=TINT, radius=0.12)
s.text(8.25, 1.75, 4.3, 0.45, '다음 실험', size=18, bold=True, color=GREEN)
s.text(8.25, 2.35, 4.3, 4.5, [Para(t, 15, bullet=True, after=16) for t in
                              ['합성 비율 바꾸기 (불량의 10% · 25% · 실제와 같은 비율) → 납땜 K=16 다시 보기',
                               '결함 종류 하나씩 빼고 학습 → 처음 보는 종류도 잡는지, PatchCore와 점수를 합치면 나은지',
                               'VisA를 상자로 학습 → 상자만으로 충분한지',
                               'SMT 납땜 · cTransGAN 나누기 추가 → 실제 몇 장 효과가 유지되는지']])

MAIN_COUNT = len(d.slides)

# ======================================================================= 6. 부록
d.section('6. 부록')
deck_v2_appendix.build(d, FIG, dict(DATA=DATA, AUG=AUG, ADM=ADM, TEN=TEN, LOSS=LOSS, pct=pct, pct_axis=pct_axis))

d.save(OUT)
args = sys.argv[1:]
preview_dir = ROOT / 'ppt_build' / 'preview_v2'
if '--preview' in args:
    i = args.index('--preview')
    preview_dir = Path(args[i + 1])
    del args[i:i + 2]
d.preview(preview_dir, only=[int(a) for a in args[0].split(',')] if args else None)
print('저장:', OUT, f'(본편 {MAIN_COUNT}장 + 부록 {len(d.slides) - MAIN_COUNT}장)', '미리보기:', preview_dir)
if not deck_lib.PREVIEW_FONTS:
    print('주의: Pretendard 글꼴 파일이 없어 미리보기를 NanumGothic으로 그렸다 (PRETENDARD_DIR)')
for w in d.warnings:
    print('경고:', w)
