"""15분 발표 pptx: 적은 이상 레이블을 최대한 활용하는 PCB 이상탐지.

    /home/junho/miniforge3/envs/pptx/bin/python ppt_build/build_deck.py      (프로젝트 루트에서)
결과: PCB_AD_presentation.pptx (루트), 미리보기 ppt_build/preview/slide-NN.png
숫자 출처: AnomalyDetection/results/ppt/grid_summary.md, fair_ad_K1_seed*.csv, gradcam_K1_seed0.csv, ppt_preview.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deck_lib import (COPPER, COPPER_T, DARK, GRAY_L, GRAY_T, GREEN, GREEN_M, INK, LINE, MUTED, TINT, WHITE,  # noqa: E402
                      Deck, Para)

ROOT = Path('/home/junho/26_2_DeepLearning')
FIG = ROOT / 'ppt_figures'
OUT = ROOT / 'PCB_AD_presentation.pptx'

# 도형 색 규칙: 데이터 = 회색, 생성(Aug) = 구리, AD = 초록, 텐서·중간값 = 흰 바탕
DATA = dict(fill=GRAY_T, line=GRAY_L)
AUG = dict(fill=COPPER_T, line=COPPER)
ADM = dict(fill=TINT, line=GREEN)
TEN = dict(fill=WHITE, line=GRAY_L)
LOSS = dict(fill=WHITE, line=COPPER, dash=True)
K_COLORS = ['9BD3B4', '4FA77F', '1E6B4E', '0B3B2A']           # K 1·2·4·8 (연한 → 진한 초록)
TRAIN_COLORS = ['A3ADA8', 'E3A869', '9C4F12']                  # 실제만 · +SD+LoRA · +cTransGAN

d = Deck()

# ======================================================================= 표지
d.section('표지')
s = d.new('적은 이상 레이블을 최대한 활용하는\nPCB 이상탐지', dark=True, title_size=36,
          notes='불량이 종류당 몇 장뿐이고 그마저 정상과 다르게 찍혔을 때, 그 몇 장을 어떻게 써야 하는지를 다룹니다. '
                '생성으로 늘리고(Aug), 결함 위치로 학습하고(AD), 촬영 환경과 떼어 놓는 세 가지를 차례로 보여 드립니다.')
d.slides[-1].pslide.shapes.title.top = int(2.0 * 914400)
d.slides[-1].pslide.shapes.title.height = int(1.6 * 914400)
s.ops[0] = ('text', 0.6, 2.0, 12.13, 1.6) + s.ops[0][5:]
s.text(0.6, 1.4, 8, 0.45, '2026-2 딥러닝 팀 프로젝트', size=16, color='B8C7BF', bold=True)
s.text(0.6, 3.85, 8.2, 0.5, '불량이 종류당 몇 장뿐이고, 그마저 다르게 찍혔다면 어떻게 써야 하나?', size=18, color='DCE8E1')
s.text(0.6, 5.9, 8, 0.8, ['데이터: VisA PCB1–3 · AIHub SMT 공정(사전·납땜)',
                          '생성: cTransGAN · SD+LoRA      탐지: PatchCore · KNN+MLP · DevNet · 위치 seg'],
       size=14, color='B8C7BF')
for i, (n, t) in enumerate([('①', '위치'), ('②', '개수'), ('③', '환경 분리')]):
    y = 1.6 + i * 1.35
    s.oval(9.7, y, 1.0, [GREEN_M, COPPER, GREEN][i], n, size=24)
    s.text(10.9, y + 0.2, 2.0, 0.6, t, size=22, color=WHITE, bold=True, anchor='m')

# ======================================================================= 목차
s = d.new('목차', notes='여섯 부분으로 진행합니다. 방법론에서는 생성 모델 두 개와 탐지 모델 네 개의 구조를 각각 보여 드립니다.')
items = [('1', '문제 정의', '불량은 적고, 그마저 다르게 찍힌다'),
         ('2', '방법론', 'Aug → AD 흐름, 생성기 2종과 AD 모델 4종의 구조'),
         ('3', '과정', '생성 흔적과 촬영 환경 지름길을 찾아 고친 과정'),
         ('4', '결과', '수치 모음, Grad-CAM, 해석'),
         ('5', '의의 · 결론 · 한계', '현장 활용, 차별점, 남은 문제'),
         ('6', '부록', '생성 예시, 전체 표, 실험 설정')]
for i, (n, t, sub) in enumerate(items):
    col, row = i % 2, i // 2
    x, y = 0.6 + col * 6.2, 1.75 + row * 1.6
    s.box(x, y, 5.9, 1.3, '', fill=TINT if i < 5 else GRAY_T, radius=0.12)
    s.oval(x + 0.3, y + 0.33, 0.64, COPPER if i < 5 else GRAY_L, n, size=20)
    s.text(x + 1.2, y + 0.2, 4.5, 0.45, t, size=20, bold=True)
    s.text(x + 1.2, y + 0.68, 4.5, 0.45, sub, size=14, color=MUTED)

# ======================================================================= 1. 문제 정의
d.section('1. 문제 정의')
s = d.new('불량 레이블은 종류당 몇 장뿐이다', '문제 정의', 1,
          notes='현장에는 정상 사진이 많고 불량은 종류당 몇 장뿐입니다. 표준 방식은 정상만으로 학습하고 불량은 버립니다. '
                '저희 직관은 이 몇 장을 학습에 넣고, 적으니 생성으로 늘려 넣자는 것이었습니다. '
                '그런데 결함이 사진의 0.2–0.5%라 사진 전체로 보면 정상과 거의 같고, 생성 불량을 넣어도 이득이 없었습니다.')
stats = [('1–8장', 'VisA 종류당 학습 불량 K (종류당 최대 10장)'),
         ('0.2–0.5%', '결함이 차지하는 사진 면적'),
         ('18종', 'SMT 결함 유형 (사전 6 · 납땜 12)')]
for i, (big, lab) in enumerate(stats):
    y = 1.7 + i * 1.75
    s.text(0.6, y, 4.6, 0.8, big, size=40, color=COPPER if i == 1 else GREEN, bold=True)
    s.text(0.6, y + 0.85, 4.6, 0.7, lab, size=14, color=MUTED)
cards = [('표준 방식', '정상만 학습한다(PatchCore 등). 불량 몇 장은 버리거나 임계값을 정하는 데만 쓴다.', GRAY_T),
         ('우리의 직관', '불량을 학습에 넣고, 적으니 생성으로 늘려 넣으면 현장 라인에 더 맞지 않을까?', TINT),
         ('막힌 지점', '결함이 작아 사진 전체로 보면 정상과 거의 같다. 생성 불량을 넣어도 이득이 없었다.', COPPER_T)]
for i, (h, body, fill) in enumerate(cards):
    y = 1.7 + i * 1.75
    s.box(5.6, y, 7.1, 1.5, '', fill=fill, radius=0.12)
    s.text(5.9, y + 0.18, 6.6, 0.45, h, size=18, bold=True, color=COPPER if i == 2 else INK)
    s.text(5.9, y + 0.65, 6.6, 0.75, body, size=15)

s = d.new('불량 사진은 정상과 다르게 찍혔다', '문제 정의', 1,
          notes='두 번째 특성은 촬영 환경입니다. VisA 논문도 불량을 사람이 만들어 찍었다고 밝힙니다. '
                '결함과 무관한 밝기·위치 통계만으로 정상과 불량을 AUROC 0.93까지 가를 수 있습니다. '
                '현장 데이터인 SMT도 같았습니다. 모델이 결함 대신 이 촬영 차이를 배우면 라인에서는 불량을 놓칩니다.')
s.image(FIG / 's03_capture_difference.png', 0.6, 1.6, h=4.95)
s.text(0.6, 6.6, 5.7, 0.4, 'PCB별 정상 vs 불량 사진의 평균과 밝기 분포', size=11, color=MUTED)
facts = [('0.93 · 0.85 · 0.76', '결함과 무관한 밝기·위치 통계 8개만으로 낸 정상 vs 불량 AUROC (PCB1·2·3). 정상끼리는 0.40–0.53'),
         ('102 vs 88', 'PCB2 사진 평균 밝기 (불량 vs 정상). 불량의 약 2/3가 정상이 없는 밝기 구간에 몰림'),
         ('72% vs 0%', 'PCB1 기판이 180° 돌아간 비율 (정상 vs 불량)'),
         ('AUROC 0.92', 'SMT 납땜: 파일명 코드만으로 정상/불량 구분. 같은 자리 전후 62쌍에서 불량이 더 밝게 찍힘')]
for i, (big, lab) in enumerate(facts):
    y = 1.65 + i * 1.3
    s.text(6.9, y, 5.8, 0.5, big, size=24, color=COPPER if i == 0 else GREEN, bold=True)
    s.text(6.9, y + 0.5, 5.8, 0.75, lab, size=14, color=INK)

s = d.new('우리가 정한 세 가지 문제와 방향', '문제 정의', 1,
          notes='정리하면 문제는 세 가지입니다. 결함이 작고, 레이블이 적고, 촬영 환경이 다릅니다. '
                '각각에 대해 위치로 학습하고, 생성으로 개수를 늘리고, 촬영 환경과 떼어 놓는 방향을 정했습니다. '
                '그리고 원래 시험셋 점수가 아니라 모델이 결함을 보는지를 기준으로 판단했습니다.')
probs = [('결함이 작다', '사진 면적의 0.2–0.5%. 사진 하나로 요약하면 정상과 거의 같다'),
         ('레이블이 적다', '종류당 1–8장(VisA), 1–16장(SMT). 표준 방식은 이마저 버린다'),
         ('촬영 환경이 다르다', '불량만 따로 찍혀, 모델이 결함 대신 촬영 환경을 배운다(지름길)')]
dirs = [('① 위치', '이미지 라벨 대신 결함 위치(마스크·박스)로 학습한다', '위치 seg'),
        ('② 개수', 'K장으로 학습한 생성기로 불량을 늘린다. 마스크가 같이 나온다', 'cTransGAN · SD+LoRA'),
        ('③ 환경 분리', '정상 사진 위에 생성하고, 촬영 조건을 무작위화하고, 공정 시험셋으로 잰다', '촬영 조건 무작위화 · 공정 시험셋')]
for i in range(3):
    x = 0.6 + i * 4.15
    s.box(x, 1.65, 3.85, 1.75, '', fill=GRAY_T, radius=0.12)
    s.text(x + 0.25, 1.8, 3.4, 0.45, f'문제 {i + 1}. {probs[i][0]}', size=17, bold=True)
    s.text(x + 0.25, 2.3, 3.4, 1.0, probs[i][1], size=14, color=INK)
    s.line(x + 1.925, 3.5, x + 1.925, 3.95, color=GRAY_L, width=2)
    s.box(x, 4.05, 3.85, 2.05, '', fill=TINT if i != 1 else COPPER_T, radius=0.12)
    s.text(x + 0.25, 4.2, 3.4, 0.45, dirs[i][0], size=19, bold=True, color=GREEN if i != 1 else COPPER)
    s.text(x + 0.25, 4.7, 3.4, 0.8, dirs[i][1], size=14)
    s.text(x + 0.25, 5.55, 3.4, 0.4, dirs[i][2], size=13, bold=True, color=MUTED)
s.text(0.6, 6.35, 12.1, 0.45, '판단 기준: 원래 시험셋 점수가 아니라 **모델이 결함을 보는가** (공정 시험셋, 결함 지우기·붙이기)',
       size=15, color=INK)

# ======================================================================= 2. 방법론
d.section('2. 방법론')
s = d.new('전체 흐름: 생성(Aug) → 탐지(AD) → 평가', '방법론', 2,
          notes='전체 흐름입니다. 실제 불량 K장으로 생성기를 학습해 정상 사진 위에 불량과 마스크를 만듭니다. '
                '이 합성 불량을 실제와 함께 AD 모델에 넣고, 원래 시험셋과 공정 시험셋으로 평가합니다. '
                'AD 모델 네 개는 같은 고정 특징 위에서 레이블을 쓰는 방식만 다르게 해 비교했습니다.')
stages = [('실제 불량 K장\n+ 마스크(박스)', DATA, None),
          ('생성기 (Aug)\ncTransGAN · SD+LoRA', AUG, '② 개수'),
          ('합성 불량 + 마스크\n정상 사진 위에', dict(fill=WHITE, line=COPPER), '③ 환경 분리'),
          ('AD 모델 4종\nPatchCore · KNN+MLP\nDevNet · 위치 seg', ADM, '① 위치'),
          ('평가\n원래 시험셋\n+ 공정 시험셋', dict(fill=WHITE, line=GREEN), '③ 환경 분리')]
whys = ['현장엔 종류당 몇 장뿐. 새 불량 유형은 처음엔 0장',
        'K장만으로 학습해 종류당 40장까지 늘린다. 마스크가 같이 나와 위치 라벨이 공짜',
        '정상 사진 위에 그려, 불량도 정상과 같은 촬영 환경을 갖게 한다',
        '같은 고정 특징(WRN50) 위에서 레이블 쓰는 방식만 바꿔 비교한다',
        '촬영 조건이 섞인 라인을 흉내 내 "결함을 보는가"를 잰다']
for i, ((t, st, tag), why) in enumerate(zip(stages, whys)):
    x = 0.6 + i * 2.5
    if tag:
        s.box(x + 0.35, 1.65, 1.4, 0.38, tag, fill=COPPER if '②' in tag else GREEN, size=12, color=WHITE, bold=True,
              radius=0.19, margin=(0.02, 0.0))
    s.box(x, 2.2, 2.1, 1.4, t, size=14, bold=False, **st)
    if i < 4:
        s.line(x + 2.12, 2.9, x + 2.48, 2.9, color=INK, width=1.75)
    s.text(x, 4.95, 2.1, 1.8, why, size=14, color=INK)
s.box(3.1, 3.95, 7.1, 0.5, '정상 사진 (수백 장): 생성의 바탕 · AD 학습의 정상', size=13, **DATA)
s.line(6.65, 3.93, 6.65, 3.62, color=GRAY_L, width=1.5)
s.line(9.15, 3.93, 9.15, 3.62, color=GRAY_L, width=1.5)
s.text(0.6, 4.6, 3, 0.35, '왜 이렇게?', size=14, bold=True, color=COPPER)

# ---- cTransGAN
s = d.new('Aug ① cTransGAN: K장으로 처음부터 학습하는 Transformer GAN', '방법론 · 생성', 2, title_size=26,
          notes='첫 번째 생성기는 cTransGAN입니다. 원 논문의 구조를 K장 설정에 맞게 줄였습니다. '
                'K장 결함 크롭으로 Transformer 분류기를 학습해 종류별 평균 특징을 조건으로 씁니다. '
                '생성기는 8×8 토큰에서 pixel shuffle로 두 번 키운 뒤 토큰을 픽셀 패치로 펼쳐 256×256을 그립니다. '
                'K장만으로 적대 손실만 쓰면 노이즈만 나와서, 결함마다 잠재 벡터를 고정하고 재구성 손실을 더했습니다.')
s.box(0.6, 1.7, 2.45, 0.75, 'K장 결함 크롭\n(32×32)', size=13, **DATA)
s.line(1.825, 2.47, 1.825, 2.78, color=INK)
s.box(0.6, 2.8, 2.45, 0.95, 'Transformer 분류기\n4×4 패치 · 폭 128 · 깊이 2', size=13, **AUG)
s.line(1.825, 3.77, 1.825, 4.08, color=INK)
s.box(0.6, 4.1, 2.45, 0.8, '조건 c\n종류별 평균 특징', size=13, **TEN)
s.text(3.45, 2.3, 4.5, 0.4, '생성기 G', size=14, bold=True, color=COPPER)
s.box(3.45, 2.8, 1.95, 0.8, '잡음 z (128)\n결함마다 고정 + 0.12', size=12, **TEN)
s.line(4.425, 3.62, 4.425, 3.98, color=INK)
gen = [(3.45, 'Linear\n[z, c] → 8×8\n토큰 × 1024'), (5.75, 'PixelShuffle ↑\n16×16 × 256\nTransformer ×2'),
       (8.05, 'PixelShuffle ↑\n32×32 × 64\nTransformer ×1')]
for gx, gt in gen:
    s.box(gx, 4.0, 1.95, 1.0, gt, size=12, **AUG)
for a, b in ((5.4, 5.75), (7.7, 8.05), (10.0, 10.3)):
    s.line(a + 0.02, 4.5, b - 0.02, 4.5, color=INK)
s.line(3.07, 4.5, 3.43, 4.5, color=INK)
s.box(10.3, 4.0, 2.4, 1.0, '토큰마다 16×16 픽셀\n패치로 펼침\n→ 256×256 RGB', size=12, **TEN)
s.line(11.5, 5.02, 11.5, 5.43, color=INK)
s.box(10.3, 5.45, 2.4, 0.95, '정상 바탕(기판 정렬)의\n다시 그릴 영역에만 끼움', size=12, **DATA)
s.line(10.28, 5.92, 10.02, 5.92, color=INK)
s.box(8.05, 5.45, 1.95, 0.95, '판별기 D\nTransformer\n+ 조건 c 토큰', size=12, **AUG)
s.line(8.03, 5.92, 7.72, 5.92, color=INK)
s.box(5.75, 5.45, 1.95, 0.95, '진짜 / 가짜', size=13, **TEN)
s.box(0.6, 5.2, 4.85, 1.45, [Para('손실', 13, COPPER, True, after=2),
                             Para('적대 ×0.25 + 다시 그릴 영역 L1 ×35 + 경계 ×14 + 고주파 ×18 + 크롭 L1 ×6', 12),
                             Para('적대 손실만 쓰면 무지개색 노이즈 → 결함마다 고정한 잠재 벡터로 재구성', 12, MUTED)],
      align='l', anchor='t', margin=(0.15, 0.1), **LOSS)
s.text(0.6, 6.8, 12.1, 0.35, 'Wang et al. (2023) Conditional TransGAN을 K장용으로 축소: G 블록 5/4/2 → 0/2/1, D 깊이 7 → 2, '
       '원래 핀 자리까지 다시 그리고 원본 해상도·JPEG q75로 붙임', size=11, color=MUTED)

# ---- SD+LoRA
s = d.new('Aug ② SD+LoRA: 사전학습 확산 모델에 LoRA만 학습', '방법론 · 생성', 2,
          notes='두 번째 생성기는 Stable Diffusion 1.5 인페인팅 모델에 LoRA만 붙여 학습한 것입니다. '
                '확산 모델은 질감과 조명을 이미 알기 때문에, K장으로는 어텐션 층의 저랭크 행렬만 학습합니다. '
                '결함 종류는 프롬프트로 구분하고, 생성할 때는 정상 사진에 새 마스크를 주고 그 안만 다시 그려 붙입니다.')
s.text(0.6, 1.55, 6, 0.35, '학습 (PCB별 LoRA 1개, 1000 step)', size=14, bold=True, color=COPPER)
s.box(0.6, 1.95, 2.35, 0.95, 'K장 결함 주변 512 크롭\n(정상 샘플 10% 섞음)', size=12, **DATA)
s.line(2.97, 2.42, 3.23, 2.42, color=INK)
s.box(3.25, 1.95, 1.5, 0.95, 'VAE 인코더\n(고정)', size=12, **TEN)
s.line(4.77, 2.42, 5.03, 2.42, color=INK)
s.box(5.05, 1.95, 3.0, 0.95, '입력 9채널 (64×64)\n잡음 latent 4 + 마스크 1\n+ 가린 사진 latent 4', size=12, **TEN)
s.line(6.55, 2.92, 6.55, 3.28, color=INK)
s.box(5.05, 3.3, 3.0, 1.25, [Para('UNet (SD1.5 인페인팅, 고정)', 13, INK, True, align='c', after=2),
                             Para('+ LoRA rank 16', 13, COPPER, True, align='c', after=2),
                             Para('어텐션 q·k·v·out: W + BA', 12, INK, align='c')], **AUG)
s.box(0.6, 3.45, 3.3, 0.95, 'CLIP 텍스트 인코더 (고정)\n결함 종류 = 프롬프트', size=12, **TEN)
s.line(3.92, 3.92, 5.03, 3.92, color=INK)
s.text(3.92, 3.55, 1.1, 0.3, '교차 어텐션', size=10, color=MUTED, align='c')
s.line(8.07, 3.92, 8.43, 3.92, color=INK)
s.box(8.45, 3.5, 1.6, 0.85, '잡음 예측 εθ', size=13, **TEN)
s.line(10.07, 3.92, 10.33, 3.92, color=INK)
s.box(10.35, 3.4, 2.35, 1.05, '손실 MSE(ε, εθ)\nLoRA만 갱신', size=13, **LOSS)
s.text(0.6, 4.75, 6, 0.35, '생성 (종류당 40장)', size=14, bold=True, color=COPPER)
gen = [(0.6, 2.35, '정상 사진 + 새 마스크\n(모양·위치 변형)', DATA), (3.25, 2.3, '같은 UNet + LoRA\n30 step 잡음 제거', AUG),
       (5.85, 1.75, 'VAE 디코더\n(고정)', TEN), (7.9, 2.4, '마스크 안만 정상 사진에\n붙임 · JPEG 맞춤', TEN),
       (10.6, 2.1, '2배 생성 →\n변화 큰 40장', TEN)]
for i, (gx, gw, gt, st) in enumerate(gen):
    s.box(gx, 5.15, gw, 0.95, gt, size=12, **st)
    if i < 4:
        s.line(gx + gw + 0.02, 5.62, gen[i + 1][0] - 0.02, 5.62, color=INK)
s.text(0.6, 6.4, 12.1, 0.6, '왜 SD+LoRA: 대규모로 사전학습된 확산 모델은 질감·조명을 이미 안다 → K장으로는 저랭크 행렬(UNet 파라미터의 1% 미만)만 '
       '학습해 외우기를 줄인다. 정상 사진 위에 그려 생성 흔적이 지름길이 되지 않게 한다', size=12, color=MUTED)

# ---- PatchCore
s = d.new('AD ① PatchCore: 정상만으로 학습하는 기준선', '방법론 · 탐지', 2,
          notes='AD 모델 네 개는 모두 ImageNet으로 학습된 같은 WideResNet50을 고정해서 씁니다. 차이는 레이블을 쓰는 방식뿐입니다. '
                'PatchCore는 정상 사진의 패치 특징을 메모리에 담고, 시험 패치가 가장 가까운 정상 패치에서 얼마나 먼지로 점수를 냅니다. '
                '불량 레이블을 전혀 안 쓰는 기준선입니다.')
s.text(0.6, 1.6, 6, 0.35, '학습: 정상 사진만', size=14, bold=True, color=GREEN)
row = [(0.6, 1.9, '정상 학습 사진\n512×512', DATA), (2.85, 2.05, 'WideResNet50\n(ImageNet, 고정)', ADM),
       (5.25, 2.25, 'layer2: 512ch, 64×64\nlayer3: 1024ch, 32×32', TEN),
       (7.85, 2.3, '3×3 평균 · 업샘플 · 연결\n→ 패치 특징 1024ch', TEN), (10.5, 2.2, '메모리 은행\ncoreset 1%\n(k-center greedy)', ADM)]
for i, (rx, rw, rt, st) in enumerate(row):
    s.box(rx, 2.0, rw, 1.1, rt, size=12, **st)
    if i < 4:
        s.line(rx + rw + 0.02, 2.55, row[i + 1][0] - 0.02, 2.55, color=INK)
s.text(0.6, 3.55, 6, 0.35, '시험', size=14, bold=True, color=GREEN)
row2 = [(0.6, 1.9, '시험 사진', DATA), (2.85, 4.65, '같은 특징 추출 (WRN50 → 패치 특징 1024ch × 64×64)', TEN),
        (7.85, 2.3, '패치마다 메모리의\n최근접 거리', ADM), (10.5, 2.2, '이미지 점수\n= 최댓값', TEN)]
for i, (rx, rw, rt, st) in enumerate(row2):
    s.box(rx, 3.95, rw, 0.95, rt, size=12, **st)
    if i < 3:
        s.line(rx + rw + 0.02, 4.42, row2[i + 1][0] - 0.02, 4.42, color=INK)
s.line(11.0, 3.12, 9.6, 3.93, color=GRAY_L, width=1.5)
s.box(0.6, 5.3, 5.9, 1.2, [Para('공통 특징 추출기', 14, GREEN, True, after=3),
                            Para('네 모델 모두 같은 고정 WRN50 특징을 쓴다. 차이는 불량 레이블을 쓰는 방식뿐이다.', 13)],
      align='l', anchor='t', margin=(0.2, 0.12), fill=GRAY_T)
s.box(6.8, 5.3, 5.9, 1.2, [Para('역할: 레이블을 안 쓰는 기준선', 14, GREEN, True, after=3),
                            Para('정상이 균일한 VisA에서는 강하다(공정 시험셋 0.188). 정상이 다양한 SMT 사전공정에서는 '
                                 '무너진다(0.910).', 13)], align='l', anchor='t', margin=(0.2, 0.12), fill=GRAY_T)

# ---- KNN+MLP / DevNet
s = d.new('AD ② KNN+MLP · DevNet: 이미지 라벨로 학습', '방법론 · 탐지', 2,
          notes='두 번째와 세 번째 모델은 사진 한 장을 2048차원 임베딩 하나로 요약한 뒤, 그 위에 작은 머리를 학습합니다. '
                '불량이 없으면 kNN 거리, 있으면 MLP 분류기나 DevNet을 씁니다. '
                'DevNet은 정상 점수는 기준 분포 근처에, 불량 점수는 5 표준편차 위로 밀어내는 deviation loss를 씁니다.')
row = [(0.6, 2.0, '사진 512×512', DATA), (2.9, 3.0, 'WRN50 고정 패치 특징\n1024ch × 64×64\n(PatchCore와 같음)', TEN),
       (6.2, 3.6, '공간 평균 + 공간 최대 이어 붙임\n→ 이미지 임베딩 2048차원\n(정상 기준으로 표준화)', ADM)]
for i, (rx, rw, rt, st) in enumerate(row):
    s.box(rx, 1.7, rw, 1.05, rt, size=12, **st)
    if i < 2:
        s.line(rx + rw + 0.02, 2.22, row[i + 1][0] - 0.02, 2.22, color=INK)
heads = [('kNN (K = 0)', '정상 임베딩 은행', ['점수 = 가장 가까운 정상 5개까지 평균 거리', '불량 레이블 없음']),
         ('MLP (K > 0)', '2048 → 256 → 1\nReLU · Dropout 0.2', ['손실: BCE', '균형 배치: 정상 절반 + 불량 절반']),
         ('DevNet (K > 0)', '같은 점수망 s(x)', ['d = (s − μR) / σR, R ~ N(0, 1) 5000개',
                                               'L = (1 − y)·|d| + y·max(0, 5 − d)', '정상은 기준 분포 근처, 불량은 5σ 위로'])]
for i, (h, net, body) in enumerate(heads):
    x = 0.6 + i * 4.1
    s.line(8.0, 2.77, x + 1.9, 3.28, color=GRAY_L, width=1.5)
    s.box(x, 3.3, 3.8, 0.5, h, size=15, bold=True, fill=GREEN, color=WHITE)
    s.box(x, 3.9, 3.8, 0.85, net, size=13, **ADM)
    s.text(x + 0.05, 4.9, 3.7, 1.2, [Para(b, 13, bullet=True, after=3) for b in body])
s.box(0.6, 6.15, 12.1, 0.7, '한계: 결함은 면적의 0.2–0.5% → 최대값을 넣어도 사진 하나로 요약하면 결함이 묻히고, 사진 전체의 촬영 차이가 더 크게 보인다',
      size=13, fill=COPPER_T, align='l', margin=(0.2, 0.05))

# ---- 위치 seg
s = d.new('AD ③ 위치 seg: 결함 지도를 직접 학습', '방법론 · 탐지', 2,
          notes='마지막이 저희의 최종 모델인 위치 seg입니다. 같은 고정 WRN50의 세 층 특징 위에 작은 FPN 디코더만 학습해 128×128 결함 지도를 냅니다. '
                '결함 픽셀에 직접 손실이 걸리므로 작은 결함이 묻히지 않고, 사진 전체의 촬영 차이로는 지도를 맞힐 수 없습니다. '
                '이미지 점수는 지도 상위 10칸 평균이고, 다른 모델과 같은 과검률 기준으로 판정합니다.')
s.box(0.6, 2.0, 1.9, 1.3, '사진 512×512\n(채널 평균 맞춤)', size=12, **DATA)
s.line(2.52, 2.65, 2.88, 2.65, color=INK)
s.box(2.9, 1.65, 2.9, 2.0, '', fill=WHITE, line=GREEN)
s.text(3.0, 1.7, 2.7, 0.35, 'WRN50 (고정)', size=13, bold=True, color=GREEN, align='c')
for i, (lab, sz) in enumerate([('layer1: 256ch, 128×128', 1.0), ('layer2: 512ch, 64×64', 0.9), ('layer3: 1024ch, 32×32', 0.8)]):
    s.box(3.1, 2.1 + i * 0.5, 2.5 * sz, 0.4, lab, size=11, fill=TINT, line=None, radius=0.05, margin=(0.04, 0.0))
s.line(5.82, 2.65, 6.18, 2.65, color=INK)
s.box(6.2, 1.65, 3.4, 2.0, [Para('FPN 디코더 (학습)', 13, GREEN, True, align='c', after=4),
                            Para('1×1 conv로 64 / 128 / 128ch', 11, align='c', after=1),
                            Para('layer3 업샘플 + layer2 → 3×3 conv·GN·ReLU', 11, align='c', after=1),
                            Para('업샘플 + layer1 연결 → 3×3 conv → 1×1', 11, align='c')], **ADM)
s.line(9.62, 2.65, 9.98, 2.65, color=INK)
s.box(10.0, 2.0, 2.7, 1.3, '결함 지도\n128×128 (logit)', size=13, bold=True, fill=COPPER_T, line=COPPER)
cards = [('손실', 'focal ×10 + dice. 정답 = 결함 마스크 (SMT는 박스)'),
         ('이미지 점수', '지도 상위 10칸 평균 → 정상 사진으로 과검률 임계값 (다른 모델과 같은 지표)'),
         ('학습', '배치 8 = 정상 4 + 불량 4 (생성이 있으면 불량의 절반은 실제). 2000 step, AdamW 1e-3, cosine'),
         ('왜', '결함 픽셀에 직접 손실이 걸려 작은 결함이 묻히지 않는다. 사진 전체의 촬영 차이로는 지도를 맞힐 수 없다')]
for i, (h, body) in enumerate(cards):
    col, r = i % 2, i // 2
    x, y = 0.6 + col * 6.15, 4.0 + r * 1.45
    s.box(x, y, 5.95, 1.25, [Para(h, 14, COPPER if h == '왜' else GREEN, True, after=3), Para(body, 13)],
          align='l', anchor='t', margin=(0.2, 0.1), fill=GRAY_T if h != '왜' else COPPER_T)

# ---- 데이터 맞춤 조정
s = d.new('데이터 맞춤 조정: 촬영 환경과 떼어 놓기', '방법론 · 학습과 평가', 2,
          notes='촬영 환경 문제에 대한 조정은 학습과 평가 두 곳에 있습니다. 학습에서는 밝기, 조명, 채도, 선명도, 잡음을 무작위로 바꿔 '
                '촬영 차이가 불량 신호가 되지 않게 합니다. 평가에서는 시험 불량마다 결함을 지운 사진과, 정상 사진에 그 결함을 붙인 사진을 만들어 '
                '두 촬영 조건이 반반 섞인 라인을 흉내 냅니다. 촬영 환경만 보는 모델은 여기서 틀립니다.')
s.box(0.6, 1.65, 4.9, 5.1, '', fill=GRAY_T, radius=0.12)
s.text(0.85, 1.8, 4.4, 0.45, '학습: 촬영 조건 무작위화', size=17, bold=True, color=GREEN)
s.text(0.85, 2.3, 4.4, 1.9, [Para(t, 14, bullet=True, after=3) for t in
                             ('밝기 · 대비', '조명 기울기', '채도', '블러 ↔ 샤픈', '잡음')])
s.text(0.85, 4.2, 4.4, 0.8, '정상 · 실제 불량 · 생성 모두에 같은 분포로 → 촬영 차이가 불량 신호가 되지 않게', size=13, color=MUTED)
s.text(0.85, 5.1, 4.4, 0.45, '평가: 공정 시험셋', size=17, bold=True, color=GREEN)
s.text(0.85, 5.6, 4.4, 1.0, '시험 불량 D마다 짝을 만들어 촬영 조건 A·B가 반반 섞인 라인을 흉내 낸다. 주 지표는 과검률 5%의 미검률',
       size=13)
s.image(FIG / 's13_fair_test_examples.png', 5.85, 1.65, w=6.85)
s.text(5.85, 6.1, 6.85, 0.7, ['양성: D 시험 불량 · ND 정상 + 결함', '음성: DR 결함 지운 불량 · N 시험 정상 · NR 붙이기 대조'],
       size=13, color=INK, after=1)

# ======================================================================= 3. 과정
d.section('3. 과정')
s = d.new('생성기 시행착오: 그럴듯함보다 흔적 없애기', '과정', 3,
          notes='생성기를 만들면서 겪은 시행착오입니다. 가장 중요한 것은 그럴듯한 그림보다 생성 흔적을 없애는 것이었습니다. '
                '예를 들어 휜 핀을 그릴 때 원래 곧은 핀이 남아 겹치는 잔상이 있었고, 이것을 색 기반 검출로 지웠습니다. '
                '잔상을 지우자 bent 미검률이 조금 줄었지만 여전히 실제만보다 나빴습니다.')
rows = [['시도', '결과', '바꾼 것'],
        ['cTransGAN, 적대 손실만', '무지개색 노이즈', '결함마다 잠재 벡터 고정 + 재구성 손실'],
        ['결함 마스크 안만 다시 그림', '원래 핀이 남아 "다리가 하나 더"', '원래 핀 자리까지 다시 그림'],
        ['SD+LoRA v1 · v2', '흰 얼룩, K장 암기, 결함을 약하게 그림', '정상 샘플 10%, rank 16, 1000 step, 2배 생성 후 선별'],
        ['bent 핀 잔상 (두 생성기)', '곧은 핀이 겹쳐 남아 bent만 성능 하락', '색 기반 핀 검출로 잔상 제거 (오른쪽)'],
        ['SMT 납땜 결함 자리', '결함 박스가 부품 크기와 안 맞아 부품이 반쯤 남음', '비슷한 크기 자리, nm = 리드 끝, ms = 자리 전체']]
s.table(0.6, 1.7, [2.45, 2.85, 2.9], rows, size=13, row_h=0.62)
s.image(FIG / 's06_pinfix_before_after.jpg', 9.0, 1.7, w=3.7)
s.text(9.0, 3.85, 3.7, 0.6, 'bent 잔상 제거 전후 (K=1). 왼쪽 SD+LoRA, 오른쪽 cTransGAN. 각 [실제 | 전 | 후]', size=11, color=MUTED)
s.box(9.0, 4.65, 3.7, 1.6, [Para('잔상 제거 효과 (K=1 bent 미검률)', 13, COPPER, True, after=4),
                            Para('SD+LoRA 0.248 → **0.230**', 14, after=2), Para('실제만 0.148', 14, after=4),
                            Para('줄었지만 bent는 여전히 손해', 13, MUTED)], align='l', anchor='t', margin=(0.2, 0.12),
      fill=COPPER_T)

s = d.new('이미지 단위의 실패: 모델이 촬영 환경을 배웠다', '과정', 3,
          notes='처음에는 이미지 단위 모델에 생성 불량을 넣었는데 점수가 오히려 나빠졌습니다. 생성 품질을 의심해 생성기를 고쳐도 그대로였고, '
                '진단을 이어가다 결함과 무관한 통계만으로 정상과 불량이 갈린다는 것을 찾았습니다. '
                '결정적인 것은 결함을 지운 불량 사진의 44%를 여전히 불량이라고 한 것입니다. 모델은 결함이 아니라 촬영 환경을 보고 있었습니다.')
steps = [('생성기 개선', '닮음 지표는 오르는데 AD는 그대로'),
         ('임베딩 진단', '생성 불량 ↔ 바탕 정상 거리 = 정상끼리의 0.02–0.07배. 결함이 묻힌다'),
         ('무관한 통계', '밝기 · 위치만으로 정상 vs 불량 AUROC 0.93 / 0.85 / 0.76'),
         ('결함 지우기 · 붙이기', '결함 지운 불량의 44%를 불량 판정, 정상 + 결함은 16%만 검출'),
         ('입력 · 라벨 손보기', '정규화 · 반례 · 약한 라벨 · 패치 단위: 일부만 개선 → 위치 학습으로')]
s.line(1.6, 2.0, 11.6, 2.0, color=LINE, width=2, arrow=False)
for i, (h, body) in enumerate(steps):
    cx = 1.6 + i * 2.5
    s.oval(cx - 0.25, 1.75, 0.5, COPPER if i == 3 else GREEN, str(i + 1), size=14)
    s.text(cx - 1.15, 2.4, 2.3, 0.45, h, size=14, bold=True, color=COPPER if i == 3 else INK, align='c')
    s.text(cx - 1.15, 2.85, 2.3, 1.5, body, size=13, align='c')
for i, lab in enumerate(['D 시험 불량', 'DR 결함 지움', 'N 시험 정상', 'ND 정상 + 결함']):
    s.text(1.0 + i * 2.825, 4.15, 2.825, 0.35, lab, size=13, bold=True, color=COPPER if i in (1, 3) else INK, align='c')
s.image(FIG / 's12_defect_removal_pcb1_bent.jpg', 1.0, 4.5, w=11.3)
s.text(1.0, 6.65, 11.3, 0.35, 'PCB1 bent 짝: 모델이 결함을 보면 결함 지운 사진(DR)은 정상, 정상 + 결함(ND)은 불량이어야 한다',
       size=12, color=MUTED, align='c')

# ======================================================================= 4. 결과
d.section('4. 결과')
s = d.new('같은 K=1장으로도 위치 라벨이 더 많이 얻는다', '결과', 4,
          notes='실제 불량 한 장만 쓸 때, 이미지 라벨 모델인 DevNet과 MLP보다 위치 seg의 미검률이 가장 낮습니다. '
                '오른쪽은 모델이 결함을 보는지입니다. DevNet은 결함을 지운 사진의 44%를 불량이라 하고, 정상에 붙인 결함은 16%만 잡습니다. '
                '위치 seg는 정상 + 결함을 74% 잡습니다.')
s.text(0.6, 1.6, 6.3, 0.4, 'K=1 실제 불량만, 미검률 (원래 시험셋 · 과검 1% · 분할 5개)', size=13, color=MUTED)
s.bar_chart(0.6, 2.0, 6.3, 4.6, ['이미지 DevNet', 'MLP', '위치 seg'], [('미검률', [0.334, 0.569, 0.223])],
            [GRAY_L], vmax=0.7, major=0.1, legend=False, point_colors=[GRAY_L, GRAY_L, GREEN], gap=110,
            lay=(0.11, 0.06, 0.86, 0.8))
s.text(7.3, 1.6, 5.4, 0.4, '결함을 보는가 (K=1, seed 0)', size=15, bold=True)
rows = [['', 'DR ↓', 'ND ↑'], ['DevNet 실제만', '0.44', '0.16'], ['DevNet + SD+LoRA', '0.09', '0.45'], ['위치 seg 실제만', '0.14', '**0.74**']]
s.table(7.3, 2.1, [2.6, 1.4, 1.4], rows, size=15, row_h=0.5)
s.text(7.3, 4.3, 5.4, 2.3, [Para('DR: 결함 지운 불량을 불량이라 한 비율 (낮을수록 좋음)', 13, bullet=True),
                            Para('ND: 정상 사진에 붙인 결함을 잡은 비율 (높을수록 좋음)', 13, bullet=True),
                            Para('위치 seg 지도의 최고점이 결함 위에 있는 비율 0.89', 13, bullet=True)])

s = d.new('공정 시험셋: 생성은 위치 seg에서 도움이 된다', '결과', 4,
          notes='지름길을 뺀 공정 시험셋 결과입니다. 위치 seg에 생성 불량을 넣으면 두 생성기 모두 분할 다섯 개 전부에서 좋아집니다. '
                '촬영 조건 무작위화와 함께 쓰면 SD+LoRA의 이득은 거의 사라지고 cTransGAN은 남아, K=1에서 정상만 쓰는 PatchCore를 넘는 유일한 조합이 됩니다.')
s.text(0.6, 1.6, 7.6, 0.4, '공정 시험셋 미검률 (과검 5%, K=1, 분할 5개, 합성 40장). 점선 = PatchCore(정상만) 0.188', size=13, color=MUTED)
s.bar_chart(0.6, 2.0, 7.4, 4.8, ['위치 seg', 'seg + 촬영 조건 무작위화'],
            [('실제만', [0.244, 0.217]), ('+ SD+LoRA', [0.195, 0.205]), ('+ cTransGAN', [0.188, 0.180])],
            TRAIN_COLORS, vmax=0.30, major=0.05, lay=(0.1, 0.05, 0.87, 0.74), ref=(0.188, None))
cards = [('0.180', 'seg + 무작위화 + cTransGAN: K=1에서 PatchCore를 넘은 유일한 조합 (분할 4/5 개선)', GREEN),
         ('0.244 → 0.19', 'seg에 생성 추가: 두 생성기 모두 분할 5/5 개선', GREEN),
         ('0.86 → 0.61', '이미지 DevNet도 생성으로 좋아지지만 여전히 쓸 수 없는 수준. 원래 시험셋에선 오히려 나빠짐', COPPER)]
for i, (big, body, c) in enumerate(cards):
    y = 1.75 + i * 1.7
    s.box(8.4, y, 4.3, 1.5, '', fill=GRAY_T, radius=0.12)
    s.text(8.65, y + 0.12, 3.9, 0.55, big, size=24, bold=True, color=c)
    s.text(8.65, y + 0.68, 3.9, 0.8, body, size=13)

s = d.new('VisA: 실제 K가 가장 크고, 합성은 생성기에 따라 다르다', '결과', 4, title_size=26,
          notes='x축은 종류당 합성 장수, 선은 실제 불량 수 K입니다. K를 늘리는 효과가 가장 큽니다. '
                'cTransGAN은 합성을 늘릴수록 꾸준히 좋아지지만 SD+LoRA는 경향이 없습니다. '
                '그리고 진짜 한 장을 더하는 것이 합성 수십 장보다 낫습니다.')
s.text(0.6, 1.55, 12.1, 0.4, '공정 시험셋 미검률 (seg + 무작위화, 과검 5%). x = 종류당 합성 장수 N, 선 = 실제 K, 점선 = PatchCore 0.188. K=1은 분할 5개, K≥2는 1개',
       size=13, color=MUTED)
N = ['0', '1', '5', '10', '20', '40']
sd = [[.217, .196, .239, .229, .217, .205], [.155, .174, .163, .184, .150, .151],
      [.076, .055, .043, .050, .058, .068], [.037, .051, .037, .046, .043, .042]]
ctg = [[.217, .202, .200, .191, .177, .180], [.155, .147, .125, .109, .100, .102],
       [.076, .080, .062, .067, .061, .071], [.037, .036, .037, .026, .022, .023]]
for i, (name, data) in enumerate((('SD+LoRA', sd), ('cTransGAN', ctg))):
    x = 0.6 + i * 6.15
    s.text(x, 1.95, 5.9, 0.4, name, size=16, bold=True, color=COPPER)
    s.line_chart(x, 2.35, 5.9, 3.65, N, [(f'K={k}', v) for k, v in zip((1, 2, 4, 8), data)], K_COLORS,
                 0, 0.25, 0.05, ref=(0.188, None), lay=(0.11, 0.04, 0.86, 0.72))
s.box(0.6, 6.15, 5.9, 0.8, '실제 K가 가장 크다: 0.217 → 0.155 → 0.076 → 0.037. K2 실제만(0.155)이 K1 + 합성 20장(0.177)보다 낫다',
      size=13, fill=TINT, align='l', margin=(0.15, 0.05))
s.box(6.75, 6.15, 5.95, 0.8, 'cTransGAN은 N을 늘릴수록 꾸준히 좋아지고 SD+LoRA는 경향이 없다. bent는 둘 다 손해 (0.148 → 0.21–0.23)',
      size=13, fill=COPPER_T, align='l', margin=(0.15, 0.05))

s = d.new('SMT 현장 데이터: 사전공정은 늘 이득, 납땜 K=16은 손해', '결과', 4, title_size=26,
          notes='현장 데이터인 SMT입니다. 정상이 다양해서 정상만 쓰는 PatchCore가 무너지고, 불량 한 장이 결정적입니다. '
                '위치 seg에 합성을 넣으면 사전공정은 두 생성기 모두 모든 K에서 좋아집니다. '
                '납땜 K=16처럼 실제가 충분하면 합성이 오히려 방해가 됩니다.')
s.text(0.6, 1.55, 12.1, 0.4, 'SMT 위치 seg 미검률 (과검 1%, seed 3개 평균, 종류당 합성 40장, 결함 자리 규칙을 고친 생성)', size=13, color=MUTED)
rows = [['공정', 'K', '실제만', '+ SD+LoRA', '+ cTransGAN'],
        ['사전', '1', '0.199', '0.136', '0.163'], ['사전', '4', '0.106', '0.063', '0.063'], ['사전', '16', '0.034', '0.016', '0.019'],
        ['납땜', '1', '0.802', '0.772', '0.725'], ['납땜', '4', '0.564', '0.494', '0.509'], ['납땜', '16', '0.409', '0.496', '0.728']]
fills = [[None] * 5] + [[None, None, None, TINT if float(r[3]) < float(r[2]) else COPPER_T,
                         TINT if float(r[4]) < float(r[2]) else COPPER_T] for r in rows[1:]]
s.table(0.6, 2.05, [1.2, 0.9, 1.7, 1.9, 1.9], rows, size=15, row_h=0.55, fills=fills)
s.text(0.6, 6.0, 7.6, 0.7, '초록 = 실제만보다 좋아짐, 구리 = 나빠짐. ms(미삽) 따로 실험은 부록', size=12, color=MUTED)
cards = [('0.910 → 0.213', '사전공정: 정상만 쓰는 PatchCore가 무너지고, 불량 1장(DevNet K=1)으로 크게 좋아진다 → 불량 레이블이 필수', GREEN),
         ('사전공정', '두 생성기 모두 모든 K에서 좋아진다 (K=1: 0.199 → 0.136 / 0.163)', GREEN),
         ('납땜 K=16', '둘 다 나빠진다 (0.409 → 0.496 / 0.728). 실제가 충분하면 합성이 방해', COPPER)]
for i, (big, body, c) in enumerate(cards):
    y = 2.05 + i * 1.6
    s.box(8.55, y, 4.15, 1.4, '', fill=GRAY_T, radius=0.12)
    s.text(8.8, y + 0.1, 3.7, 0.5, big, size=20, bold=True, color=c)
    s.text(8.8, y + 0.58, 3.7, 0.8, body, size=12)

s = d.new('Grad-CAM: 이미지 모델은 기판 전체를, seg는 결함을 본다', '결과', 4, title_size=26,
          notes='모델이 어디를 보는지 확인했습니다. 이미지 DevNet의 Grad-CAM은 기판 전체에 퍼져 있고, 결함 영역 안의 질량은 우연 수준입니다. '
                '위치 seg의 지도는 결함 위에 모이고, 생성을 넣으면 더 좁고 정확해집니다.')
s.image(FIG / 's12_s14_gradcam_vs_seg_map.jpg', 0.6, 1.6, w=6.9, crop=(0, 80, 1430, 935))
s.text(0.6, 5.8, 6.9, 0.9, 'PCB1 melt, K=1. 행: D 시험 불량 / DR 결함 지움 / ND 정상 + 결함. 열: 원본 · DevNet Grad-CAM(실제만, +SD+LoRA) · '
       'seg 지도(실제만, +SD+LoRA). 흰 선 = 결함, 자홍 원 = 최고점', size=11, color=MUTED)
rows = [['D(시험 불량) 기준', '결함 안\n질량', '최고점\n결함 위', '질량 절반\n면적'],
        ['DevNet 실제만', '0.04', '0.11', '16%'], ['DevNet + SD+LoRA', '0.05', '0.20', '12%'],
        ['위치 seg 실제만', '**0.75**', '**0.86**', '2.3%'], ['위치 seg + SD+LoRA', '**0.86**', '**0.88**', '**0.2%**']]
s.table(7.8, 1.65, [2.2, 0.9, 0.9, 0.9], rows, size=13, row_h=0.5)
s.text(7.8, 4.45, 4.9, 0.4, '결함 영역은 사진의 2.8% → 우연 수준 질량 ≈ 0.03', size=12, color=MUTED)
s.box(7.8, 5.0, 4.9, 1.35, [Para('DevNet이 결함 지운 사진(DR)을 불량이라 할 때', 13, after=2),
                           Para('지운 자리 안 질량 0.03 = 우연', 15, COPPER, True, after=2),
                           Para('→ 결함이 아니라 기판 전체(촬영 환경)를 본다', 13)], align='l', anchor='t',
      margin=(0.2, 0.12), fill=COPPER_T)

s = d.new('결과 해석', '결과', 4,
          notes='결과를 네 줄로 정리합니다. 위치로 학습하면 같은 몇 장에서 더 많이 얻습니다. 이미지 단위 모델은 촬영 환경 지름길을 배웁니다. '
                '진짜 한 장이 합성 수십 장보다 큽니다. 합성은 위치 seg 위에서, 생성기가 결함을 제대로 그릴 때 도움이 됩니다.')
takes = [('위치로 학습하면 같은 K장에서 더 얻는다', 'K=1 미검률 DevNet 0.334 → seg 0.223. seg 지도의 결함 안 질량 0.75 vs DevNet 0.04', GREEN),
         ('이미지 단위 모델은 촬영 환경 지름길을 배운다', '결함 지운 불량의 44%를 불량 판정. Grad-CAM 질량은 우연 수준', COPPER),
         ('진짜 1장이 합성 수십 장보다 크다', 'VisA: K2 실제만 0.155 vs K1 + 합성 20장 0.177. SMT 사전: K4 실제만 0.106 vs K1 + 합성 40장 0.136', GREEN),
         ('합성은 위치 seg 위에서, 결함을 제대로 그릴 때 돕는다', 'cTransGAN·SMT 사전공정은 꾸준히 이득, bent·납땜 K=16은 손해. 닮음 지표로는 미리 가릴 수 없었다', COPPER)]
for i, (h, body, c) in enumerate(takes):
    col, r = i % 2, i // 2
    x, y = 0.6 + col * 6.15, 1.7 + r * 2.6
    s.box(x, y, 5.95, 2.35, '', fill=TINT if c == GREEN else COPPER_T, radius=0.12)
    s.oval(x + 0.25, y + 0.25, 0.6, c, str(i + 1), size=18)
    s.text(x + 1.05, y + 0.22, 4.7, 0.75, h, size=17, bold=True)
    s.text(x + 1.05, y + 1.05, 4.7, 1.2, body, size=15)

# ======================================================================= 5. 의의 · 결론 · 한계
d.section('5. 의의 · 결론 · 한계')
s = d.new('의의: 같은 레이블의 값어치를 올리는 법', '의의', 5,
          notes='차별점은 세 가지입니다. 레이블 단위만 바꿨는데 생성의 효과가 해로움에서 도움으로 뒤집혔습니다. '
                '촬영 환경 차이를 재고 지름길을 뺀 평가를 만들었습니다. 그리고 K장 전용 생성기 두 종류를 같은 규칙에서 비교했습니다. '
                '현장에서는 불량 유형이 정해진 라인에서 몇 장에 박스만 그려 쓰는 방식을 제안합니다.')
s.text(0.6, 1.6, 6, 0.45, '차별점', size=18, bold=True, color=GREEN)
diffs = ['같은 K장 · 같은 생성 데이터에서 레이블 단위만 바꿨더니 생성의 효과가 해로움에서 도움으로 뒤집혔다',
         '불량 사진의 촬영 환경 차이를 재고, 지름길을 뺀 평가(공정 시험셋, 결함 지우기·붙이기)를 만들었다',
         'K장 전용 생성기 두 종류(처음부터 학습하는 GAN vs 사전학습 확산 + LoRA)를 같은 규칙에서 비교했다']
for i, t in enumerate(diffs):
    y = 2.15 + i * 1.3
    s.oval(0.6, y + 0.1, 0.5, GREEN, str(i + 1), size=14)
    s.text(1.3, y, 4.9, 1.3, t, size=15)
s.box(6.75, 1.6, 5.95, 4.3, '', fill=GRAY_T, radius=0.12)
s.text(7.0, 1.75, 5.5, 0.45, '현장 활용', size=18, bold=True, color=COPPER)
s.text(7.0, 2.3, 5.5, 4.3, [Para('**불량 유형이 정해진 라인**(IPC-A-610 등): 불량 몇 장에 박스·마스크만 그려 위치 seg + 생성으로 학습', 14, bullet=True, after=8),
                            Para('**배포 전 점검**: 결함 지우기·붙이기로 모델이 결함을 보는지 확인', 14, bullet=True, after=8),
                            Para('**정상이 다양한 라인**(SMT): 정상만 쓰는 방식은 무너진다 → 불량 레이블이 필수', 14, bullet=True, after=8),
                            Para('**새 불량 유형 대비**: 정상만 쓰는 PatchCore와 함께 (제안, 시험 안 함)', 14, bullet=True, after=8),
                            Para('근거: PCB 데이터 두 종류 (VisA 사람이 만든 불량, SMT 현장 수집)', 13, MUTED)])

s = d.new('결론과 한계', '결론', 5, dark=True,
          notes='결론입니다. 이상 레이블이 적고 불량 사진의 촬영 환경이 다를 때, 레이블은 위치까지 쓰고, 생성으로 늘리고, 촬영 환경과 떼어 놓아야 제대로 쓰입니다. '
                '한계는 seed 수, 생성기와 유형에 따라 다른 합성 효과, 처음 보는 불량 유형, 데이터 범위입니다. '
                '다음 단계로는 어떤 합성이 도움이 될지 미리 판정하는 방법을 제안합니다.')
s.text(0.6, 1.65, 12.1, 1.3, '이상 레이블이 적고 불량 사진의 촬영 환경이 다를 때, 레이블은 **위치**까지 쓰고, 생성으로 **늘리고**, '
       '촬영 **환경과 떼어 놓아야** 제대로 쓰인다', size=22, color=WHITE)
s.box(0.6, 3.35, 6.0, 2.75, '', fill='1B3B30', radius=0.12)
s.text(0.85, 3.5, 5.5, 0.45, '한계', size=17, bold=True, color='F0C9A0')
s.text(0.85, 4.0, 5.5, 2.7, [Para(t, 14, 'E3ECE7', bullet=True, after=5) for t in
                             ('seed 수가 적다 (VisA K≥2 분할 1개, SMT 3개)',
                              '합성 이득이 생성기·유형·K마다 다르다 (bent, 납땜 K=16은 손해)',
                              '처음 보는 불량 유형은 시험하지 않았다',
                              '데이터가 PCB 두 종류다')])
s.box(6.85, 3.35, 5.85, 2.75, '', fill='1B3B30', radius=0.12)
s.text(7.1, 3.5, 5.4, 0.45, '다음 단계: 의미 있는 합성 판정', size=17, bold=True, color='F0C9A0')
s.text(7.1, 4.0, 5.4, 2.7, [Para('닮음 지표는 유형별 이득을 못 맞힌다 (순위상관 0.17)', 14, 'E3ECE7', bullet=True, after=5),
                            Para('정상만 학습한 모델로 "마스크 안은 이상, 밖은 정상"인지 재서 라벨이 맞는 합성만 쓰기', 14, 'E3ECE7', bullet=True, after=5),
                            Para('목표: 어떤 유형·K에서도 손해 없는 합성', 14, 'E3ECE7', bullet=True)])

# ======================================================================= 6. 부록
d.section('6. 부록')
s = d.new('부록 A. 생성 예시 (잔상 제거 후, K=1)', '부록', 6)
s.image(FIG / 's08_sd_lora_v3_pcb3_sheet.jpg', 0.6, 1.7, w=6.2)
s.text(0.6, 6.25, 6.2, 0.5, 'SD+LoRA v3, PCB3. 종류마다 [실제 K=1 결함 | 정상 바탕 | 생성] × 2', size=12, color=MUTED)
s.image(FIG / 's07_ctransgan_K1_review.png', 7.1, 1.7, w=5.6, crop=(0, 1216, 770, 1834))
s.text(7.1, 6.25, 5.6, 0.5, 'cTransGAN, PCB3. 종류마다 [실제 결함 | 정상 바탕 | 생성 | 마스크]', size=12, color=MUTED)

s = d.new('부록 B. VisA 생성기 × K × N 전체와 bent', '부록', 6)
s.image(FIG / 's14_grid_visa.png', 0.6, 1.7, w=7.4)
s.text(0.6, 4.75, 7.4, 0.5, '공정 시험셋 미검률. 점선 = PatchCore(정상만)', size=12, color=MUTED)
rows = [['bent 미검률', 'N=0', '1', '5', '10', '20', '40'],
        ['SD K1', '.148', '.175', '.299', '.260', '.262', '.230'], ['SD K2', '.141', '.183', '.162', '.183', '.162', '.204'],
        ['SD K4', '.037', '.037', '.046', '.065', '.065', '.037'], ['SD K8', '.019', '.019', '.000', '.000', '.009', '.009'],
        ['cTG K1', '.148', '.195', '.206', '.216', '.222', '.206'], ['cTG K2', '.141', '.183', '.141', '.141', '.100', '.204'],
        ['cTG K4', '.037', '.046', '.074', '.093', '.093', '.102'], ['cTG K8', '.019', '.019', '.028', '.000', '.065', '.102']]
s.table(8.3, 1.7, [1.1] + [0.55] * 6, rows, size=11, row_h=0.36)
s.text(8.3, 5.15, 4.4, 1.4, 'bent는 두 생성기 모두 합성을 넣으면 대체로 나빠진다. 잔상을 지운 뒤에도 남은 문제 (K=1은 휜 핀 모양이 하나뿐)',
       size=12, color=INK)

s = d.new('부록 C. SMT 생성기별 그림과 ms(미삽) 따로 실험', '부록', 6)
s.image(FIG / 's14_grid_smt.png', 0.6, 1.65, h=5.15)
rows = [['K', 'N', 'SD가 그린 ms', '실제 빈 패드'], ['1', '5', '0.82', '0.76'], ['1', '40', '0.80', '0.77'],
        ['4', '5', '0.67', '0.67'], ['4', '40', '0.65', '0.65'], ['16', '5', '0.63', '0.64'], ['16', '40', '0.61', '0.60']]
s.text(7.75, 1.65, 4.95, 0.4, '납땜 ms 미검률 (seed 3개)', size=14, bold=True)
s.table(7.75, 2.1, [0.7, 0.7, 1.75, 1.75], rows, size=13, row_h=0.4, bold_first_col=False)
s.text(7.75, 5.1, 4.95, 1.6, 'SD는 ms 자리에 부품을 다시 그려 "미삽"이 안 되는 장이 많았다. 실제 빈 패드를 붙이고 약하게(strength 0.3) 다시 그린 세트로 바꾸면 '
       'K=1에서만 효과가 있다', size=12, color=INK)

s = d.new('부록 D. 기존 연구와 우리의 위치', '부록', 6)
refs = [('정상만 학습', 'PatchCore 등. VisA 논문의 기본 평가도 정상만 학습(1-class)'),
        ('불량 레이블 사용', 'DevNet · DRA · AHL(이미지 라벨), PRN(마스크까지). 표준 시험셋 점수로 비교'),
        ('Baitieva et al. (CVPR 2024)', 'VisA에서 지도 학습 점수가 유난히 높다고 지적 → 우리는 촬영 환경 지름길이라는 설명을 더한다'),
        ('주의', '"기존 연구는 불량 레이블을 안 썼다"가 아니다. 불량 사진의 촬영 환경 차이를 따지지 않았다는 것')]
for i, (h, body) in enumerate(refs):
    col, r = i % 2, i // 2
    x, y = 0.6 + col * 6.15, 1.75 + r * 2.1
    s.box(x, y, 5.95, 1.8, [Para(h, 17, COPPER if i == 3 else GREEN, True, after=6), Para(body, 14)], align='l', anchor='t',
          margin=(0.25, 0.2), fill=GRAY_T if i < 3 else COPPER_T)

s = d.new('부록 E. 실험 설정', '부록', 6)
rows = [['항목', '설정'],
        ['VisA', 'PCB1–3, 결함 4종(bent · melt · scratch · missing), 분할 seed 7017 고정. K = 1, 2, 4, 8, K=1은 분할 5개, K≥2는 1개'],
        ['SMT (AIHub)', '사전 6종 · 납땜 12종, 박스 라벨. K = 1, 4, 16 × seed 0–2, 공정별 시험 정상 1000장'],
        ['지표', '미검률 = 과검률 1%(원래 시험셋) · 5%(공정 시험셋) 임계값에서 놓친 불량 비율. 낮을수록 좋음'],
        ['생성', '종류당 40장. 합성 장수 N = 1, 5, 10, 20, 40은 고정 순서의 앞 N장. seg 배치에서 실제 : 생성 = 반반'],
        ['seg 학습', '2000 step, 배치 8, AdamW 1e-3, cosine, focal ×10 + dice. 헤드 seed 2개'],
        ['DevNet · MLP', '600 step, 배치 64, Adam 1e-3, 2048 → 256 → 1. 헤드 seed 2개'],
        ['생성기 학습', 'cTransGAN: 크롭당 125 업데이트, Adam(G 5e-4, D 1e-4). SD+LoRA: 1000 step, rank 16'],
        ['환경', 'GPU 서버(A100). 특징 추출기는 모두 ImageNet WRN50 고정']]
s.table(0.6, 1.7, [2.2, 9.9], rows, size=13, row_h=0.52, align=['l', 'l'])

s = d.new('부록 F. 원래 시험셋 K 곡선과 지름길 진단', '부록', 6)
s.image(FIG / 's11_kcurve_standard_test.png', 0.6, 1.7, w=5.9)
s.text(0.6, 5.55, 5.9, 0.6, '원래 시험셋 K 곡선, 실제 불량만 (PatchCore, DevNet, kNN/MLP)', size=12, color=MUTED)
s.image(FIG / 's12_shortcut_diagnosis.png', 6.8, 1.7, w=5.9)
s.text(6.8, 4.1, 5.9, 0.6, '검출기가 무엇을 보나: 실제 불량 / 결함 지운 사진 / 정상에 붙인 결함을 불량이라 한 비율 (K=1, 파랑 = 실제만, 주황 = + SD+LoRA)', size=12, color=MUTED)

d.save(OUT)
paths = d.preview(ROOT / 'ppt_build' / 'preview', only=[int(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else None)
print('저장:', OUT, f'({len(d.slides)}장)')
for w in d.warnings:
    print('경고:', w)
