"""공개 수치 비교 1–3단계 정리 (11/20) → AnomalyDetection/results/ppt/public_comparison_summary.md

입력: results/official/1cls.csv, 1cls_256.csv (1단계), segad_protocol.csv + results/external/{dra,devnet}.csv (3단계),
      results/official/2cls_fewshot.csv, 2cls_fewshot_ctg.csv (2단계)
공개 수치(논문 표에서 옮김, 단위 0–1):
  PatchCore 1cls  FAIR(arXiv 2309.07068, WRN-50 256, 공식 코드 재실행) / EfficientAD 논문의 재실행 JSON(WRN-101 224)
  SegAD 규칙      SegAD(CVPR 2024) 부록 표 8, seed 10개: DRA, DevNet, EfficientAD(정상만), EfficientAD + SegAD
  2cls_fewshot    VisA 논문 부록 표 12·14 (10-shot = 학습에서 정상·불량 10장씩만 뽑음, 5번 평균, ResNet-50 분류 / U-Net 분할)

    python -m AnomalyDetection.ad.public_compare_summary
"""
import numpy as np
import pandas as pd

from .data import PCBS, ROOT

R = ROOT / 'AnomalyDetection/results'
PUB_PC = {'FAIR (WRN-50, 256)': {'pcb1': (.984, .999, .943), 'pcb2': (.972, .990, .899), 'pcb3': (.980, .993, .908)},
          'EfficientAD 재실행 (WRN-101, 224)': {'pcb1': (.9886, .9949, .9132), 'pcb2': (.9712, .9746, .8574),
                                               'pcb3': (.9632, .9545, .7319)}}
PUB_SEGAD = {'DRA (논문)': (.921, .916, .777), 'DevNet (논문)': (.944, .895, .734), 'EfficientAD, 정상만 (논문)': (.986, .999, .984),
             'EfficientAD + SegAD (논문)': (.995, .997, .984)}
PUB_VISA = {'분류 ResNet-50, 10-shot (VisA 논문)': ('image_auroc', (.942, .892, .862)),
            '분할 U-Net, 10-shot (VisA 논문)': ('pixel_auroc', (.665, .729, .758))}
NAME = {'patchcore': '정상만 배우는 기존 방법 (PatchCore)', 'devnet': '사진 전체 모델 (우리 단순판 DevNet)',
        'seg': '위치 모델', 'seg_aug': '위치 모델 + 촬영 환경 섞기', 'dra': 'DRA 공개 코드', 'devnet_official': 'DevNet 공개 코드',
        '+sd_lora_official_pinfix': '+ 가짜 불량 SD+LoRA', '+ctransgan_official_pinfix': '+ 가짜 불량 cTransGAN'}


def f(x, n=3):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def pct(x):
    return '' if x is None or np.isnan(x) else f'{100 * x:.0f}%'


def stage1(lines):
    lines += ['## 1단계: 정상만 배우는 기존 방법(PatchCore) 재현, 공식 "정상만" 분할', '',
              '사진 / 픽셀 / 결함 덮기(AUPRO). 1이면 완벽.', '',
              '| 설정 | PCB1 | PCB2 | PCB3 |', '|---|---|---|---|']
    for name, v in PUB_PC.items():
        lines.append(f'| 공개: {name} | ' + ' | '.join(' / '.join(f(x) for x in v[p]) for p in PCBS) + ' |')
    for fn in ('1cls.csv', '1cls_256.csv'):
        p = R / 'official' / fn
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for cfg, g in d.groupby('train', sort=False):
            g = g.set_index('pcb')
            lines.append(f'| 우리 {cfg} (해상도:자르기:coreset) | ' +
                         ' | '.join(f'{f(g.loc[p, "image_auroc"])} / {f(g.loc[p, "pixel_auroc"])} / {f(g.loc[p, "aupro"])}'
                                    for p in PCBS) + ' |')
    lines.append('')


def stage3(lines):
    d = pd.read_csv(R / 'official/segad_protocol.csv')
    ext = [pd.read_csv(R / f'external/{m}.csv').rename(columns={'method': 'model'}) for m in ('dra', 'devnet')
           if (R / f'external/{m}.csv').exists()]
    for e in ext:
        e.loc[e.model == 'devnet', 'model'] = 'devnet_official'
    a = pd.concat([d] + ext)
    order = ['dra', 'devnet_official', 'devnet', 'patchcore', 'seg', 'seg_aug']
    lines += ['## 2·3단계: SegAD 규칙 (공식 정상만 분할 + 시험 불량 10장을 학습으로, seed 0–4 평균)', '',
              '모든 방법이 같은 seed에서 같은 10장으로 학습했다(공개 코드의 고르기와 같음). '
              '원래 시험 = 남은 불량 90 + 정상 약 100. 공정한 시험 = 같은 시험 불량으로 만든 결함 지운 사진·붙인 사진을 섞은 것.', '',
              '### 원래 시험: 사진 단위 점수(AUROC, 1이면 완벽)', '', '| 방법 | PCB1 | PCB2 | PCB3 | 평균 |', '|---|---|---|---|---|']
    for name, v in PUB_SEGAD.items():
        lines.append(f'| 공개: {name} | ' + ' | '.join(f(x) for x in v) + f' | {f(np.mean(v))} |')
    for m in order:
        g = a[a.model == m]
        if len(g):
            v = [g[g.pcb == p].image_auroc.mean() for p in PCBS]
            lines.append(f'| {NAME[m]} | ' + ' | '.join(f(x) for x in v) + f' | {f(np.mean(v))} |')
    lines += ['', '### 공정한 시험: 놓친 불량 (과검 5%, 낮을수록 좋음)과 결함을 보는지', '',
              '| 방법 | PCB1 | PCB2 | PCB3 | 평균 | 정상·불량 가르기 | 결함 지운 사진을 불량이라 함 | 붙인 결함 잡음 |',
              '|---|---|---|---|---|---|---|---|']
    for m in order:
        g = a[a.model == m]
        if len(g):
            v = [g[g.pcb == p].fair_fnr5.mean() for p in PCBS]
            lines.append(f'| {NAME[m]} | ' + ' | '.join(pct(x) for x in v) + f' | **{pct(np.mean(v))}** | {f(g.fair_auroc.mean())} | '
                         f'{pct(g.v_DR.mean())} | {pct(g.v_ND.mean())} |')
    lines += ['', '- "결함 지운 사진을 불량이라 함"·"붙인 결함 잡음"의 임계값은 시험 정상의 과검 1%.', '']


def stage2(lines):
    fs = [R / 'official' / n for n in ('2cls_fewshot.csv', '2cls_fewshot_ctg.csv')]
    fs = [p for p in fs if p.exists()]
    if not fs:
        return
    a = pd.concat([pd.read_csv(p) for p in fs])
    a['key'] = a.model + '|' + a.train
    lines += ['## 2단계: 공식 "불량 조금" 분할 (PCB마다 학습 불량 20장 + 정상 약 200장, 시험 불량 80 + 정상 약 800)', '',
              '가짜 불량은 공식 학습 불량 20장만으로 생성기를 새로 학습해 만들었다(종류당 40장, 핀 잔상 제거, 누수 검사 통과). '
              '위치 모델은 헤드 seed 3개 평균.', '',
              '| 방법 | 사진 (PCB 1/2/3) | 픽셀 (PCB 1/2/3) | 결함 덮기 AUPRO | 공정한 시험 놓친 불량 | 결함 지운 사진 불량 판정 | 붙인 결함 잡음 |',
              '|---|---|---|---|---|---|---|']
    for name, (col, v) in PUB_VISA.items():
        cells = ['', '', '']
        cells[0 if col == 'image_auroc' else 1] = ' / '.join(f(x) for x in v)
        lines.append(f'| 공개: {name} | ' + ' | '.join(cells) + ' | | | |')
    for key in dict.fromkeys(a.key):
        g = a[a.key == key]
        m, t = key.split('|')
        name = NAME.get(m, m) + ('' if t in ('none', '512:0:0.01') else ' ' + NAME.get(t, t))
        per = lambda c: ' / '.join(f(g[g.pcb == p][c].mean()) for p in PCBS)  # noqa: E731
        lines.append(f'| {name} | {per("image_auroc")} | {per("pixel_auroc")} | {per("aupro")} | **{pct(g.fair_fnr5.mean())}** | '
                     f'{pct(g.v_DR.mean())} | {pct(g.v_ND.mean())} |')
    lines += ['', '- VisA 논문의 10-shot은 학습에서 정상·불량을 10장씩만 뽑았다. 우리는 공식 학습 전부(불량 20 + 정상 약 200)를 써서 '
              '그대로 같은 조건은 아니다(우리가 학습 장수에서 유리).', '']


def main():
    lines = ['# 공개 수치 비교 (11/20): 1단계 재현 · 2단계 공식 분할 · 3단계 공개 방법을 공정한 시험에', '']
    stage1(lines)
    stage3(lines)
    stage2(lines)
    lines += ['## 해석 (11/20)', '',
              '- **1단계:** 우리 PatchCore(512)는 공개 재실행 수치와 같거나 높다 → 실험 장치에 문제 없음. 256으로 줄이면 사진 점수가 '
              '0.03–0.04 낮은데(픽셀 점수는 같음), 원 구현의 최근접 재가중을 뺀 차이로 보인다.',
              '- **3단계:** 공개 지도 학습 방법(DRA, DevNet)을 공개 코드 그대로 돌리면 원래 시험 점수는 논문 보고값과 맞는다'
              '(DRA 0.913 vs 0.871, DevNet 0.883 vs 0.858). 그러나 공정한 시험에서는 불량의 86–91%를 놓치고, 결함을 지운 사진의 '
              '42–48%를 불량이라 한다 → 이미지 라벨로 배우는 공개 방법도 촬영 환경 지름길에 빠진다. 우리 단순판 DevNet(86%)과 같다.',
              '- **같은 규칙에서 우리 위치 모델(+ 촬영 환경 섞기)은 공정한 시험 놓친 불량 17%로 가장 낮다.** 정상만 배우는 PatchCore는 '
              'PCB1에서만 낫고(6%) PCB2·3에서 39–47%로 무너진다(평균 31%). 원래 시험 점수(0.971)는 최신 방법(EfficientAD + SegAD 0.992)보다 낮다.',
              '- **2단계 공식 "불량 조금" 분할:** 위치 모델 원래 점수 0.989–0.999, 공정한 시험 놓친 불량 11%. 가짜 불량을 더하면 '
              'cTransGAN 8%(헤드 seed × PCB 9칸 중 6칸 개선), SD+LoRA 10%(3칸 개선, PCB3에서 손해). 생성기에 따라 이득이 다르다는 '
              '우리 분할 결과와 같은 모양이다. PatchCore 30%, 사진 전체 모델 81%.',
              '- 같은 조건이 아닌 비교: VisA 논문 10-shot 수치는 학습 장수가 우리보다 적고, SegAD 논문 수치는 seed 10개(우리 5개)다.', '']
    out = ROOT / 'AnomalyDetection/results/ppt/public_comparison_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
