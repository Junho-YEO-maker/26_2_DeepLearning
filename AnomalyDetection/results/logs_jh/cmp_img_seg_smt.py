"""SMT: 사진 단위 모델(DevNet · MLP, smt/img_ad.py)과 위치 모델(seg, smt/seg_ad.py 'none')을 같은 (공정, K, 나누기)끼리 비교한다.
지표: 시험셋 놓친 불량(FNR@과검 1%), AUROC, 지름길 확인 D · DR · ND (seg_ad.py 머리말).
  DR = 결함을 지운 불량 사진을 불량이라 한 비율 (낮을수록 결함을 본다), ND = 정상에 결함을 붙인 것을 잡은 비율 (높을수록 좋다).
K=32는 학습 후보 32장을 다 써서 확인용 사진이 없다(D · DR · ND 빈칸).

    python AnomalyDetection/results/logs_jh/cmp_img_seg_smt.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / 'smt'
pd.set_option('display.width', 200)
rng = np.random.default_rng(0)
for cat in ('pre', 'solder'):
    img = pd.read_csv(R / f'img_ad_{cat}.csv')
    seg = pd.read_csv(R / f'seg_ad_{cat}.csv')
    seg = seg[(seg.cond == 'none') & (seg.steps == 2000)].groupby(['K', 'seed'])[['fnr', 'auroc', 'D', 'DR', 'ND']].mean()
    seg['model'] = 'seg (위치)'
    allm = pd.concat([seg.reset_index(), img[['K', 'seed', 'model', 'fnr', 'auroc', 'D', 'DR', 'ND']]])
    keys = set(map(tuple, seg.reset_index()[['K', 'seed']].values)) & set(map(tuple, img[['K', 'seed']].values))
    allm = allm[[(k, s) in keys for k, s in zip(allm.K, allm.seed)]]
    t = allm.groupby(['model', 'K'])[['fnr', 'auroc', 'D', 'DR', 'ND']].mean().unstack('model')
    n = allm.groupby('K').seed.nunique()
    print(f'\n######## {cat}  (나누기 수: {dict(n)})')
    for col, name in (('fnr', '놓친 불량 %'), ('DR', '결함 지운 사진을 불량이라 함 % (낮을수록 결함을 봄)'), ('ND', '정상 + 결함을 잡음 %'), ('auroc', 'AUROC')):
        x = t[col] * (1 if col == 'auroc' else 100)
        print(f'-- {name}')
        print(x.round(3 if col == 'auroc' else 1).to_string())
    # 짝 비교: DevNet − seg (같은 K, 나누기)
    p = img[img.model == 'devnet'].set_index(['K', 'seed'])[['fnr', 'DR', 'ND']].join(seg[['fnr', 'DR', 'ND']], rsuffix='_seg', how='inner')
    print('-- DevNet − seg (같은 K · 나누기 짝, %p, 괄호 = DevNet이 큰 나누기 수)')
    for k, g in p.groupby(level='K'):
        parts = []
        for col in ('fnr', 'DR', 'ND'):
            dlt = (g[col] - g[col + '_seg']).dropna() * 100
            if len(dlt):
                parts.append(f'{col} {dlt.mean():+.1f} ({int((dlt > 0).sum())}/{len(dlt)})')
        print(f'   K={k}: ' + ', '.join(parts))
