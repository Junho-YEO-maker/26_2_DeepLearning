"""VisA 배경 섞은 시험셋: 모델별 · 실제 장수별 비교 (놓친 불량 @과검 5%, AUROC).
같은 (나누기, PCB) 단위끼리만 짝지어 비교. PatchCore는 정상만 쓰므로 PCB마다 값 하나."""
import glob, numpy as np, pandas as pd
R = str(__import__('pathlib').Path(__file__).resolve().parents[1])  # AnomalyDetection/results
rng = np.random.default_rng(0)
rows = []
for f in glob.glob(f'{R}/fair_ad_K*_seed?.csv') + glob.glob(f'{R}/fair_ad_K*_seed?_grid_ctg.csv') + glob.glob(f'{R}/fair_ad_K*_seed?_ctg.csv'):
    k = int(f.split('_K')[1].split('_')[0]); s = int(f.split('_seed')[1][0])
    d = pd.read_csv(f); d['K'] = k; d['seed'] = s; d['src'] = f.split('fair_ad_')[1]
    rows.append(d)
d = pd.concat(rows)
# 같은 (모델, 학습, K, 나누기)가 여러 파일에 있으면 grid 파일 우선 (K1 나누기 0의 _ctg는 머리 시드가 3개라 뺀다)
d = d[~d.src.str.endswith('_ctg.csv') | d.src.str.contains('grid')]
d = d[d.train.isin(['none', '+copypaste', '+ctransgan_diverse'])]
u = d.groupby(['model', 'train', 'K', 'seed', 'pcb'])[['fair_fnr5', 'fair_auroc']].mean().reset_index()
pc = u[u.model == 'patchcore'].groupby('pcb')[['fair_fnr5', 'fair_auroc']].mean()
print('PatchCore (정상만, PCB별):'); print(pc.round(3).to_string()); print('평균', pc.mean().round(3).to_dict(), '\n')
ext = pd.concat([pd.read_csv(f'{R}/external/{m}.csv') for m in ('dra', 'devnet')])
print('공개 코드 (불량 10장 규칙, 나누기 5):'); print(ext.groupby('method')[['fair_fnr5', 'fair_auroc', 'image_auroc']].mean().round(3).to_string(), '\n')
tab = u[u.model != 'patchcore'].groupby(['model', 'train', 'K']).agg(n=('seed', 'nunique'), miss=('fair_fnr5', 'mean'), auroc=('fair_auroc', 'mean')).reset_index()
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
for met in ('miss', 'auroc'):
    t = tab.pivot_table(index=['model', 'train'], columns='K', values=met)
    print(f'== {met}'); print((t * 100).round(1).to_string()); print()
print('== 나누기 수'); print(tab.pivot_table(index=['model', 'train'], columns='K', values='n').to_string()); print()
# PatchCore 대비 짝 비교 (위치 모델 + 촬영 조건 섞기, 실제만 / + cTransGAN 40)
print('== PatchCore 대비 차이 (놓친 불량 %p, AUROC ×100), 95% 범위는 (나누기, PCB) 복원 추출')
for tr in ('none', '+ctransgan_diverse', '+copypaste'):
    for k in sorted(u.K.unique()):
        x = u[(u.model == 'seg_aug') & (u.train == tr) & (u.K == k)]
        if x.seed.nunique() < 5: continue
        for met in ('fair_fnr5', 'fair_auroc'):
            dd = (x[met].values - pc.loc[x.pcb, met].values) * 100
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'  seg_aug {tr:20s} K{k:<2d} {met:10s} {dd.mean():+6.1f}  [{lo:+.1f}, {hi:+.1f}]  PatchCore보다 나은 단위 {int(((dd < 0) if met == "fair_fnr5" else (dd > 0)).sum())}/{len(dd)}')

print('\n== PCB별 놓친 불량 % (위치 모델 + 촬영 조건 섞기, 나누기 5 평균) vs PatchCore')
x = u[(u.model == 'seg_aug') & (u.train.isin(['none', '+ctransgan_diverse', '+copypaste'])) & (u.K.isin([1, 2, 4, 8, 10]))]
t = x.groupby(['train', 'K', 'pcb']).fair_fnr5.mean().unstack('pcb') * 100
t.loc[('PatchCore', 0), :] = pc.fair_fnr5 * 100
print(t.round(1).to_string())
print('\n== cTransGAN 40 vs 단순 복사·붙여넣기 (같은 단위 짝 비교)')
for k in (1, 2, 4):
    a = u[(u.model == 'seg_aug') & (u.train == '+ctransgan_diverse') & (u.K == k)].set_index(['seed', 'pcb'])
    b = u[(u.model == 'seg_aug') & (u.train == '+copypaste') & (u.K == k)].set_index(['seed', 'pcb'])
    c = u[(u.model == 'seg_aug') & (u.train == 'none') & (u.K == k)].set_index(['seed', 'pcb'])
    j = a.index.intersection(b.index)
    for met in ('fair_fnr5', 'fair_auroc'):
        for nm, o in (('cTG-복붙', b), ('복붙-실제만', c)):
            dd = ((a if nm.startswith('cTG') else b).loc[j, met] - o.loc[j, met]).values * 100
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'  K{k} {met:10s} {nm:8s} {dd.mean():+5.1f} [{lo:+.1f}, {hi:+.1f}]  n={len(dd)}')
