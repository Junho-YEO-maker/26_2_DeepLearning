import glob, sys, numpy as np, pandas as pd
R=str(__import__('pathlib').Path(__file__).resolve().parents[1])  # AnomalyDetection/results
GEN = sys.argv[1] if len(sys.argv) > 1 else 'ctg'
BASE = {'ctg': '+ctransgan_diverse', 'v3': '+sd_lora_v3'}[GEN]
M = ['fair_auroc','fair_fnr5','fair_fnr','fair_miss_D','fair_miss_ND','fair_fa_A','fair_fa_DR','fnr5','fnr']
rng = np.random.default_rng(0)
def um(f, tr):
    d = pd.read_csv(f); d = d[(d.model=='seg_aug') & (d.train==tr)]
    return d.groupby('pcb')[M].mean()
real = {}
for k in [1,2,3,4,5,6,8,10]:
    for s in range(5):
        for p, r in um(f'{R}/fair_ad_K{k}_seed{s}.csv','none').iterrows(): real[(k,s,p)] = r
print('실제만, 15단위 평균')
print(pd.DataFrame({k: pd.concat([real[(k,s,p)] for s in range(5) for p in ('pcb1','pcb2','pcb3')],axis=1).mean(axis=1) for k in [1,2,3,4,5,6,8,10]}).T.round(3).to_string())
rows=[]
for k in [1,2,4]:
    for n in [5,10,20,40]:
        tr = BASE if n==40 else f'{BASE}_n{n}'
        a, b = [], []
        for s in range(5):
            for p, r in um(f'{R}/fair_ad_K{k}_seed{s}_grid_{GEN}.csv', tr).iterrows():
                a.append(r.values); b.append(real[(k,s,p)].values)
        a, b = np.array(a), np.array(b); d = a-b
        bs = np.array([d[rng.integers(0,len(d),len(d))].mean(0) for _ in range(2000)])
        lo, hi = np.percentile(bs,[2.5,97.5],axis=0)
        for j,m in enumerate(M):
            rows.append(dict(K=k,N=n,metric=m,real=b[:,j].mean(),aug=a[:,j].mean(),diff=d[:,j].mean(),lo=lo[j],hi=hi[j],better_units=int(((d[:,j]>0) if m=='fair_auroc' else (d[:,j]<0)).sum()),units=len(d)))
o=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
for m in M:
    print('\n==',m); print(o[o.metric==m].drop(columns='metric').round(4).to_string(index=False))
