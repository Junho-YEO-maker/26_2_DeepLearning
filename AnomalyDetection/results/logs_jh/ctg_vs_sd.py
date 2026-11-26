import numpy as np, pandas as pd
R=str(__import__('pathlib').Path(__file__).resolve().parents[1])  # AnomalyDetection/results
rng=np.random.default_rng(0)
def um(f,tr):
    d=pd.read_csv(f); d=d[(d.model=='seg_aug')&(d.train==tr)]; return d.groupby('pcb')[['fair_fnr5','fair_auroc']].mean()
print(' K  N   cTG_miss  SD_miss  SD-cTG [95%]          | cTG_AUROC SD_AUROC SD-cTG [95%]        | SD가 나은 단위')
for k in [1,2,4]:
    for n in [5,10,20,40]:
        a,b=[],[]
        for s in range(5):
            c=um(f'{R}/fair_ad_K{k}_seed{s}_grid_ctg.csv','+ctransgan_diverse'+('' if n==40 else f'_n{n}'))
            v=um(f'{R}/fair_ad_K{k}_seed{s}_grid_v3.csv','+sd_lora_v3'+('' if n==40 else f'_n{n}'))
            for p in ('pcb1','pcb2','pcb3'): a.append(c.loc[p].values); b.append(v.loc[p].values)
        a,b=np.array(a),np.array(b); d=b-a
        bs=np.array([d[rng.integers(0,len(d),len(d))].mean(0) for _ in range(2000)]); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
        print(f'{k:2d} {n:2d}   {a[:,0].mean()*100:6.1f}  {b[:,0].mean()*100:6.1f}  {d[:,0].mean()*100:+5.1f} [{lo[0]*100:+.1f},{hi[0]*100:+.1f}]   |  {a[:,1].mean():.3f}   {b[:,1].mean():.3f}  {d[:,1].mean():+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}] | {int((d[:,0]<0).sum())}/15')
