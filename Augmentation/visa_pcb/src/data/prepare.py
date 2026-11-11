"""Stage 1 only: inspect original VisA, stratify, freeze, and validate splits."""
import argparse, csv, hashlib, json, math, random, runpy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp

if not __debug__:
    raise RuntimeError('Do not run leakage checks with python -O or PYTHONOPTIMIZE')

PCBS = ('pcb1', 'pcb2', 'pcb3', 'pcb4')
FIELDS = ['file_path','label','usage','defect_type','mask_path','defect_ids','width','height','image_sha256','pixel_sha256','mask_sha256','source_label']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def safe_path(root, rel):
    p=(Path(root)/rel).resolve()
    assert p.is_relative_to(Path(root).resolve()), 'Path escapes dataset root'
    assert p.is_file(), f'Missing file: {p}'
    return p

def write_csv(path, rows, fields=FIELDS):
    import io
    s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=fields); w.writeheader(); w.writerows(rows)
    b=s.getvalue().encode('utf-8-sig'); path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes()==b, f'Frozen split differs: {path}; use a NEW experiment directory'
    else:
        tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(b); tmp.replace(path)

def stratified_select(rows, n, seed):
    """Integer allocation by complete multilabel signature, with per-class balance.
    Minimize sum absolute signature deviations + sum absolute class deviations.
    Classes with >=2 images must occur in both partitions. Singletons stay heldout.
    """
    groups=defaultdict(list)
    for r in rows: groups[r['defect_ids']].append(r)
    keys=sorted(groups); counts=np.array([len(groups[k]) for k in keys]); g=len(keys)
    classes=sorted({c for k in keys for c in k.split('|') if c})
    membership=np.array([[int(c in k.split('|')) for k in keys] for c in classes],dtype=float).reshape(len(classes),g)
    A=np.vstack([np.eye(g), membership]); target=A@counts*(n/len(rows)); m=len(target)
    # Variables: integer number selected per signature, followed by absolute deviations.
    c=np.r_[np.zeros(g), np.ones(m)]
    constraints=[LinearConstraint(np.r_[np.ones(g),np.zeros(m)][None,:],n,n),
                 LinearConstraint(np.c_[A,-np.eye(m)],-np.inf,target),
                 LinearConstraint(np.c_[-A,-np.eye(m)],-np.inf,-target)]
    totals=membership@counts
    if len(classes):
        lo=np.where(totals>=2,1,0); hi=np.where(totals>=2,totals-1,0)
        constraints.append(LinearConstraint(np.c_[membership,np.zeros((len(classes),m))],lo,hi))
    result=milp(c,integrality=np.r_[np.ones(g),np.zeros(m)],bounds=Bounds(np.zeros(g+m),np.r_[counts,np.full(m,np.inf)]),constraints=constraints,options={'time_limit':60})
    assert result.success, f'Stratification infeasible: {result.message}; inspect rare labels before proceeding'
    quotas=np.rint(result.x[:g]).astype(int); rng=random.Random(seed); selected=set()
    for k,q in zip(keys,quotas):
        group=sorted(groups[k],key=lambda x:x['file_path']); rng.shuffle(group)
        selected.update(r['file_path'] for r in group[:q])
    assert len(selected)==n
    detail=[{'defect_ids':k,'total':int(counts[i]),'gen_train':int(quotas[i]),'heldout':int(counts[i]-quotas[i])} for i,k in enumerate(keys)]
    return selected, detail

def validate(rows, root=None):
    paths=[r['file_path'] for r in rows]; assert len(paths)==len(set(paths)), 'Duplicate paths'
    assert all(r['label'] in ('normal','anomaly') and r['usage'] in ('gen_train','heldout') for r in rows)
    for key in ('image_sha256','pixel_sha256'):
        train={r[key] for r in rows if r['usage']=='gen_train'}
        held={r[key] for r in rows if r['usage']=='heldout'}
        assert not train & held, f'Content leakage: {key}'
    if root:
        for r in rows:
            assert sha(safe_path(root,r['file_path']))==r['image_sha256'], 'Image changed'
            if r['mask_path']: assert sha(safe_path(root,r['mask_path']))==r['mask_sha256'], 'Mask changed'
    return True

def prepare(root, output, mapping_path, seed=7017, normal_fraction=.8):
    assert seed==7017, 'Shared experiment fixes seed=7017'
    assert 0 < normal_fraction < 1
    root=Path(root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    mapping=runpy.run_path(str(mapping_path))['id2class_map']; report={'seed':seed,'normal_train_fraction':normal_fraction,'pcb':{}}
    all_rows=[]
    for pcb in PCBS:
        source=read_csv(root/pcb/'image_anno.csv'); rows=[]; modes=Counter(); values=Counter(); dimensions=Counter()
        assert set(source[0])=={'image','label','mask'}, f'Unexpected schema: {source[0].keys()}'
        for x in source:
            p=safe_path(root,x['image'])
            with Image.open(p) as im:
                im.load(); w,h=im.size; rgb=np.asarray(im.convert('RGB')); pixel=hashlib.sha256(f'{w},{h},RGB:'.encode()+rgb.tobytes()).hexdigest()
            dimensions[f'{w}x{h}']+=1; ids=[]; masksha=''
            label='normal' if x['label']=='normal' else 'anomaly'
            if label=='anomaly':
                mp=safe_path(root,x['mask']); masksha=sha(mp)
                with Image.open(mp) as mask:
                    mask.load(); a=np.asarray(mask); modes[mask.mode]+=1
                    assert mask.size==(w,h), f'Mask size mismatch: {mp}'
                    assert a.ndim==2, f'Unexpected mask channels: {mp}'
                    unique=np.unique(a).tolist(); values.update(unique); ids=[str(int(i)) for i in unique if i!=0]
                assert ids and set(ids)<=set(mapping[pcb]), f'Empty or unknown mask IDs: {mp}'
                names={mapping[pcb][i].lower() for i in ids}
                csv_names={name.strip().lower().replace('_',' ') for name in x['label'].split(',')}
                assert names==csv_names, f'CSV/mask class disagreement: {mp}: {csv_names} vs {names}'
            else: assert not x['mask'], 'Unexpected normal mask'
            rows.append(dict(file_path=x['image'],label=label,source_label=x['label'],usage='',defect_type='|'.join(mapping[pcb][i] for i in ids),mask_path=x['mask'],defect_ids='|'.join(ids),width=w,height=h,image_sha256=sha(p),pixel_sha256=pixel,mask_sha256=masksha))
        anomaly=[r for r in rows if r['label']=='anomaly']; normal=sorted([r for r in rows if r['label']=='normal'],key=lambda x:x['file_path'])
        chosen, strata=stratified_select(anomaly,len(anomaly)//3,seed)
        rng=random.Random(seed); rng.shuffle(normal); chosen.update(r['file_path'] for r in normal[:math.floor(len(normal)*normal_fraction)])
        for r in rows: r['usage']='gen_train' if r['file_path'] in chosen else 'heldout'
        rows.sort(key=lambda x:x['file_path']); validate(rows)
        # Exact annotation coverage: no silently ignored images or anomaly masks.
        actual_images={p.relative_to(root).as_posix() for p in (root/pcb/'Data/Images').rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png')}
        assert actual_images=={r['file_path'] for r in rows}, 'Image inventory mismatch'
        actual_masks={p.relative_to(root).as_posix() for p in (root/pcb/'Data/Masks').rglob('*') if p.is_file() and p.suffix.lower()=='.png'}
        assert actual_masks=={r['mask_path'] for r in rows if r['mask_path']}, 'Mask inventory mismatch'
        write_csv(output/f'{pcb}_split.csv',rows); all_rows.extend(rows)
        class_counts={mapping[pcb][i]:dict(Counter(r['usage'] for r in rows if i in r['defect_ids'].split('|'))) for i in mapping[pcb] if i!='0'}
        report['pcb'][pcb]={'source_columns':list(source[0]),'counts':dict(Counter(f"{r['label']}/{r['usage']}" for r in rows)), 'dimensions':dict(dimensions),'mask_modes':dict(modes),'mask_values_image_occurrences':dict(values),'multilabel_images':sum('|' in r['defect_ids'] for r in rows),'strata':strata,'class_counts':class_counts}
    validate(all_rows)
    report['split_sha256']={p.name:sha(p) for p in sorted(output.glob('*_split.csv'))}
    frozen=output/'manifest.json'; body=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if frozen.exists(): assert frozen.read_text()==body, 'Manifest changed; do not overwrite shared split'
    else: frozen.write_text(body)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',default='splits'); p.add_argument('--mapping',default='upstream/id2class.py'); p.add_argument('--normal-fraction',type=float,default=.8)
    p.add_argument('--config'); a=p.parse_args()
    seed=7017
    if a.config:
        import yaml
        config=yaml.safe_load(Path(a.config).read_text());seed=config['seed'];a.normal_fraction=config['normal_train_fraction']
        assert config['anomaly_rounding']=='floor' and abs(config['anomaly_train_fraction']-1/3)<1e-12
    print(json.dumps(prepare(a.root,a.output,a.mapping,seed=seed,normal_fraction=a.normal_fraction),ensure_ascii=False,indent=2))
