import csv,json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.data.prepare import prepare,read_csv,validate,PCBS
from src.data.access import a_rows,open_a_pair,checked_manifest

def must_fail(fn):
    try: fn()
    except AssertionError: return
    raise RuntimeError('Expected leakage guard to reject input')

def run(callback=None):
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/'data'; out=Path(td)/'splits'; rng=np.random.default_rng(7017)
        for pcb in PCBS:
            rows=[]
            for label,n in [('normal',10),('anomaly',12)]:
                for i in range(n):
                    ip=f'{pcb}/Data/Images/{label.title()}/{i:04d}.png'; p=root/ip;p.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rng.integers(0,256,(24,32,3),dtype=np.uint8)).save(p)
                    mp=''
                    if label=='anomaly':
                        mp=f'{pcb}/Data/Masks/Anomaly/{i:04d}.png'; p=root/mp;p.parent.mkdir(parents=True,exist_ok=True)
                        a=np.zeros((24,32),dtype=np.uint8); a[3:8,4:9]=i%3+1
                        if i>=9:a[10:12,10:12]=(i+1)%3+1
                        Image.fromarray(a).save(p)
                    source_label=label
                    if label=='anomaly':
                        names={1:'bent',2:'melt',3:'scratch'} if pcb!='pcb4' else {1:'burnt',2:'scratch',3:'missing'}
                        source_label=','.join(names[int(v)] for v in np.unique(a) if v)
                    rows.append({'image':ip,'label':source_label,'mask':mp})
            with (root/pcb/'image_anno.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['image','label','mask']);w.writeheader();w.writerows(rows)
        report=prepare(root,out,'upstream/id2class.py'); prepare(root,out,'upstream/id2class.py')
        for pcb in PCBS:
            rows=checked_manifest(out,pcb); validate(rows,root)
            for purpose in ('train','tune','generate_condition','quality_reference'):
                chosen=a_rows(out,pcb,purpose,label='anomaly' if purpose=='quality_reference' else None)
                im,m=open_a_pair(root,out,pcb,chosen[0],purpose);assert im.size==m.size==(32,24)
            held=next(r for r in rows if r['usage']=='heldout' and r['label']=='anomaly')
            for purpose in ('train','tune','quality_reference','generate_condition'):
                must_fail(lambda:open_a_pair(root,out,pcb,held,purpose))
            fake=dict(held,usage='gen_train');must_fail(lambda:open_a_pair(root,out,pcb,fake,'train'))
            changed=[dict(r) for r in rows];tr=next(r for r in changed if r['usage']=='gen_train');he=next(r for r in changed if r['usage']=='heldout');he['pixel_sha256']=tr['pixel_sha256'];must_fail(lambda:validate(changed))
            p=out/f'{pcb}_split.csv';original=p.read_bytes();p.write_bytes(original+b'\n');must_fail(lambda:checked_manifest(out,pcb));p.write_bytes(original)
        if callback is not None: callback(root,out)
        return {'status':'PASS','seed':7017,'tests':['dummy end-to-end prepare and guarded image/mask loading (4 PCBs)','byte-identical split and manifest on repeat','heldout blocked for all four A-team purposes','forged usage blocked','decoded-pixel duplicate leakage blocked','CSV tampering blocked','original aspect ratio retained']}
if __name__=='__main__':print(json.dumps(run(),indent=2))
