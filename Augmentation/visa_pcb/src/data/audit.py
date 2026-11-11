import argparse,json
from pathlib import Path
from .prepare import PCBS,validate
from .access import checked_manifest,a_rows,open_a_pair

def audit(root,splits):
    all_rows=[]; counts={}
    for pcb in PCBS:
        rows=checked_manifest(splits,pcb);validate(rows,root);all_rows.extend(rows)
        refs=a_rows(splits,pcb,'quality_reference',label='anomaly')
        for r in refs:
            im,m=open_a_pair(root,splits,pcb,r,'quality_reference')
            assert im.size==m.size==(int(r['width']),int(r['height']))
        counts[pcb]=len(refs)
    validate(all_rows)
    return {'status':'PASS','verified_images':len(all_rows),'quality_reference_counts':counts,'checks':['frozen CSV SHA256','all file and mask hashes','cross-PCB and cross-split byte and pixel hashes','all quality references load through gen_train-only gate']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--splits',default='splits');a=p.parse_args();print(json.dumps(audit(a.root,a.splits),indent=2))
