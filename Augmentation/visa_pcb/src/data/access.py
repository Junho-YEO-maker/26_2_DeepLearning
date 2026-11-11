"""Only use these accessors for A-team training, tuning, and quality references."""
import json
from pathlib import Path
from .prepare import read_csv, sha, safe_path, validate

ALLOWED = {'train','tune','quality_reference','generate_condition'}

def checked_manifest(split_dir, pcb):
    assert pcb in ('pcb1','pcb2','pcb3','pcb4')
    split_dir=Path(split_dir); manifest=json.loads((split_dir/'manifest.json').read_text())
    assert manifest['seed']==7017
    path=split_dir/f'{pcb}_split.csv'
    assert sha(path)==manifest['split_sha256'][path.name], 'Split tampering detected'
    rows=read_csv(path); validate(rows); return rows

def a_rows(split_dir, pcb, purpose, label=None):
    assert purpose in ALLOWED, 'A-team purpose is not allowed'
    rows=checked_manifest(split_dir,pcb)
    chosen=[r for r in rows if r['usage']=='gen_train' and (label is None or r['label']==label)]
    assert chosen and all(r['usage']=='gen_train' for r in chosen)
    heldout={r['file_path'] for r in rows if r['usage']=='heldout'}
    assert not heldout & {r['file_path'] for r in chosen}, 'Heldout leak'
    if purpose=='quality_reference': assert all(r['label']=='anomaly' for r in chosen), 'Quality reference must be gen_train anomaly'
    return chosen

def open_a_pair(root, split_dir, pcb, row, purpose):
    from PIL import Image
    import numpy as np
    # Caller cannot bypass the gate by changing usage or supplying arbitrary paths.
    approved={r['file_path']:r for r in a_rows(split_dir,pcb,purpose,label='anomaly' if purpose=='quality_reference' else None)}
    assert row['file_path'] in approved and row==approved[row['file_path']], 'Unapproved or forged A-team input'
    image_path=safe_path(root,row['file_path']); assert sha(image_path)==row['image_sha256']
    with Image.open(image_path) as im: image=im.convert('RGB').copy()
    if row['mask_path']:
        mp=safe_path(root,row['mask_path']); assert sha(mp)==row['mask_sha256']
        with Image.open(mp) as m: mask=Image.fromarray((np.asarray(m)>0).astype('uint8')*255)
    else: mask=Image.new('L',image.size,0)
    return image,mask
