import json
import numpy as np
from scipy.linalg import sqrtm

from tests.smoke_stage1 import run, must_fail
from src.models.copy_paste import CopyPaste
from src.metrics.evaluate import evaluate, fid_lowrank, lpips_cached_distances, validate_outputs
from src.data.prepare import read_csv


def exercise(root, splits):
    model = CopyPaste(root, splits, root.parent/'outputs', long_side=32)
    out = model.generate('pcb1', 6, 7017)
    old = read_csv(out/'meta.csv')
    model.generate('pcb1', 8, 7017)
    rows = read_csv(out/'meta.csv')
    assert old == rows[:6]
    other = CopyPaste(root, splits, root.parent/'fresh', long_side=32).generate('pcb1', 8, 7017)
    for row, fresh in zip(rows, read_csv(other/'meta.csv')):
        for key in ('image_sha256', 'mask_sha256', 'condition'):
            assert row[key] == fresh[key]
    config = dict(device='cpu', cpu_threads=2, inception_size=299, lpips_size=256,
                  batch_size=4, kid_subset_size=3, kid_subsets=5, lpips_pairs=8)
    report = evaluate(root, splits, out, 'pcb1', config, 8,
                      root.parent/'results', root.parent/'figures')
    assert all(np.isfinite(report[key]) for key in ('kid_mean','fid','lpips_mean'))
    path = out/'images'/'000000.png'
    original = path.read_bytes()
    path.write_bytes(original + b'corrupt')
    must_fail(lambda: validate_outputs(splits, out, 'pcb1', 8))
    path.write_bytes(original)


if __name__ == '__main__':
    import torch
    import lpips
    rng = np.random.default_rng(7017)
    x, y = rng.normal(size=(20,7)), rng.normal(size=(15,7))
    a, b = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    reference = np.square(x.mean(0)-y.mean(0)).sum() + np.trace(a+b-2*sqrtm(a@b).real)
    assert abs(fid_lowrank(x,y)-reference) < 1e-8
    torch.manual_seed(7017)
    torch.set_num_threads(2)
    model = lpips.LPIPS(net='alex', version='0.1').eval()
    images = torch.rand(4,3,64,64)*2-1
    pairs = [(0,1),(0,2),(2,3)]
    with torch.inference_mode():
        standard = model(images[[0,0,2]],images[[1,2,3]]).flatten().numpy()
    np.testing.assert_allclose(lpips_cached_distances(model, list(images), pairs, 2, 'cpu'), standard, atol=1e-6)
    run(exercise)
    print(json.dumps({'status':'PASS','tests':['heldout/split guards','resume and deterministic output',
          'corrupt output rejected','dummy end-to-end including pretrained metrics and figures',
          'FID equals sqrtm formulation','cached LPIPS equals official forward']},indent=2))
