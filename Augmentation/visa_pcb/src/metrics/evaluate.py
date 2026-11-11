"""Evaluate synthetic images with the frozen gen_train reference split."""
import argparse
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.data.access import a_rows, open_a_pair
from src.data.prepare import PCBS, read_csv, sha
from src.models.copy_paste import atomic_json


def fid_lowrank(real, fake):
    real, fake = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    assert min(len(real), len(fake)) > 1
    a, b = real - real.mean(0), fake - fake.mean(0)
    a /= np.sqrt(len(real) - 1)
    b /= np.sqrt(len(fake) - 1)
    # Nuclear norm equals the covariance cross-root trace, including singular cases.
    cross_trace = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    value = np.square(real.mean(0) - fake.mean(0)).sum() + np.square(a).sum() + np.square(b).sum() - 2 * cross_trace
    assert value > -1e-6
    return max(0.0, float(value))


def kid_repeated(real, fake, subset_size=25, repeats=100, seed=7017):
    real, fake = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    n = min(subset_size, len(real), len(fake))
    assert n >= 2 and repeats >= 2
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        a = real[rng.choice(len(real), n, replace=False)]
        b = fake[rng.choice(len(fake), n, replace=False)]
        aa = (1 + a @ a.T / a.shape[1]) ** 3
        bb = (1 + b @ b.T / b.shape[1]) ** 3
        ab = (1 + a @ b.T / a.shape[1]) ** 3
        values.append(float((aa.sum() - aa.trace() + bb.sum() - bb.trace()) / (n * (n-1)) - 2 * ab.mean()))
    return {'kid_mean': float(np.mean(values)), 'kid_std': float(np.std(values, ddof=1)),
            'kid_subset_size': n, 'kid_subsets': repeats, 'kid_replicates': values}


def tensor_image(image, size, normalize=False):
    import torch
    square = ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(square).copy()).permute(2, 0, 1)
    return tensor.float() / 127.5 - 1 if normalize else tensor


def inception_features(model, items, batch_size, device):
    import torch
    with torch.inference_mode():
        return np.concatenate([
            model(torch.stack(items[start:start+batch_size]).to(device))[0].cpu().numpy()
            for start in range(0, len(items), batch_size)
        ])


def lpips_cached_distances(model, images, pairs, batch_size, device):
    """Evaluate LPIPS v0.1 with each image's normalized activations cached on CPU."""
    import lpips
    import torch
    maps = None
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(images[start:start+batch_size]).to(device)
            outputs = model.net(model.scaling_layer(batch))
            if maps is None:
                maps = [torch.empty((len(images), *f.shape[1:]), dtype=f.dtype) for f in outputs]
            for cache, layer in zip(maps, outputs):
                cache[start:start+len(batch)] = lpips.normalize_tensor(layer).cpu()
        values = []
        for start in range(0, len(pairs), batch_size):
            left, right = zip(*pairs[start:start+batch_size])
            distance = None
            for cache, linear in zip(maps, model.lins):
                difference = (cache[list(left)] - cache[list(right)]).square().to(device)
                score = linear(difference).mean(dim=(2, 3), keepdim=True)
                distance = score if distance is None else distance + score
            values.extend(distance.flatten().cpu().tolist())
    return values


def validate_outputs(splits, out, pcb, expected_n):
    rows = read_csv(out/'meta.csv')
    names = [row['filename'] for row in rows]
    assert len(names) == len(set(names)) == expected_n
    for folder in ('images', 'masks'):
        assert set(names) == {path.name for path in (out/folder).glob('*.png')}
    normals = {row['file_path']: row for row in a_rows(splits, pcb, 'generate_condition', 'normal')}
    donors = {row['file_path']: row for row in a_rows(splits, pcb, 'generate_condition', 'anomaly')}
    for row in rows:
        assert Path(row['filename']).name == row['filename'] and row['seed'] == '7017'
        assert row['source_normal'] in normals and row['source_anomaly'] in donors, 'Unapproved source'
        assert row['source_mask'] == donors[row['source_anomaly']]['mask_path']
        assert sha(out/'images'/row['filename']) == row['image_sha256']
        assert sha(out/'masks'/row['filename']) == row['mask_sha256']
    return rows, normals


def write_figures(pcb, rows, references, out, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    for closeup in (False, True):
        columns = 4 if closeup else 3
        canvas = Image.new('RGB', (columns*300, 4*255), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, ((real, real_mask), row) in enumerate(zip(references[:4], rows[:4])):
            with Image.open(out/'images'/row['filename']) as gen, Image.open(out/'masks'/row['filename']) as mask:
                if closeup:
                    def crop(im, m):
                        left, top, right, bottom = m.getbbox()
                        pad = max(12, int(max(right-left, bottom-top) * .4))
                        return im.crop((max(0, left-pad), max(0, top-pad), min(im.width, right+pad), min(im.height, bottom+pad)))
                    items = [(real, 'Real (gen_train)'), (crop(real, real_mask), 'Real defect crop'),
                             (gen, out.parent.name), (crop(gen, mask), 'Generated defect crop')]
                else:
                    items = [(real, 'Real (gen_train)'), (gen, out.parent.name), (mask, 'Output mask')]
                for col, (image, title) in enumerate(items):
                    tile = ImageOps.pad(image.convert('RGB'), (292, 225),
                                        method=Image.Resampling.LANCZOS, color=(235,235,235))
                    canvas.paste(tile, (col*300+4, index*255+24))
                    draw.text((col*300+4, index*255+6), f'{pcb} / {title}', fill='black')
        suffix = 'closeups' if closeup else 'comparison'
        canvas.save(figures_dir/f'{pcb}_{suffix}.png')


def evaluate(data_root, splits, out, pcb, config, expected_n, results_dir='results', figures_dir='figures'):
    import lpips
    import torch
    from skimage.metrics import structural_similarity
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    start = time.perf_counter()
    torch.manual_seed(7017)
    torch.set_num_threads(config['cpu_threads'])
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    out, results_dir = Path(out), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows, normals = validate_outputs(splits, out, pcb, expected_n)
    refs = a_rows(splits, pcb, 'quality_reference', 'anomaly')
    references = [open_a_pair(data_root, splits, pcb, row, 'quality_reference') for row in refs]
    real = [tensor_image(image, config['inception_size']) for image, _ in references]
    fake, diversity = [], []
    psnr, ssim = [], []

    @lru_cache(maxsize=96)
    def resized_normal(path, size):
        image, _ = open_a_pair(data_root, splits, pcb, normals[path], 'generate_condition')
        return np.asarray(image.resize(size, Image.Resampling.LANCZOS))

    for row in rows:
        with Image.open(out/'images'/row['filename']) as image, Image.open(out/'masks'/row['filename']) as mask:
            size = (int(row['output_width']), int(row['output_height']))
            assert image.mode == 'RGB' and mask.mode == 'L' and image.size == mask.size == size
            pixels, m = np.asarray(image), np.asarray(mask)
            assert set(np.unique(m)) == {0, 255}
            original = resized_normal(row['source_normal'], size)
            assert np.array_equal(original[m == 0], pixels[m == 0]), 'Pixels changed outside mask'
            mse = float(np.square(original.astype(np.float32) - pixels).mean(dtype=np.float64))
            assert mse > 0
            psnr.append(float(10 * np.log10(255**2 / mse)))
            ssim.append(float(structural_similarity(original, pixels, data_range=255, channel_axis=2)))
            fake.append(tensor_image(image, config['inception_size']))
            diversity.append(tensor_image(image, config['lpips_size'], normalize=True))

    batch = config['batch_size']
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).to(device).eval()
    x = inception_features(extractor, real, batch, device)
    y = inception_features(extractor, fake, batch, device)
    del extractor, real, fake
    kid = kid_repeated(x, y, config['kid_subset_size'], config['kid_subsets'])
    fid = fid_lowrank(x, y)
    candidates = list(itertools.combinations(range(len(rows)), 2))
    chosen = np.random.default_rng(7017).choice(len(candidates), min(config['lpips_pairs'], len(candidates)), replace=False)
    pairs = [candidates[i] for i in chosen]
    metric = lpips.LPIPS(net='alex', version='0.1').to(device).eval()
    lp = lpips_cached_distances(metric, diversity, pairs, batch, device)
    write_figures(pcb, rows, references, out, Path(figures_dir))
    report = {
        'model': out.parent.name, 'pcb': pcb, 'seed': 7017, 'n_generated': len(rows),
        'n_real_reference': len(refs), 'reference_usage': 'gen_train only',
        'reference_paths': [row['file_path'] for row in refs],
        'metric_device': device, 'settings': config, **kid, 'fid': fid,
        'lpips_mean': float(np.mean(lp)), 'lpips_pair_std': float(np.std(lp, ddof=1)),
        'lpips_pairs': len(pairs), 'lpips_pair_indices': pairs,
        'source_normal_psnr_mean': float(np.mean(psnr)), 'source_normal_ssim_mean': float(np.mean(ssim)),
        'outside_mask_pixel_changes': 0, 'validation_status': 'PASS',
        'evaluation_seconds': time.perf_counter() - start,
        'evaluation_gpu_peak_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'generation_cost': json.loads((out/'cost.json').read_text()),
        'notes': 'PSNR/SSIM pair with source normal and measure preservation. FID uses 33 training references in the real experiment.',
    }
    atomic_json(results_dir/f'{pcb}_metrics.json', report)
    return report


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage2.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--pcb', choices=PCBS)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for pcb in ([args.pcb] if args.pcb else PCBS):
        output = Path(cfg['output_root']) / cfg['model'] / pcb
        result = evaluate(args.data_root or cfg['data_root'], cfg['splits'], output,
                          pcb, cfg['metrics'], cfg['n_per_pcb'],
                          results_dir='results' if cfg['model']=='copy_paste' else f"results/{cfg['model']}",
                          figures_dir='figures' if cfg['model']=='copy_paste' else f"figures/{cfg['model']}")
        print(pcb, {key: result[key] for key in ('kid_mean','kid_std','fid','lpips_mean')}, flush=True)
