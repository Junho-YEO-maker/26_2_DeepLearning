"""Seeded mask copy/paste baseline with resumable output."""
import csv
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from src.data.access import a_rows, open_a_pair
from src.data.prepare import sha

FIELDS = ['filename', 'source_normal', 'condition', 'seed', 'source_anomaly',
          'source_mask', 'source_defect_type', 'original_width', 'original_height',
          'output_width', 'output_height', 'resize', 'sample_index',
          'generation_seconds', 'image_sha256', 'mask_sha256']


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def size_for(size, long_side):
    if not long_side:
        return size
    scale = min(1, long_side / max(size))
    return tuple(max(1, round(v * scale)) for v in size)


def paste(target, donor, mask, rng, rotation_degrees, feather):
    bbox = mask.getbbox()
    assert bbox, 'Anomaly disappeared on resize'
    patch = np.asarray(donor.crop(bbox), dtype=np.float32)
    support = np.asarray(mask.crop(bbox)) > 0
    angle = float(rng.uniform(*rotation_degrees))
    support_image = Image.fromarray(support.astype('uint8') * 255)
    # Rotate premultiplied RGB so background pixels do not enter the foreground.
    alpha = support_image.rotate(angle, Image.Resampling.BILINEAR, expand=True)
    fg = Image.fromarray((patch * support[..., None]).astype('uint8'))
    fg = fg.rotate(angle, Image.Resampling.BILINEAR, expand=True)
    binary = support_image.rotate(angle, Image.Resampling.NEAREST, expand=True)
    pw, ph = fg.size
    width, height = target.size
    fit = min(1, (width - 2) / pw, (height - 2) / ph)
    patch_size = (max(1, int(pw * fit)), max(1, int(ph * fit)))
    fg = fg.resize(patch_size, Image.Resampling.BILINEAR)
    alpha = alpha.resize(patch_size, Image.Resampling.BILINEAR)
    binary = binary.resize(patch_size, Image.Resampling.NEAREST)
    coverage = np.asarray(alpha, dtype=np.float32) / 255
    valid = (np.asarray(binary) > 0) & (coverage > 0)
    assert valid.any(), 'Empty transformed mask'
    rgb = np.clip(np.asarray(fg, dtype=np.float32) / np.maximum(coverage[..., None], 1e-6), 0, 255)
    distance = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]
    blend = (np.minimum(1, distance / max(feather, 1e-6)) * valid)[..., None]
    patch_width, patch_height = patch_size
    x = int(rng.integers(width - patch_width + 1))
    y = int(rng.integers(height - patch_height + 1))
    source = np.asarray(target)
    result = source.copy()
    roi = np.s_[y:y+patch_height, x:x+patch_width]
    result[roi] = np.rint(source[roi] * (1 - blend) + rgb * blend).clip(0, 255).astype('uint8')
    output_mask = np.zeros((height, width), dtype='uint8')
    output_mask[roi] = valid.astype('uint8') * 255
    assert np.array_equal(result[output_mask == 0], source[output_mask == 0])
    assert np.any(result != source), 'No effective anomaly pasted'
    ys, xs = np.where(output_mask > 0)
    area = len(ys) / (height * width)
    condition = {
        'rotation_degrees': angle, 'paste_xy': [x, y], 'patch_scale_to_fit': fit,
        'area_fraction': area,
        'grid_3x3': [min(2, int(ys.mean()*3/height)), min(2, int(xs.mean()*3/width))],
        'connected_components': int(label(valid)[1]),
        'size_bin': 'small' if area < .01 else ('medium' if area < .05 else 'large'),
        'size_bin_thresholds': [.01, .05],
    }
    return Image.fromarray(result), Image.fromarray(output_mask), condition


class CopyPaste:
    name = 'copy_paste'

    def __init__(self, data_root, splits='splits', output_root='outputs',
                 long_side=512, rotation_degrees=(-180, 180), feather_pixels=2.0):
        self.root = Path(data_root)
        self.splits = Path(splits)
        self.output_root = Path(output_root)
        self.long_side = long_side
        self.rotation_degrees = list(rotation_degrees)
        self.feather = float(feather_pixels)

    def generate(self, pcb, n=500, seed=7017):
        assert seed == 7017 and n > 0
        normal = a_rows(self.splits, pcb, 'generate_condition', 'normal')
        donors = a_rows(self.splits, pcb, 'generate_condition', 'anomaly')
        out = self.output_root / self.name / pcb
        for folder in ('images', 'masks', 'records'):
            (out / folder).mkdir(parents=True, exist_ok=True)
        settings = {
            'model': self.name, 'pcb': pcb, 'seed': seed, 'long_side': self.long_side,
            'rotation_degrees': self.rotation_degrees, 'feather_pixels': self.feather,
            'split_sha256': sha(self.splits / f'{pcb}_split.csv'),
            'implementation_sha256': sha(Path(__file__)),
        }
        run_file = out / 'run.json'
        if run_file.exists():
            assert json.loads(run_file.read_text()) == settings, 'Use a new output directory for changed settings/code'
        else:
            atomic_json(run_file, settings)
        assert all(int(p.stem) < n for p in (out/'records').glob('*.json')), 'Refusing to shrink completed output'

        @lru_cache(maxsize=96)
        def resized(is_donor, index):
            row = (donors if is_donor else normal)[index]
            image, mask = open_a_pair(self.root, self.splits, pcb, row, 'generate_condition')
            original = image.size
            target = size_for(original, self.long_side)
            return (image.resize(target, Image.Resampling.LANCZOS),
                    mask.resize(target, Image.Resampling.NEAREST), original)

        rows = []
        for index in range(n):
            filename = f'{index:06d}.png'
            image_path = out / 'images' / filename
            mask_path = out / 'masks' / filename
            record = out / 'records' / f'{index:06d}.json'
            if record.exists():
                row = json.loads(record.read_text())
                assert sha(image_path) == row['image_sha256'] and sha(mask_path) == row['mask_sha256']
                rows.append(row)
                continue
            start = time.perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(pcb[-1]), index]))
            ni = int(rng.integers(len(normal)))
            di = int(rng.integers(len(donors)))
            target, _, original = resized(False, ni)
            donor, mask, _ = resized(True, di)
            image, output_mask, condition = paste(target, donor, mask, rng, self.rotation_degrees, self.feather)
            dr = donors[di]
            condition['defect_types'] = dr['defect_type'].split('|')
            for path, item in [(image_path, image), (mask_path, output_mask)]:
                tmp = path.with_suffix('.tmp')
                item.save(tmp, format='PNG')
                tmp.replace(path)
            row = dict(
                filename=filename, source_normal=normal[ni]['file_path'],
                condition=json.dumps(condition, sort_keys=True), seed=seed,
                source_anomaly=dr['file_path'], source_mask=dr['mask_path'],
                source_defect_type=dr['defect_type'], original_width=original[0],
                original_height=original[1], output_width=image.width, output_height=image.height,
                resize=json.dumps({'long_side': self.long_side, 'image': 'LANCZOS',
                                   'mask': 'NEAREST', 'crop': False, 'pad': False}),
                sample_index=index, image_sha256=sha(image_path), mask_sha256=sha(mask_path),
                generation_seconds=time.perf_counter()-start,
            )
            atomic_json(record, row)
            rows.append(row)
        tmp_meta = out / 'meta.csv.tmp'
        with tmp_meta.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_meta.replace(out / 'meta.csv')
        seconds = sum(row['generation_seconds'] for row in rows)
        atomic_json(out / 'cost.json', {'training_status': 'N/A: nonlearned baseline',
                    'generation_seconds_sum': seconds, 'seconds_per_image': seconds / n,
                    'n': n, 'device': 'CPU', 'gpu_peak_memory_bytes': None})
        return out
