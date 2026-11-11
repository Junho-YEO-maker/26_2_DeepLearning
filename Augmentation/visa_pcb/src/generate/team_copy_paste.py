"""Generate B-team copy-paste data for one shared K/seed CSV."""
import argparse
import csv
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.data.team_split import DEFECT_TYPES, PCB_TYPES, load_team_split, raw_path
from src.models.copy_paste import paste

FIELDS = ('image', 'mask', 'label', 'source_image', 'score',
          'category', 'defect_type', 'base_normal_image')


def generate(repo_root, k, seed, per_class=40):
    repo_root = Path(repo_root).resolve()
    if per_class < 1:
        raise ValueError('per_class must be positive')
    normals, defects = load_team_split(repo_root, k, seed)
    relative = Path('Dataset/generated/copypaste') / f'K{k}_seed{seed}'
    output = repo_root / relative
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=16)
    def rgb(path):
        with Image.open(path) as image:
            return image.convert('RGB')

    with TemporaryDirectory(prefix='.copypaste-', dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        (staging / 'images').mkdir(parents=True)
        (staging / 'masks').mkdir()
        with (staging / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for pcb_number, pcb in enumerate(PCB_TYPES, start=1):
                for class_id, defect_type in DEFECT_TYPES.items():
                    donors = defects[pcb, class_id]
                    rng = np.random.default_rng(np.random.SeedSequence([7017, seed, pcb_number, class_id]))
                    donor_order = rng.permutation(len(donors))
                    for index in range(per_class):
                        donor = donors[donor_order[index % len(donors)]]
                        normal = normals[pcb][int(rng.integers(len(normals[pcb])))]
                        source = rgb(raw_path(repo_root, donor, 'image'))
                        target = rgb(raw_path(repo_root, normal, 'image'))
                        with Image.open(raw_path(repo_root, donor, 'mask')) as annotated:
                            classes = np.asarray(annotated)
                            if (annotated.size != source.size or classes.ndim != 2
                                    or set(np.unique(classes)) != {0, class_id}):
                                raise ValueError(f'Unexpected class-ID mask: {donor["mask"]}')
                            mask = Image.fromarray((classes == class_id).astype('uint8') * 255)
                        image, support, _ = paste(target, source, mask, rng, (-180, 180), 2.0)
                        class_mask = Image.fromarray((np.asarray(support) > 0).astype('uint8') * class_id)
                        name = f'{pcb}_{defect_type.casefold()}_{index:04d}.png'
                        image.save(staging / 'images' / name)
                        class_mask.save(staging / 'masks' / name)
                        writer.writerow(dict(image=(relative / 'images' / name).as_posix(),
                                             mask=(relative / 'masks' / name).as_posix(), label=1,
                                             source_image=donor['image'], score='', category=pcb,
                                             defect_type=defect_type, base_normal_image=normal['image']))
        staging.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--per-class', type=int, default=40)
    args = parser.parse_args()
    print(generate(args.repo_root, args.k, args.seed, args.per_class))
