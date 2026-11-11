"""Single-class, connected-component crops drawn only from frozen gen_train."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import find_objects, label

from src.data.access import a_rows, open_a_pair
from src.data.prepare import safe_path, sha


class PatchDataset:
    def __init__(self, root, splits, pcb, patch_size=32, context_pixels=2):
        self.root, self.splits, self.pcb = Path(root), Path(splits), pcb
        self.patch_size = patch_size
        self.rows = a_rows(splits, pcb, 'train', 'anomaly')
        self.by_path = {row['file_path']: row for row in self.rows}
        names = runpy.run_path(str(Path(__file__).resolve().parents[2]/'upstream/id2class.py'))['id2class_map'][pcb]
        patches, records = [], []
        for row in self.rows:
            image, _ = open_a_pair(root, splits, pcb, row, 'train')
            mask_path = safe_path(root, row['mask_path'])
            assert sha(mask_path) == row['mask_sha256']
            with Image.open(mask_path) as mask:
                classes = np.asarray(mask).copy()
            for class_id in sorted(int(v) for v in np.unique(classes) if v):
                components, _ = label(classes == class_id)
                for component_id, bounds in enumerate(find_objects(components), start=1):
                    if bounds is None:
                        continue
                    ys, xs = bounds
                    box = [max(0, xs.start-context_pixels), max(0, ys.start-context_pixels),
                           min(image.width, xs.stop+context_pixels), min(image.height, ys.stop+context_pixels)]
                    patch = image.crop(box).resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                    pixels = torch.from_numpy(np.asarray(patch).copy()).permute(2,0,1).float()/127.5-1
                    patches.append(pixels)
                    records.append({'source_anomaly': row['file_path'], 'source_mask': row['mask_path'],
                                    'class_id': class_id, 'class_name': names[str(class_id)],
                                    'component_id': component_id, 'box': box})
        assert records, 'No annotated anomaly components in gen_train'
        self.records = records
        self.class_ids = sorted({record['class_id'] for record in records})
        self.class_names = [names[str(key)] for key in self.class_ids]
        self.images = torch.stack(patches)
        self.targets = torch.tensor([self.class_ids.index(r['class_id']) for r in records])
        payload = {'split': sha(Path(splits)/f'{pcb}_split.csv'), 'patch_size': patch_size,
                   'context_pixels': context_pixels, 'records': records,
                   'sources': [(r['file_path'], r['image_sha256'], r['mask_sha256']) for r in self.rows]}
        self.fingerprint = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def batch(self, count, device):
        indices = torch.randint(len(self.records), (count,))
        return self.images[indices].to(device), self.targets[indices].to(device)

    def component_mask(self, record):
        assert record in self.records, 'Unapproved mask provenance'
        row = self.by_path[record['source_anomaly']]
        path = safe_path(self.root, row['mask_path'])
        assert sha(path) == row['mask_sha256']
        with Image.open(path) as image:
            classes = np.asarray(image)
        components, _ = label(classes == record['class_id'])
        selected = Image.fromarray((components == record['component_id']).astype('uint8')*255)
        return selected.crop(record['box'])

    def describe(self):
        return {'pcb': self.pcb, 'real_anomaly_images': len(self.rows), 'patches': len(self.records),
                'class_ids': self.class_ids, 'class_names': self.class_names,
                'patch_counts_by_class': {name: int((self.targets==i).sum()) for i,name in enumerate(self.class_names)},
                'fingerprint': self.fingerprint}
