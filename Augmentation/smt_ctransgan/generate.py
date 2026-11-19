"""SMT cTransGAN using the diffusion split and placement protocol."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from smt_sd_lora.smt import Split, OUT, RES, box_mask, load_rgb, paste
from smt_sd_lora.generate import leak_check, rel, save_like, sheet, change_score
from smt_sd_lora.generate_fix import choose_site, place_fixed
from smt_ctransgan.core import train, to_tensor, NOISE


def run(a):
    torch.set_num_threads(4)
    sp = Split(a.k, a.seed)
    directory = OUT / 'ctransgan' / f'K{a.k}_seed{a.seed}'
    if (directory / 'manifest.csv').exists():
        print(f'Already complete: {directory}', flush=True)
        return
    (directory / 'images').mkdir(parents=True, exist_ok=True)
    (directory / 'masks').mkdir(exist_ok=True)
    rows = []
    for cat in ('pre', 'solder'):
        rng = np.random.default_rng([7017, a.k, a.seed, 0 if cat == 'pre' else 1])
        torch.manual_seed(int(rng.integers(2**31)))
        normals = sp.normals(cat)
        types = sp.types(cat)
        crops, donors = [], []
        for label, typ in enumerate(types):
            for source, boxes in sp.defects(cat, typ):
                donor = load_rgb(source['image'])
                for box in boxes:
                    candidates = [(row, site) for row, sites in normals for t, site in sites if t == typ]
                    if not candidates:
                        raise ValueError(f'No normal site for {cat}/{typ}')
                    bases = []
                    for _ in range(3):
                        (row, site), _ = choose_site(candidates, typ, box, rng)
                        base = load_rgb(row['image']).crop(site)
                        bases.append(to_tensor(base, a.res))
                    crops.append(dict(real=to_tensor(donor.crop(box), a.res), normals=bases,
                                      masks=[torch.ones(1, a.res, a.res)] * 3, label=label))
                    donors.append((source, box, typ, label, candidates))
        print(f'{cat} K{a.k} seed{a.seed}: {len(crops)} training crops', flush=True)
        config = SimpleNamespace(res=a.res, classes=len(types), steps=a.steps, classifier_steps=60,
                                 batch=4, lr=5e-4, edge_weight=14, texture_weight=18)
        generator, embeddings, anchors = train(crops, config, torch.device('cuda'))
        checkpoint = directory / f'{cat}_model.pt'
        torch.save(dict(generator=generator.state_dict(), embeddings=embeddings.cpu(),
                        anchors=anchors.cpu(), types=types, config=vars(config)), checkpoint)
        for typ in types:
            group = [i for i, donor in enumerate(donors) if donor[2] == typ]
            for number in range(a.per_type):
                index = group[number % len(group)]
                source, source_box, _, label, candidates = donors[index]
                (base_row, site), cost = choose_site(candidates, typ, source_box, rng)
                target_box, rule = place_fixed(site, source_box, typ, rng)
                base = load_rgb(base_row['image'])
                with torch.inference_mode():
                    latent = anchors[index:index+1]
                    sample = generator(latent + NOISE * torch.randn_like(latent),
                                       embeddings[label:label+1])[0].cpu()
                pixels = ((sample.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
                patch = Image.fromarray(pixels)
                sw, sh = source_box[2]-source_box[0], source_box[3]-source_box[1]
                tw, th = target_box[2]-target_box[0], target_box[3]-target_box[1]
                if (sw >= sh) != (tw >= th):
                    patch = patch.rotate(90)
                patch = patch.resize((tw, th), Image.Resampling.LANCZOS)
                canvas = base.copy()
                canvas.paste(patch, target_box)
                image = paste(base, canvas, box_mask([target_box], 3), 3)
                stem = f'{cat}_{typ}_{number:03d}'
                ip, mp = directory/'images'/f'{stem}.jpg', directory/'masks'/f'{stem}.png'
                save_like(image, ip, base_row['image'])
                Image.fromarray(box_mask([target_box]).astype(np.uint8)*255).save(mp)
                condition = dict(box=list(target_box), site=list(site), donor_box=list(source_box),
                                 donor_label_json=source['label_json'], rule=rule, site_cost=cost,
                                 train_steps=a.steps*len(crops), gen_res=a.res)
                rows.append(dict(image=rel(ip), mask=rel(mp), label='1', source_image=source['image'],
                                 score=change_score(image, base, target_box), category=cat, defect_type=typ,
                                 base_normal_image=base_row['image'], gen_seed=7017,
                                 condition=json.dumps(condition, ensure_ascii=False)))
            print(f'{cat} {typ}: {a.per_type} generated', flush=True)
        sheet(pd.DataFrame(rows).query('category == @cat'), cat, directory/f'sheet_{cat}.jpg')
    frame = pd.DataFrame(rows)
    print(leak_check(frame, sp), flush=True)
    temporary = directory/'manifest.tmp'
    frame.to_csv(temporary, index=False)
    temporary.replace(directory/'manifest.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, choices=(1, 4, 16))
    parser.add_argument('--seed', type=int, default=0, choices=range(3))
    parser.add_argument('--steps', type=int, default=125)
    parser.add_argument('--res', type=int, default=256, choices=(128,256))
    parser.add_argument('--per-type', type=int, default=40)
    run(parser.parse_args())
