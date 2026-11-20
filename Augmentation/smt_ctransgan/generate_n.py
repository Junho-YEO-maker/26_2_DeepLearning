"""생성량 실험용 (11/20): 이미 학습한 SMT cTransGAN 체크포인트로 종류당 N장을 새로 뽑는다. 학습은 다시 하지 않는다.

generate.py(팀원 코드, 고치지 않음)가 학습 직후 저장한 {cat}_model.pt를 불러,
같은 순서로 donor 목록(종류 → K장 → 박스)을 다시 만들고 generate.py와 같은 생성 규칙(자리 고르기, 방향 맞춤, 붙이기, JPEG)으로 뽑는다.
선별(overgen)은 없다(generate.py와 같음). 난수열만 원래 40장과 다르게 둔다.
manifest 앞 N장 = 무작위 N장이므로 seg_ad의 `ctransgan_n160:N:0.5`가 20 ⊂ 40 ⊂ 80 ⊂ 160으로 자른다.

    cd Augmentation
    python -m smt_ctransgan.generate_n --k 1 --seed 0 --per-type 160 --tag n160
결과: Dataset/aihub_smt/generated/ctransgan_{tag}/K{k}_seed{s}/{images,masks,manifest.csv,sheet_pre.jpg} (AIHub 파생물, git 제외)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ctransgan.model import build_models
from smt_ctransgan.core import NETWORK, NOISE
from smt_sd_lora.generate import change_score, leak_check, rel, save_like, sheet
from smt_sd_lora.generate_fix import choose_site, place_fixed
from smt_sd_lora.smt import OUT, Split, box_mask, load_rgb, paste


def run(a):
    torch.set_num_threads(4)
    dev = torch.device('cuda')
    sp = Split(a.k, a.seed)
    src = OUT / 'ctransgan' / f'K{a.k}_seed{a.seed}'
    directory = OUT / f'ctransgan_{a.tag}' / f'K{a.k}_seed{a.seed}'
    if (directory / 'manifest.csv').exists():
        print(f'이미 있음: {directory}', flush=True)
        return
    (directory / 'images').mkdir(parents=True, exist_ok=True)
    (directory / 'masks').mkdir(exist_ok=True)
    rows = []
    for cat in a.category:
        ck = torch.load(src / f'{cat}_model.pt', map_location='cpu', weights_only=False)
        types = sp.types(cat)
        assert list(ck['types']) == list(types), (ck['types'], types)
        normals = sp.normals(cat)
        donors = []                                   # generate.py와 같은 순서 (anchor 번호가 이 순서)
        for label, typ in enumerate(types):
            for source, boxes in sp.defects(cat, typ):
                for box in boxes:
                    cands = [(row, site) for row, sites in normals for t, site in sites if t == typ]
                    donors.append((source, box, typ, label, cands))
        anchors, emb = ck['anchors'].to(dev), ck['embeddings'].to(dev)
        assert len(donors) == len(anchors), (len(donors), len(anchors))
        cfg = ck['config']
        _, generator, _ = build_models(dict(NETWORK, patch=cfg['res'] // 32), cfg['classes'])
        generator.load_state_dict(ck['generator'])
        generator = generator.to(dev).eval()
        rng = np.random.default_rng([7017, a.k, a.seed, 0 if cat == 'pre' else 1, a.per_type])
        torch.manual_seed(int(rng.integers(2**31)))
        for typ in types:
            group = [i for i, d in enumerate(donors) if d[2] == typ]
            for number in range(a.per_type):
                index = group[number % len(group)]
                source, source_box, _, label, cands = donors[index]
                (base_row, site), cost = choose_site(cands, typ, source_box, rng)
                target_box, rule = place_fixed(site, source_box, typ, rng)
                base = load_rgb(base_row['image'])
                with torch.inference_mode():
                    latent = anchors[index:index + 1]
                    sample = generator(latent + NOISE * torch.randn_like(latent), emb[label:label + 1])[0].cpu()
                patch = Image.fromarray(((sample.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy())
                sw, sh = source_box[2] - source_box[0], source_box[3] - source_box[1]
                tw, th = target_box[2] - target_box[0], target_box[3] - target_box[1]
                if (sw >= sh) != (tw >= th):
                    patch = patch.rotate(90)
                patch = patch.resize((tw, th), Image.Resampling.LANCZOS)
                canvas = base.copy()
                canvas.paste(patch, target_box)
                image = paste(base, canvas, box_mask([target_box], 3), 3)
                stem = f'{cat}_{typ}_{number:03d}'
                ip, mp = directory / 'images' / f'{stem}.jpg', directory / 'masks' / f'{stem}.png'
                save_like(image, ip, base_row['image'])
                Image.fromarray(box_mask([target_box]).astype(np.uint8) * 255).save(mp)
                cond = dict(box=list(target_box), site=list(site), donor_box=list(source_box),
                            donor_label_json=source['label_json'], rule=rule, site_cost=cost,
                            checkpoint=rel(src / f'{cat}_model.pt'), gen_res=cfg['res'])
                rows.append(dict(image=rel(ip), mask=rel(mp), label='1', source_image=source['image'],
                                 score=change_score(image, base, target_box), category=cat, defect_type=typ,
                                 base_normal_image=base_row['image'], gen_seed=7017,
                                 condition=json.dumps(cond, ensure_ascii=False)))
            print(f'{cat} {typ}: {a.per_type}장', flush=True)
        sheet(pd.DataFrame(rows).query('category == @cat'), cat, directory / f'sheet_{cat}.jpg')
    frame = pd.DataFrame(rows)
    print(leak_check(frame, sp), flush=True)
    tmp = directory / 'manifest.tmp'
    frame.to_csv(tmp, index=False)
    tmp.replace(directory / 'manifest.csv')
    print(f'저장: {directory} ({len(frame)}행)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True, choices=(1, 4, 16))
    p.add_argument('--seed', type=int, default=0, choices=range(3))
    p.add_argument('--category', nargs='+', default=['pre'], choices=['pre', 'solder'])
    p.add_argument('--per-type', type=int, default=160)
    p.add_argument('--tag', default='n160')
    run(p.parse_args())
