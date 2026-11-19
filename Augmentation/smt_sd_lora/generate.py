"""AIHub SMT: 학습한 LoRA로 조건 (K, seed)마다 category × 종류 × 40장 생성 불량을 만든다.

한 장 만들기
  1. K장 중 그 종류 불량 하나(donor)와 그 박스 하나를 고른다.
  2. train_normal 중 '정상_{종류}' 박스가 있는 이미지 하나(바탕)를 고르고, 그 자리에 donor 박스 크기(×0.9–1.1)로
     결함 박스를 둔다(중심 ±5%).
  3. 박스를 8px 넓힌 직사각형을 종류 프롬프트로 인페인팅하고, 그 영역 안만 바탕에 붙인다(가장자리 3px feather).
  4. 바탕과 같은 JPEG 양자화 테이블·서브샘플링으로 저장한다(압축 흔적으로 생성을 알아채는 지름길 방지).
--overgen N: 종류마다 40×N장을 만들고 박스 안에서 바탕과 가장 많이 달라진 40장만 남긴다(score = 박스 안 평균 |생성 − 바탕|).

    cd Augmentation
    python -m smt_sd_lora.generate --k 1 --seed 0
    python -m smt_sd_lora.generate --k 1 --seed 0 --category pre --per-type 4 --overgen 1 --tag test   # 짧은 시험
결과: Dataset/aihub_smt/generated/sd_lora[_{tag}]/K{k}_seed{s}/{images,masks,manifest.csv,sheet_{category}.jpg}
AD: 저장소 루트에서 python -m AnomalyDetection.smt.run --models knn_mlp devnet --ks 1 --seeds 0 --augs sd_lora
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, JpegImagePlugin

from smt_sd_lora.smt import (BASE_MODEL, CATEGORIES, NEGATIVE, OUT, PROMPT, RECIPE, RES, ROOT, RUNS, Split, box_mask,
                             load_rgb, paste, place, rng_for)


def change_score(image, base, box):
    m = box_mask([box])
    a = np.asarray(image.convert('L'), np.float32)[m]
    b = np.asarray(base.convert('L'), np.float32)[m]
    return float(np.abs(a - b).mean())


def save_like(image, path, base_rel):
    """바탕 JPEG와 같은 양자화 테이블·크로마 서브샘플링으로 저장."""
    with Image.open(ROOT / base_rel) as b:
        q, sub = b.quantization, JpegImagePlugin.get_sampling(b)
    image.save(path, 'JPEG', qtables=q, subsampling=sub if sub in (0, 1, 2) else 2)


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def sheet(rows, cat, path, per_type=2, cell=192):
    """종류마다 per_type줄: 바탕 | 생성(박스 빨강) | donor 실제 불량(종류 박스 초록)."""
    from smt_sd_lora.smt import boxes
    sel = rows.groupby('defect_type').head(per_type)
    canvas = Image.new('RGB', (cell * 3, cell * len(sel)), 'white')
    for i, (_, r) in enumerate(sel.iterrows()):
        c = json.loads(r.condition)
        gen = Image.open(ROOT / r.image).convert('RGB')
        ImageDraw.Draw(gen).rectangle(c['box'], outline=(255, 0, 0), width=3)
        donor = load_rgb(r.source_image)
        d = ImageDraw.Draw(donor)
        for lab, typ, b in boxes({'label_json': c['donor_label_json']}, cat):
            if lab == '불량' and typ == r.defect_type:
                d.rectangle(b, outline=(0, 255, 0), width=3)
        for j, im in enumerate((load_rgb(r.base_normal_image), gen, donor)):
            canvas.paste(im.resize((cell, cell)), (j * cell, i * cell))
        ImageDraw.Draw(canvas).text((4, i * cell + 4), f'{r.defect_type} {float(r.score):.0f}', fill=(255, 255, 0))
    canvas.save(path, quality=90)


def leak_check(df, sp):
    for _, r in df.iterrows():
        src = sp.kshot[(sp.kshot.category == r.category) & (sp.kshot.defect_type == r.defect_type)]
        assert r.source_image in set(src.image), f'source가 train_K 밖: {r.source_image}'
        assert r.base_normal_image in set(sp.normal[sp.normal.category == r.category].image), r.base_normal_image
        assert r.source_image not in sp.forbidden and r.base_normal_image not in sp.forbidden
    return f'누수 검사 PASS ({len(df)}행)'


def main(a):
    from sd_lora._compat import disable_old_torchao
    disable_old_torchao()
    from diffusers import StableDiffusionInpaintPipeline

    sp = Split(a.k, a.seed)
    method = 'sd_lora' + (f'_{a.tag}' if a.tag else '')
    cdir = OUT / method / f'K{a.k}_seed{a.seed}'
    mpath = cdir / 'manifest.csv'
    old = pd.read_csv(mpath) if mpath.exists() else None
    todo = [c for c in a.category if a.force or old is None or c not in set(old.category)]
    if not todo:
        print(f'[건너뜀] {cdir} (이미 있음, 다시 하려면 --force)')
        return
    (cdir / 'images').mkdir(parents=True, exist_ok=True)
    (cdir / 'masks').mkdir(exist_ok=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(a.base_model, torch_dtype=torch.float16,
                                                          safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for cat in todo:
        lora = Path(a.runs_dir) / f'K{a.k}_seed{a.seed}' / cat
        info = json.loads((lora / 'train.json').read_text())
        assert info['recipe'] == RECIPE, f'{lora}: 다른 방식 LoRA ({info["recipe"]})'
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora)
        normals = sp.normals(cat)
        for t in sp.types(cat):
            donors = sp.defects(cat, t)
            bases = [(r, s) for r, s in normals if any(typ == t for typ, _ in s)]
            jobs = []
            for i in range(a.per_type * a.overgen):
                rng, gseed = rng_for(cat, a.k, a.seed, t, i)
                drow, dboxes = donors[int(rng.integers(len(donors)))]
                dbox = dboxes[int(rng.integers(len(dboxes)))]
                brow, bsites = bases[int(rng.integers(len(bases)))]
                box, site = place(bsites, t, dbox, rng)
                jobs.append(dict(donor=drow['image'], donor_json=drow['label_json'], base=brow['image'], box=box,
                                 site=site, donor_box=dbox, region=box_mask([box], a.mask_dilate),
                                 sampler_seed=int(rng.integers(2**31)), gen_seed=gseed))
            made = []
            for s in range(0, len(jobs), a.batch):
                chunk = jobs[s:s + a.batch]
                bimgs = [load_rgb(j['base']) for j in chunk]
                outs = pipe(prompt=[PROMPT[cat][t]] * len(chunk), negative_prompt=[NEGATIVE] * len(chunk),
                            image=bimgs, mask_image=[Image.fromarray(j['region'].astype(np.uint8) * 255) for j in chunk],
                            height=RES, width=RES, num_inference_steps=a.steps, guidance_scale=a.guidance,
                            generator=[torch.Generator('cuda').manual_seed(j['sampler_seed']) for j in chunk]).images
                for j, b, o in zip(chunk, bimgs, outs):
                    img = paste(b, o, j['region'], a.feather)
                    made.append((change_score(img, b, j['box']), img, j))
            keep = sorted(sorted(range(len(made)), key=lambda x: -made[x][0])[:a.per_type])
            for n, x in enumerate(keep):
                score, img, j = made[x]
                ip, mp = cdir / 'images' / f'{cat}_{t}_{n:03d}.jpg', cdir / 'masks' / f'{cat}_{t}_{n:03d}.png'
                save_like(img, ip, j['base'])
                Image.fromarray(box_mask([j['box']]).astype(np.uint8) * 255).save(mp)
                cond = dict(box=list(j['box']), site=list(j['site']), donor_box=list(j['donor_box']),
                            donor_label_json=j['donor_json'], sampler_seed=j['sampler_seed'], steps=a.steps,
                            guidance=a.guidance, mask_dilate=a.mask_dilate, overgen=a.overgen, lora=rel(lora))
                rows.append(dict(image=rel(ip), mask=rel(mp), label=1, source_image=j['donor'], score=f'{score:.2f}',
                                 category=cat, defect_type=t, base_normal_image=j['base'], gen_seed=j['gen_seed'],
                                 condition=json.dumps(cond, ensure_ascii=False)))
            sc = [made[x][0] for x in keep]
            print(f'  {cat} {t}: {len(keep)}/{len(made)}장, change {np.mean(sc):.1f} (min {np.min(sc):.1f}), '
                  f'바탕 후보 {len(bases)}장', flush=True)
    new = pd.DataFrame(rows)
    df = pd.concat([old[~old.category.isin(todo)], new]) if old is not None else new
    print(leak_check(df, sp), flush=True)
    df.to_csv(mpath, index=False)
    for cat in todo:
        sheet(new[new.category == cat], cat, cdir / f'sheet_{cat}.jpg')
    print(f'저장: {cdir} ({len(df)}행)', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--category', nargs='+', default=list(CATEGORIES), choices=CATEGORIES)
    p.add_argument('--per-type', type=int, default=40)
    p.add_argument('--overgen', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=30, help='디퓨전 샘플링 단계 수')
    p.add_argument('--guidance', type=float, default=7.5)
    p.add_argument('--mask-dilate', type=int, default=8)
    p.add_argument('--feather', type=int, default=3)
    p.add_argument('--tag', default='', help='결과 폴더·aug 이름 접미사 (예: test → sd_lora_test)')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--runs-dir', default=str(RUNS))
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    assert torch.cuda.is_available(), 'GPU가 필요합니다'
    main(a)
