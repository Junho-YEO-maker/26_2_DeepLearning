"""AIHub SMT SD+LoRA: 납땜 문제 유형(nm·ms·rs·sh·ws)만 자리 규칙을 고쳐 다시 만든다 (11/18).

원래 generate.py의 문제 (K=1 seed 0 그림·수치로 확인)
  결함 박스를 실제 결함 박스 크기 그대로, 바탕의 '정상_{종류}' 자리 **한가운데에** 두고, 자리는 무작위로 고른다.
  - nm(미납): 실제 결함은 리드 하나(예: 97×36)인데 자리는 부품 전체(중앙값 160×253) → 부품 몸통 가운데에 그려지고 박스가 자리의 29%만 덮는다.
  - ms(미삽): 크기는 비슷하나 방향을 안 맞춘다 → 세로 결함이 가로 부품에 놓이면 가운데만 지워지고 부품 양끝이 남는다.
  - rs·sh·ws: 실제 결함은 작은 칩인데 큰 IC 자리에 놓이기도 해서 IC 위에 줄무늬·X 같은 무늬가 생긴다.
고친 규칙 (place_fixed, 다른 생성기도 그대로 가져다 쓸 수 있다)
  1. 자리 고르기: 그 종류 자리 전체에서 실제 결함 박스와 크기·가로세로비(방향 무시)가 가까운 순으로 상위 25%(최소 8개) 안에서 무작위.
     nm은 결함이 리드 하나라 크기는 안 보고 가로세로비도 안 본다(어느 부품이든 리드는 끝에 있다).
  2. 방향: 자리가 가로로 길면 박스도 가로로 (실제 결함 박스의 가로·세로를 바꿔서).
  3. ms: 박스 = 자리 전체(+6%). 부품이 통째로 없어져야 하므로.
     nm: 박스 크기는 실제 결함(방향 맞춤) × 0.9–1.1, 위치는 자리의 긴 축 **한쪽 끝**(리드·납땜 자리).
     rs·sh·ws: 박스 크기·위치 규칙은 원래와 같다(자리 중심 ±5%). 고른 자리가 비슷한 부품이라 무늬 문제가 줄어든다.
나머지 유형(납땜 cc·fl·ic·ml·pb·ph·st, 사전 6종)은 원래 sd_lora 행을 그대로 가져온다(같은 이미지 파일을 가리킨다).

    cd Augmentation
    python -m smt_sd_lora.generate_fix --k 1 --seed 0
결과: Dataset/aihub_smt/generated/sd_lora_fix/K{k}_seed{s}/{images,masks,manifest.csv,sheet_fix.jpg}
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from smt_sd_lora.generate import change_score, leak_check, rel, save_like
from smt_sd_lora.smt import BASE_MODEL, NEGATIVE, OUT, PROMPT, RECIPE, RES, RUNS, Split, box_mask, load_rgb, paste, rng_for

FIX = {'solder': ('nm', 'ms', 'rs', 'sh', 'ws')}
# ms: 영역이 부품 전체라도 SD가 주변 문맥을 따라 부품을 다시 그려서(11/18 K1 seed 0) 부품 단어를 부정 프롬프트에 넣는다
NEG_EXTRA = {'ms': 'electronic component, chip, resistor, capacitor, part body on the pads'}
TOP, TOP_MIN = .25, 8


def _wh(b):
    return b[2] - b[0], b[3] - b[1]


def site_cost(site, donor_box, t):
    """자리와 실제 결함 박스가 얼마나 다른가 (방향 무시). 작을수록 비슷한 부품."""
    if t == 'nm':
        return 0.0
    sw, sh = _wh(site)
    dw, dh = _wh(donor_box)
    aspect = abs(np.log(max(sw, sh) / max(min(sw, sh), 1)) - np.log(max(dw, dh) / max(min(dw, dh), 1)))
    size = abs(np.log((sw * sh) / max(dw * dh, 1)))
    return aspect + size


def choose_site(cands, t, donor_box, rng):
    """cands: [(바탕 row, 자리 박스)] → 비슷한 상위 25%(최소 8개) 안에서 무작위."""
    cost = np.array([site_cost(s, donor_box, t) for _, s in cands])
    n = max(TOP_MIN, int(np.ceil(len(cands) * TOP)))
    top = np.argsort(cost, kind='stable')[:n]
    return cands[int(top[rng.integers(len(top))])], float(cost[top].mean())


def place_fixed(site, donor_box, t, rng, jitter=0.05, scale=(0.9, 1.1)):
    """고친 자리 규칙: (결함 박스, 규칙 이름)."""
    sx0, sy0, sx1, sy1 = site
    sw, sh = sx1 - sx0, sy1 - sy0
    if t == 'ms':                                       # 부품 전체
        px, py = int(sw * .06), int(sh * .06)
        return (max(0, sx0 - px), max(0, sy0 - py), min(RES, sx1 + px), min(RES, sy1 + py)), 'whole_site'
    dw, dh = _wh(donor_box)
    if t == 'nm':                                       # 긴 축 한쪽 끝의 리드: 긴 축 방향 두께 = 실제 결함의 짧은 변, 가로지르는 폭 = 부품 폭
        thick = min(dw, dh) * rng.uniform(*scale)
        end = rng.integers(2)
        if sw >= sh:
            w, h = thick, sh * 1.06
            x0 = sx0 if end == 0 else sx1 - w
            y0 = (sy0 + sy1) / 2 - h / 2
        else:
            w, h = sw * 1.06, thick
            y0 = sy0 if end == 0 else sy1 - h
            x0 = (sx0 + sx1) / 2 - w / 2
        w, h = min(w, RES), min(h, RES)
        x0 = int(np.clip(x0, 0, RES - w))
        y0 = int(np.clip(y0, 0, RES - h))
        return (x0, y0, int(x0 + w), int(y0 + h)), 'lead_end'
    if (sw >= sh) != (dw >= dh):                        # 방향 맞춤
        dw, dh = dh, dw
    w = min(dw * rng.uniform(*scale), RES)
    h = min(dh * rng.uniform(*scale), RES)
    cx = (sx0 + sx1) / 2 + rng.uniform(-jitter, jitter) * sw
    cy = (sy0 + sy1) / 2 + rng.uniform(-jitter, jitter) * sh
    x0 = int(np.clip(cx - w / 2, 0, RES - w))
    y0 = int(np.clip(cy - h / 2, 0, RES - h))
    return (x0, y0, int(x0 + w), int(y0 + h)), 'matched_site'


def sheet(rows, path, cell=170, n=3):
    """문제 유형마다: 원래 sd_lora 1장 | 고친 것 n장 (박스 빨강)."""
    from PIL import ImageDraw
    sel = [(t, g) for t, g in rows.groupby('defect_type')]
    canvas = Image.new('RGB', (cell * (n + 1), cell * len(sel)), 'white')
    for i, (t, g) in enumerate(sel):
        for j, (_, r) in enumerate(g.head(n + 1).iterrows()):
            c = json.loads(r.condition)
            im = load_rgb(r.image)
            ImageDraw.Draw(im).rectangle(c['box'], outline=(255, 0, 0), width=2)
            x0, y0, x1, y1 = c['box']
            s = max(x1 - x0, y1 - y0) + 120
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            canvas.paste(im.crop((cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)).resize((cell, cell)), (j * cell, i * cell))
            ImageDraw.Draw(canvas).text((j * cell + 3, i * cell + 3), f'{r.category} {t} {"old" if j == 0 else "fix"}',
                                        fill=(255, 255, 0))
    canvas.save(path, quality=90)


def main(a):
    from sd_lora._compat import disable_old_torchao
    disable_old_torchao()
    from diffusers import StableDiffusionInpaintPipeline

    sp = Split(a.k, a.seed)
    src = OUT / 'sd_lora' / f'K{a.k}_seed{a.seed}' / 'manifest.csv'
    cdir = OUT / a.out_name / f'K{a.k}_seed{a.seed}'
    mpath = cdir / 'manifest.csv'
    types_now = {c: tuple(t for t in ts if t in a.types) for c, ts in FIX.items()}
    partial = set(a.types) != {t for ts in FIX.values() for t in ts}
    if mpath.exists() and not a.force and not partial:
        print(f'[건너뜀] {mpath} (이미 있음, 다시 하려면 --force)')
        return
    old = pd.read_csv(src, dtype=str, keep_default_na=False)
    (cdir / 'images').mkdir(parents=True, exist_ok=True)
    (cdir / 'masks').mkdir(exist_ok=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(a.base_model, torch_dtype=torch.float16,
                                                          safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for cat, types in types_now.items():
        lora = RUNS / f'K{a.k}_seed{a.seed}' / cat
        info = json.loads((lora / 'train.json').read_text())
        assert info['recipe'] == RECIPE, f'{lora}: 다른 방식 LoRA ({info["recipe"]})'
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora)
        normals = sp.normals(cat)
        for t in types:
            donors = sp.defects(cat, t)
            cands = [(r, b) for r, s in normals for typ, b in s if typ == t]
            jobs = []
            for i in range(a.per_type * a.overgen):
                rng, gseed = rng_for('fix', cat, a.k, a.seed, t, i)
                drow, dboxes = donors[int(rng.integers(len(donors)))]
                dbox = dboxes[int(rng.integers(len(dboxes)))]
                (brow, site), cost = choose_site(cands, t, dbox, rng)
                box, rule = place_fixed(site, dbox, t, rng)
                jobs.append(dict(donor=drow['image'], donor_json=drow['label_json'], base=brow['image'], box=box, site=site,
                                 donor_box=dbox, rule=rule, cost=cost, region=box_mask([box], a.mask_dilate),
                                 sampler_seed=int(rng.integers(2**31)), gen_seed=gseed))
            made = []
            for s in range(0, len(jobs), a.batch):
                chunk = jobs[s:s + a.batch]
                bimgs = [load_rgb(j['base']) for j in chunk]
                neg = NEGATIVE + (', ' + NEG_EXTRA[t] if t in NEG_EXTRA else '')
                outs = pipe(prompt=[PROMPT[cat][t]] * len(chunk), negative_prompt=[neg] * len(chunk),
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
                            guidance=a.guidance, mask_dilate=a.mask_dilate, overgen=a.overgen, lora=rel(lora),
                            fix=dict(rule=j['rule'], site_cost=round(j['cost'], 3)))
                rows.append(dict(image=rel(ip), mask=rel(mp), label='1', source_image=j['donor'], score=f'{score:.2f}',
                                 category=cat, defect_type=t, base_normal_image=j['base'], gen_seed=str(j['gen_seed']),
                                 condition=json.dumps(cond, ensure_ascii=False)))
            sc = [made[x][0] for x in keep]
            print(f'  {cat} {t}: {len(keep)}/{len(made)}장, change {np.mean(sc):.1f}, 자리 후보 {len(cands)}개', flush=True)
    new = pd.DataFrame(rows)
    fixed = {(c, t) for c, ts in types_now.items() for t in ts}
    prev = pd.read_csv(mpath, dtype=str, keep_default_na=False) if (partial and mpath.exists()) else old   # 일부 유형만 다시: 이전 고친 행 유지
    keep_old = prev[[(c, t) not in fixed for c, t in zip(prev.category, prev.defect_type)]]
    df = pd.concat([keep_old, new])[old.columns]
    print(leak_check(df, sp), flush=True)
    df.to_csv(mpath, index=False)
    both = pd.concat([old[[(c, t) in fixed for c, t in zip(old.category, old.defect_type)]].groupby('defect_type').head(1), new])
    sheet(both.sort_values('defect_type', kind='stable'), cdir / 'sheet_fix.jpg')
    print(f'저장: {cdir} ({len(df)}행, 다시 만든 {len(new)}행)', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--per-type', type=int, default=40)
    p.add_argument('--overgen', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance', type=float, default=7.5)
    p.add_argument('--mask-dilate', type=int, default=8)
    p.add_argument('--feather', type=int, default=3)
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--types', nargs='+', default=list(FIX['solder']), choices=FIX['solder'], help='다시 만들 유형 (일부만 주면 나머지 고친 행은 유지)')
    p.add_argument('--out-name', default='sd_lora_fix', help='결과 폴더 이름 (시험용으로 바꿀 수 있음)')
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    assert torch.cuda.is_available(), 'GPU가 필요합니다'
    main(a)
