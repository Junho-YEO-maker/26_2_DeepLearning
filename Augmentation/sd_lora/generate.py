"""SD 인페인팅 + LoRA 생성: 조건 (K, seed)마다 PCB 3개 × 종류 4개 × 40장.

생성 규칙은 copy-paste와 같다. K장 중 하나의 결함 마스크를 train_normal 바탕의 원래 자리 ± 30px로 옮기고,
그 주변 512×512 크롭을 LoRA로 인페인팅한 뒤 마스크(조금 넓힌) 안만 원본 해상도 바탕에 다시 붙인다.

    cd Augmentation
    python -m sd_lora.generate --k 2 --seed 0
결과: Dataset/generated/sd_lora/K{k}_seed{s}/{images,masks,manifest.csv}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.kshot import (PCBS, STRUCTURE_TYPES, TYPES, TYPE_ID, KShot, ManifestWriter, change_region, crop_box,
                          dilate, paste_back, place_mask, rng_for)
from sd_lora.train import BASE_MODEL, PROMPT, RECIPE

HERE = Path(__file__).resolve().parent
NEGATIVE = 'blurry, cartoon, drawing, text, watermark'


def main(a):
    from sd_lora._compat import disable_old_torchao
    disable_old_torchao()
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image

    ks = KShot(a.k, a.seed, a.data_root)
    writer = ManifestWriter('sd_lora', a.k, a.seed, out_root=a.out_root, force=a.force)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(a.base_model, torch_dtype=torch.float16,
                                                          safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for pcb in a.pcb:
        lora = Path(a.runs_dir) / f'K{a.k}_seed{a.seed}' / pcb
        assert (lora / 'pytorch_lora_weights.safetensors').exists(), f'LoRA 없음: {lora} (train 먼저)'
        recipe = json.loads((lora / 'train.json').read_text()).get('recipe', 'v1')
        assert recipe == RECIPE, f'{lora}: 옛 방식({recipe}) LoRA입니다. train을 다시 돌리세요 (지금 방식 {RECIPE})'
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora)
        normals = ks.normals(pcb)
        for ti, t in enumerate(TYPES):
            donors = ks.defects(pcb, t)
            jobs = []
            for i in range(a.per_type):
                rng = rng_for('sd_lora', pcb, a.k, a.seed, 2, ti, i)
                donor = donors[int(rng.integers(len(donors)))]
                base = normals[int(rng.integers(len(normals)))]
                d_img, d_ids = ks.load(donor)
                b_img, _ = ks.load(base)
                mask, place = place_mask(d_img, d_ids, t, b_img, rng, a.jitter)   # 기판 정렬 + 종류별 jitter
                # bent·missing: 마스크 밖에 남는 원래 핀까지 다시 그린다 (common/kshot.py change_region)
                paint = change_region(d_img, b_img, mask, place) if t in STRUCTURE_TYPES and a.structure else mask
                region = dilate(paint, a.mask_dilate)
                box = crop_box(region, a.res, b_img.size)
                jobs.append((donor, base, b_img, mask, place, box, region, int(rng.integers(2**31))))
            for s in range(0, len(jobs), a.batch):
                chunk = jobs[s:s + a.batch]
                crops = [j[2].crop(j[5]).resize((a.res, a.res)) for j in chunk]
                masks = [Image.fromarray(j[6][j[5][1]:j[5][3], j[5][0]:j[5][2]].astype(np.uint8) * 255)
                         .resize((a.res, a.res)) for j in chunk]
                gens = [torch.Generator('cuda').manual_seed(j[7]) for j in chunk]
                outs = pipe(prompt=[PROMPT[t]] * len(chunk), negative_prompt=[NEGATIVE] * len(chunk),
                            image=crops, mask_image=masks, height=a.res, width=a.res,
                            num_inference_steps=a.steps, guidance_scale=a.guidance, generator=gens).images
                for (donor, base, b_img, mask, place, box, region, gseed), out in zip(chunk, outs):
                    image = paste_back(b_img, out, box, region, feather=a.feather)
                    writer.add(image, mask, pcb, t, donor, base, dict(
                        place, crop_box=list(box), sampler_seed=gseed,
                        steps=a.steps, guidance=a.guidance, mask_dilate=a.mask_dilate,
                        inpaint_px=int(region.sum()), mask_px=int(mask.sum())))
            print(f'  {pcb} {t}: {a.per_type}장', flush=True)
    print(writer.close(), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pcb', nargs='+', default=list(PCBS))
    p.add_argument('--per-type', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=30, help='디퓨전 샘플링 단계 수')
    p.add_argument('--guidance', type=float, default=7.5)
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--jitter', type=int, default=None, help='기본: 종류별 (bent·missing 5px, melt·scratch 30px, copy-paste와 같음)')
    p.add_argument('--mask-dilate', type=int, default=8)
    p.add_argument('--feather', type=int, default=3)
    p.add_argument('--no-structure', dest='structure', action='store_false',
                   help='bent·missing에서 원래 핀 자리까지 다시 그리는 처리를 끈다')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--data-root')
    p.add_argument('--runs-dir', default=str(HERE / 'runs'), help='train.py와 같은 LoRA 위치')
    p.add_argument('--out-root', help='기본: <저장소>/Dataset/generated/sd_lora')
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    assert torch.cuda.is_available(), 'GPU가 필요합니다'
    main(a)
