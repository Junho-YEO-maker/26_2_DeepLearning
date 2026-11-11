"""SD 인페인팅 + LoRA 학습: 조건 (K, seed)마다 PCB별 LoRA 하나 (결함 4종류는 프롬프트로 구분).

학습 데이터는 그 PCB의 K장 불량뿐이다(종류당 K장). 결함 주변 512×512 크롭을 매번 조금씩 다른 위치에서
잘라 쓰고, 결함 영역(조금 넓힘)을 인페인팅 마스크로 준다.

    cd Augmentation
    python -m sd_lora.train --k 2 --seed 0                  # pcb1–3 LoRA 3개
    python -m sd_lora.train --k 2 --seed 0 --pcb pcb1 --steps 50   # 짧은 시험
결과: sd_lora/runs/K{k}_seed{s}/{pcb}/  (pytorch_lora_weights.safetensors, train.json)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.kshot import PCBS, TYPES, TYPE_ID, KShot, crop_box, dilate, rng_for

HERE = Path(__file__).resolve().parent
BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
PROMPT = {
    'bent': 'a close-up photo of a sks printed circuit board with a bent pin defect',
    'melt': 'a close-up photo of a sks printed circuit board with a melted component defect',
    'scratch': 'a close-up photo of a sks printed circuit board with a scratch defect',
    'missing': 'a close-up photo of a sks printed circuit board with a missing part defect',
}


def to_tensor(image):
    return torch.from_numpy(np.asarray(image, np.float32) / 127.5 - 1).permute(2, 0, 1)


def make_sample(data, rng, res, mask_dilate):
    """K장 중 하나에서 결함 크롭, 인페인팅 마스크, 프롬프트를 만든다."""
    image, ids, t = data[int(rng.integers(len(data)))]
    m = ids == TYPE_ID[t]
    box = crop_box(m, res, image.size, rng, max_offset=res // 4)
    crop = image.crop(box)
    x0, y0, x1, y1 = box
    mk = dilate(m, mask_dilate)[y0:y1, x0:x1]
    if crop.size != (res, res):
        crop = crop.resize((res, res), Image.Resampling.LANCZOS)
        mk = np.asarray(Image.fromarray(mk.astype(np.uint8) * 255).resize((res, res))) > 127
    return to_tensor(crop), torch.from_numpy(mk.astype(np.float32))[None], PROMPT[t]


def train_one(ks, pcb, args, out_dir):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    dev = 'cuda'
    torch.manual_seed(int(rng_for('sd_lora', pcb, ks.k, ks.seed).integers(2**31)))
    rng = rng_for('sd_lora', pcb, ks.k, ks.seed, 1)
    data = []
    for t in TYPES:
        for row in ks.defects(pcb, t):
            image, ids = ks.load(row)
            data.append((image, ids, t))

    tok = CLIPTokenizer.from_pretrained(args.base_model, subfolder='tokenizer')
    te = CLIPTextModel.from_pretrained(args.base_model, subfolder='text_encoder').to(dev).eval()
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder='vae').to(dev).eval()
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder='unet').to(dev)
    sched = DDPMScheduler.from_pretrained(args.base_model, subfolder='scheduler')
    for m in (te, vae, unet):
        m.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights='gaussian',
                                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    with torch.no_grad():
        ids = tok([PROMPT[t] for t in TYPES], padding='max_length', max_length=tok.model_max_length,
                  truncation=True, return_tensors='pt').input_ids.to(dev)
        emb = dict(zip(TYPES, te(ids)[0]))
    prompt_to_type = {PROMPT[t]: t for t in TYPES}

    unet.train()
    start, losses = time.time(), []
    for step in range(args.steps):
        batch = [make_sample(data, rng, args.res, args.mask_dilate) for _ in range(args.batch)]
        img = torch.stack([b[0] for b in batch]).to(dev)
        msk = torch.stack([b[1] for b in batch]).to(dev)
        cond = torch.stack([emb[prompt_to_type[b[2]]] for b in batch])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            sf = vae.config.scaling_factor
            lat = vae.encode(img).latent_dist.sample() * sf
            masked = vae.encode(img * (msk < 0.5)).latent_dist.sample() * sf
        small = F.interpolate(msk, size=lat.shape[-2:])
        noise = torch.randn_like(lat)
        ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
        noisy = sched.add_noise(lat, noise, ts)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = unet(torch.cat([noisy, small, masked], 1), ts, cond).sample
        target = noise if sched.config.prediction_type == 'epsilon' else sched.get_velocity(lat, noise, ts)
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0 or step == 0:
            print(f'  {pcb} step {step + 1}/{args.steps} loss {np.mean(losses[-100:]):.4f}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(out_dir, unet_lora_layers=lora, safe_serialization=True)
    info = dict(pcb=pcb, k=ks.k, seed=ks.seed, base_model=args.base_model, steps=args.steps, batch=args.batch,
                lr=args.lr, rank=args.rank, res=args.res, mask_dilate=args.mask_dilate,
                n_train_images=len(data), sources=[r['image'] for t in TYPES for r in ks.defects(pcb, t)],
                final_loss=float(np.mean(losses[-100:])), minutes=(time.time() - start) / 60)
    (out_dir / 'train.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f'  저장: {out_dir}  ({info["minutes"]:.1f}분)', flush=True)
    del unet, vae, te
    torch.cuda.empty_cache()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pcb', nargs='+', default=list(PCBS))
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--mask-dilate', type=int, default=8, help='결함 마스크를 몇 px 넓혀 인페인팅할지')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: <저장소>/Dataset/visa/raw)')
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    assert torch.cuda.is_available(), 'GPU가 필요합니다'
    ks = KShot(a.k, a.seed, a.data_root)
    for pcb in a.pcb:
        out = HERE / 'runs' / f'K{a.k}_seed{a.seed}' / pcb
        if (out / 'pytorch_lora_weights.safetensors').exists() and not a.force:
            print(f'[건너뜀] {out}')
            continue
        train_one(ks, pcb, a, out)
