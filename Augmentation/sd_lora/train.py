"""SD 인페인팅 + LoRA 학습: 조건 (K, seed)마다 PCB별 LoRA 하나 (결함 4종류는 프롬프트로 구분).

학습 데이터는 그 PCB의 K장 불량(종류당 K장)과, 그 결함 자리를 비교·채우기용으로 쓰는 train_normal 몇 장이다.
- 결함 샘플: 결함 주변 512×512 크롭(매번 조금 다른 위치)에서 인페인팅 영역을 가리고 실제 불량을 정답으로 준다.
  bent·missing은 영역이 결함 마스크 + 정상 기판과 비교해 달라진 곳(원래 핀 자리)이다. 생성 때와 같은
  common/kshot.py change_region을 쓴다. 그래서 "휜 핀은 그리고 원래 핀 자리는 배경으로 채우기"를 배운다.
- 정상 샘플(--p-normal 비율): 같은 영역을 정상 기판의 같은 자리에 씌우고 결함 단어 없는 프롬프트로
  정상 기판을 정답으로 준다. 결함 단어가 없으면 그냥 기판으로 채우게 해서 흰 얼룩 남발과 K장 암기를 줄인다.

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

from common.kshot import (PCBS, STRUCTURE_TYPES, TYPES, TYPE_ID, KShot, board_transform, change_region, crop_box,
                          dilate, rng_for, to_base_frame)

HERE = Path(__file__).resolve().parent
BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
# 종류끼리 헷갈리지 않게 "무엇이 어떻게 보이는지"를 쓴다 (단어 하나만 다른 프롬프트는 서로 섞이기 쉽다).
# 'sks' = 이 PCB 스타일을 묶는 표시. 정상 프롬프트에는 결함 단어를 하나도 넣지 않는다.
PROMPT = {
    'bent': 'close-up photo of sks circuit board, a metal header pin bent and tilted out of line',
    'melt': 'close-up photo of sks circuit board, a component melted into a white burnt blob',
    'scratch': 'close-up photo of sks circuit board, thin white scratch lines on the board surface',
    'missing': 'close-up photo of sks circuit board, a pin or component is missing, empty spot',
}
PROMPT_NORMAL = 'close-up photo of sks circuit board, clean, intact, all parts in place'
RECIPE = 'v2-region+normal'      # 학습 방식이 바뀌면 올린다. 옛 방식 LoRA는 건너뛰지 않고 다시 학습, 생성에서는 거부


def to_tensor(image):
    return torch.from_numpy(np.asarray(image, np.float32) / 127.5 - 1).permute(2, 0, 1)


def build_data(ks, pcb, refs, rng):
    """불량 K장마다 정상 refs장을 정렬해 인페인팅 영역을 만든다.
    돌아가는 값: 결함 샘플 [(불량 이미지, 영역들, 종류)], 정상 샘플 [(정상 이미지, 영역)], 쓴 정상 경로,
    불량마다 (영역 넓이 / 결함 마스크 넓이) 평균."""
    normals = ks.normals(pcb)
    data, normal_samples, used, grow = [], [], [], []
    for t in TYPES:
        for row in ks.defects(pcb, t):
            image, ids = ks.load(row)
            m = ids == TYPE_ID[t]
            regions = []
            for r in rng.choice(len(normals), refs, replace=False):
                n_img, _ = ks.load(normals[int(r)])
                used.append(normals[int(r)]['image'])
                if t in STRUCTURE_TYPES:
                    tf = board_transform(n_img, image)           # 정상 → 불량 좌표
                    info = {'align': {'rot180': tf['rot180']}, 'applied_shift_xy': [tf['dx'], tf['dy']]}
                    region = change_region(n_img, image, m, info)
                else:
                    region = m
                regions.append(region)
                normal_samples.append((n_img, to_base_frame(region, board_transform(image, n_img))))
            data.append((image, regions, t))
            grow.append(float(np.mean([r.sum() for r in regions]) / max(int(m.sum()), 1)))
    return data, normal_samples, used, grow


def make_sample(data, normal_samples, rng, res, mask_dilate, p_normal):
    """결함 샘플(또는 p_normal 확률로 정상 샘플) 하나: 크롭, 인페인팅 마스크, 프롬프트."""
    if normal_samples and rng.random() < p_normal:
        image, region = normal_samples[int(rng.integers(len(normal_samples)))]
        prompt = PROMPT_NORMAL
    else:
        image, regions, t = data[int(rng.integers(len(data)))]
        region = regions[int(rng.integers(len(regions)))]
        prompt = PROMPT[t]
    box = crop_box(region, res, image.size, rng, max_offset=res // 4)
    crop = image.crop(box)
    x0, y0, x1, y1 = box
    mk = np.asarray(Image.fromarray(dilate(region, mask_dilate)).crop(box)) > 0   # box가 이미지 밖이어도 됨
    if crop.size != (res, res):
        crop = crop.resize((res, res), Image.Resampling.LANCZOS)
        mk = np.asarray(Image.fromarray(mk.astype(np.uint8) * 255).resize((res, res))) > 127
    x = to_tensor(crop)
    x = (x * rng.uniform(0.95, 1.05) + rng.uniform(-0.05, 0.05)).clamp(-1, 1)    # 밝기·대비 ±5%
    return x, torch.from_numpy(mk.astype(np.float32))[None], prompt


def train_one(ks, pcb, args, out_dir):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from sd_lora._compat import disable_old_torchao
    disable_old_torchao()
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'   # CPU는 코드 점검용
    torch.manual_seed(int(rng_for('sd_lora', pcb, ks.k, ks.seed).integers(2**31)))
    rng = rng_for('sd_lora', pcb, ks.k, ks.seed, 1)
    data, normal_samples, normal_used, grow = build_data(ks, pcb, args.refs, rng_for('sd_lora', pcb, ks.k, ks.seed, 3))

    # A100 등(compute capability ≥ 8)은 bf16, Colab T4는 fp16 + GradScaler + gradient checkpointing.
    # 고정 가중치는 이 dtype으로, 학습하는 LoRA 가중치만 fp32로 둔다.
    if dev == 'cuda':
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dt = torch.float32
    tok = CLIPTokenizer.from_pretrained(args.base_model, subfolder='tokenizer')
    te = CLIPTextModel.from_pretrained(args.base_model, subfolder='text_encoder', torch_dtype=dt).to(dev, dtype=dt).eval()
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder='vae', torch_dtype=dt).to(dev, dtype=dt).eval()
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder='unet', torch_dtype=dt).to(dev, dtype=dt)
    sched = DDPMScheduler.from_pretrained(args.base_model, subfolder='scheduler')
    for m in (te, vae, unet):
        m.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights='gaussian',
                                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    if dt == torch.float16 or args.grad_ckpt:
        unet.enable_gradient_checkpointing()
    scaler = torch.amp.GradScaler(dev, enabled=dt == torch.float16)
    gpu = torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    print(f'  {pcb}: {gpu} dtype={dt} 불량 {len(data)}장, 정상 샘플 {len(normal_samples)}개, '
          f'p_normal={args.p_normal}', flush=True)
    prompts = [PROMPT[t] for t in TYPES] + [PROMPT_NORMAL]
    with torch.no_grad():
        ids = tok(prompts, padding='max_length', max_length=tok.model_max_length,
                  truncation=True, return_tensors='pt').input_ids.to(dev)
        emb = dict(zip(prompts, te(ids)[0]))

    unet.train()
    start, losses = time.time(), []
    for step in range(args.steps):
        batch = [make_sample(data, normal_samples, rng, args.res, args.mask_dilate, args.p_normal)
                 for _ in range(args.batch)]
        img = torch.stack([b[0] for b in batch]).to(dev, dtype=dt)
        msk = torch.stack([b[1] for b in batch]).to(dev, dtype=dt)
        cond = torch.stack([emb[b[2]] for b in batch])
        with torch.no_grad():
            sf = vae.config.scaling_factor
            lat = vae.encode(img).latent_dist.sample() * sf
            masked = vae.encode(img * (msk < 0.5)).latent_dist.sample() * sf
        small = F.interpolate(msk, size=lat.shape[-2:])
        noise = torch.randn_like(lat)
        ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
        noisy = sched.add_noise(lat, noise, ts)
        with torch.autocast(dev, dtype=dt, enabled=dt != torch.float32):
            pred = unet(torch.cat([noisy, small, masked], 1), ts, cond).sample
        target = noise if sched.config.prediction_type == 'epsilon' else sched.get_velocity(lat, noise, ts)
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % 100 == 0 or step == 0:
            print(f'  {pcb} step {step + 1}/{args.steps} loss {np.mean(losses[-100:]):.4f}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(out_dir, unet_lora_layers=lora, safe_serialization=True)
    info = dict(recipe=RECIPE, p_normal=args.p_normal, refs=args.refs, normal_refs=normal_used,
                region_vs_mask=[round(g, 2) for g in grow],
                pcb=pcb, k=ks.k, seed=ks.seed, base_model=args.base_model, steps=args.steps, batch=args.batch,
                lr=args.lr, rank=args.rank, res=args.res, mask_dilate=args.mask_dilate,
                n_train_images=len(data), gpu=gpu, dtype=str(dt), sources=[r['image'] for t in TYPES for r in ks.defects(pcb, t)],
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
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--p-normal', type=float, default=0.25, help='배치 중 정상 샘플 비율')
    p.add_argument('--refs', type=int, default=3, help='불량 1장당 비교·채우기에 쓸 train_normal 수')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--mask-dilate', type=int, default=8, help='결함 마스크를 몇 px 넓혀 인페인팅할지')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: <저장소>/Dataset/visa/raw)')
    p.add_argument('--grad-ckpt', action='store_true', help='gradient checkpointing (T4에서는 자동으로 켜짐)')
    p.add_argument('--runs-dir', default=str(HERE / 'runs'), help='LoRA 저장 위치 (Colab에서는 Drive로)')
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    assert torch.cuda.is_available(), 'GPU가 필요합니다'
    ks = KShot(a.k, a.seed, a.data_root)
    for pcb in a.pcb:
        out = Path(a.runs_dir) / f'K{a.k}_seed{a.seed}' / pcb
        if (out / 'pytorch_lora_weights.safetensors').exists() and not a.force:
            old = json.loads((out / 'train.json').read_text()) if (out / 'train.json').exists() else {}
            if old.get('recipe') == RECIPE:
                print(f'[건너뜀] {out}')
                continue
            print(f'[다시 학습] {out}: 옛 방식({old.get("recipe", "v1")}) LoRA')
        train_one(ks, pcb, a, out)
