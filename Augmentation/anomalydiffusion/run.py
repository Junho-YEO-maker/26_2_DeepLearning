"""AnomalyDiffusion(Hu et al., AAAI 2024, 공식 코드)을 K장 실험에 맞춰 돌리는 어댑터.

공식 코드를 그대로 쓰면 이 실험과 맞지 않는 점이 세 가지 있어서, 입력과 출력을 이렇게 바꿔 끼운다.
  1. 공식 로더는 test/<종류>/ 이미지 중 앞쪽 1/3만 학습에 쓴다(idx ≤ len//3).
     → 진짜 K장 뒤에 같은 이미지 사본을 붙여 개수를 3n−1로 맞춘다. 그러면 정확히 진짜 n장만 학습된다.
       (사본은 공식 코드의 검증용 자리만 채운다. test/val 이미지는 하나도 들어가지 않는다.)
  2. 공식 생성은 이미지 전체를 256px로 줄여서 만든다. PCB 결함은 이미지의 0.1–6%라 거의 사라진다.
     → 결함 주변 512×512 크롭을 "샘플"로 넣고, 생성한 크롭을 원본 해상도 바탕에 다시 붙인다.
  3. 공식 마스크 생성 모델은 (샘플, 종류)마다 30,001 step을 학습한다(우리 조건이면 수십 번).
     → 마스크는 K장 결함 마스크를 원래 자리 ± jitter로 옮겨 쓴다(copy-paste, sd_lora와 같은 규칙).

샘플 하나 = K장 중 결함 이미지 하나(그 결함 주변 크롭). 이름은 {pcb}-{종류}-{j}.

    cd Augmentation
    python -m anomalydiffusion.run all --k 2 --seed 0          # prepare → train → generate → export
    python -m anomalydiffusion.run prepare --k 2 --seed 0      # 단계별로도 가능
결과: Dataset/generated/anomalydiffusion/K{k}_seed{s}/{images,masks,manifest.csv}
작업 폴더: anomalydiffusion/work/K{k}_seed{s}/ (공식 코드가 현재 폴더 기준 경로를 쓰므로 조건마다 따로 둔다)
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from common.kshot import (LANCZOS, PCBS, TYPES, TYPE_ID, KShot, ManifestWriter, aligned_crop, board_transform,
                          crop_box, dilate, jitter_mask, map_box, paste_back, rng_for)

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'upstream'
CONFIG = 'configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml'


def work_dir(k, seed):
    return HERE / 'work' / f'K{k}_seed{seed}'


def save_png(arr_or_img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.save(path)


# ---------------------------------------------------------------- prepare
def prepare(a):
    ks = KShot(a.k, a.seed, a.data_root)
    w = work_dir(a.k, a.seed)
    if w.exists():
        if not a.force:
            sys.exit(f'[중단] 이미 있음: {w} (다시 만들려면 --force)')
        shutil.rmtree(w)
    data = w / 'data'
    meta, lines = {}, []
    for pcb in PCBS:
        normals = ks.normals(pcb)
        for t in TYPES:
            for j, donor in enumerate(ks.defects(pcb, t)):
                sample = f'{pcb}-{t}-{j}'
                rng = rng_for('anomalydiffusion', pcb, a.k, a.seed, TYPE_ID[t], j)
                img, ids = ks.load(donor)
                m = ids == TYPE_ID[t]
                box = crop_box(m, a.res, img.size)
                x0, y0, x1, y1 = box
                # 1. test/<종류>: 진짜 1장 + 사본 1장 = 2장 → 공식 로더가 idx 0(진짜)만 학습
                for n in range(2):
                    save_png(img.crop(box), data / sample / 'test' / t / f'{n:03d}.png')
                    save_png((m[y0:y1, x0:x1] * 255).astype(np.uint8),
                             data / sample / 'ground_truth' / t / f'{n:03d}_mask.png')
                # 정상 바탕: 같은 PCB train_normal에서 골라, 기판을 이 결함 이미지에 맞춰(평행이동 + 180°)
                # 같은 부품 자리를 잘라 온다. export에서 같은 변환을 거꾸로 적용해 바탕 원본에 붙인다.
                chosen = [normals[i] for i in rng.choice(len(normals), a.n_normal, replace=False)]
                transforms = []
                for n, row in enumerate(chosen):
                    b_img = ks.load(row)[0]
                    tf = board_transform(img, b_img)
                    transforms.append({k_: (bool(v) if k_ == 'rot180' else (int(v) if k_ != 'score' else float(v)))
                                       for k_, v in tf.items()})
                    save_png(aligned_crop(b_img, box, tf), data / sample / 'train' / 'good' / f'{n:03d}.png')
                # 3. 생성에 쓸 마스크: 이 결함 모양을 종류별 jitter로 옮긴 것 (결함 이미지 좌표 = 정렬된 바탕 좌표)
                for n in range(a.n_masks):
                    crop_m = jitter_mask(m, t, rng, a.jitter)[y0:y1, x0:x1]
                    if crop_m.any():
                        save_png((crop_m * 255).astype(np.uint8), w / 'generated_mask' / sample / t / f'{n}.png')
                meta[sample] = dict(pcb=pcb, defect_type=t, donor=donor, box=list(box),
                                    normals=[r['image'] for r in chosen], transforms=transforms)
                lines.append(f'{sample}+{t}')
    (w / 'name-anomaly.txt').write_text('\n'.join(lines) + '\n')
    (w / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # 공식 코드가 현재 폴더 기준으로 찾는 경로들
    for name in ('models', 'configs'):
        try:
            (w / name).symlink_to(UPSTREAM / name, target_is_directory=True)
        except OSError:
            if os.name != 'nt':
                raise
            print(f'[경고] Windows에서 {name} 링크를 못 만듦. 실제 학습은 Linux(A100)에서 하세요.')
    cfg = yaml.safe_load((UPSTREAM / CONFIG).read_text())
    cfg['lightning']['trainer']['max_steps'] = a.steps
    (w / 'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'준비 완료: {w}  (샘플 {len(lines)}개, 학습 {a.steps} step)')


# ---------------------------------------------------------------- train / generate (공식 코드)
def upstream(a, w, args):
    env = dict(os.environ, PYTHONPATH=str(UPSTREAM))
    cmd = shlex.split(a.ad_python) + args
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=w, env=env, check=True)


def train(a):
    w = work_dir(a.k, a.seed)
    assert (w / 'name-anomaly.txt').exists(), 'prepare 먼저'
    upstream(a, w, [str(UPSTREAM / 'main.py'), '--spatial_encoder_embedding', '--data_enhance',
                    '--base', 'train.yaml', '-t', '--actual_resume', 'models/ldm/text2img-large/model.ckpt',
                    '-n', 'kshot', '--gpus', '0,', '--init_word', 'anomaly', '--mvtec_path=data'])
    ck = w / 'logs/anomaly-checkpoints/checkpoints'
    missing = [f for f in ('embeddings.pt', 'spatial_encoder.pt') if not (ck / f).exists()]
    assert not missing, f'학습 결과에 {missing} 없음: {ck} 를 확인하세요 (README "처음 실행 점검" 참고)'


def patched_generator(a, w, limit):
    """generate_with_mask.py 사본: 생성 장수 상한과 해상도만 바꾼다. 바꿀 문자열이 없으면 멈춘다."""
    src = (UPSTREAM / 'generate_with_mask.py').read_text()
    src, n = re.subn(r'cnt\s*>\s*500', f'cnt>={limit}', src)
    assert n >= 1, '상한(cnt>500)을 못 찾음: 공식 코드가 바뀌었는지 확인'
    if a.gen_res != 256:
        src, n = re.subn(r'size\s*=\s*256', f'size={a.gen_res}', src)
        assert n >= 1, '해상도(size=256)를 못 찾음: --gen-res 256 으로 돌리거나 공식 코드 확인'
    path = w / 'generate_with_mask_kshot.py'
    path.write_text(src)
    return path


def generate(a):
    w = work_dir(a.k, a.seed)
    meta = json.loads((w / 'meta.json').read_text())
    quota = -(-a.per_type // a.k)                       # 샘플 하나가 만들 최대 장수
    script = patched_generator(a, w, quota + 8)         # 배치(8) 경계에서 멈추므로 여유
    for sample, info in meta.items():
        out = w / 'generated_dataset' / sample / info['defect_type'] / 'image'
        if out.exists() and len(list(out.iterdir())) >= quota and not a.force:
            continue
        upstream(a, w, [str(script), '--data_root=data', f'--sample_name={sample}',
                        f'--anomaly_name={info["defect_type"]}'])


# ---------------------------------------------------------------- export
def numbered(folder):
    return sorted(folder.iterdir(), key=lambda p: int(p.stem))


def export(a):
    ks = KShot(a.k, a.seed, a.data_root)
    w = work_dir(a.k, a.seed)
    meta = json.loads((w / 'meta.json').read_text())
    writer = ManifestWriter('anomalydiffusion', a.k, a.seed, force=a.force)
    normal_rows = {r['image']: r for r in ks.normal_rows}
    for pcb in PCBS:
        for t in TYPES:
            samples = [(s, m) for s, m in meta.items() if m['pcb'] == pcb and m['defect_type'] == t]
            assert len(samples) == a.k
            made = 0
            for j, (sample, info) in enumerate(samples):
                want = a.per_type // a.k + (j < a.per_type % a.k)
                gen_dir = w / 'generated_dataset' / sample / t
                x0, y0, x1, y1 = box = info['box']
                goods = [Image.open(w / 'data' / sample / 'train' / 'good' / f'{n:03d}.png').convert('RGB')
                         for n in range(len(info['normals']))]
                got = 0
                for img_path in numbered(gen_dir / 'image'):
                    if got == want:
                        break
                    gen = Image.open(img_path).convert('RGB')
                    m = np.asarray(Image.open(gen_dir / 'mask' / img_path.name).convert('L')
                                   .resize((x1 - x0, y1 - y0), Image.NEAREST)) > 127
                    if not m.any():
                        continue
                    # 어떤 정상 바탕을 썼는지는 ori/ 이미지와 가장 비슷한 바탕 크롭으로 찾는다
                    ori = np.asarray(Image.open(gen_dir / 'ori' / img_path.name).convert('RGB'), np.float32)
                    err = [np.mean((np.asarray(g.resize(ori.shape[1::-1], LANCZOS), np.float32) - ori) ** 2)
                           for g in goods]
                    bi = int(np.argmin(err))
                    base_row, tf = normal_rows[info['normals'][bi]], info['transforms'][bi]
                    base, _ = ks.load(base_row)
                    # 정렬된 크롭(결함 이미지 좌표) → 바탕 원본 좌표: 180° 되돌리고 같은 창으로 옮긴다
                    if tf['rot180']:
                        gen, m = gen.rotate(180), m[::-1, ::-1]
                    bx0, by0, bx1, by1 = box_t = map_box(box, tf, base.size)
                    W, H = base.size
                    region = np.zeros((H, W), bool)
                    cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, W), min(by1, H)
                    region[cy0:cy1, cx0:cx1] = m[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
                    if not region.any():
                        continue
                    image = paste_back(base, gen, box_t, dilate(region, 2), feather=2)
                    writer.add(image, region, pcb, t, info['donor'], base_row, dict(
                        sample=sample, file=img_path.name, crop_box=list(box_t), align=tf,
                        base_match_mse=float(min(err)),
                        base_match_margin=float(sorted(err)[1] - min(err)) if len(err) > 1 else None,
                        gen_res=a.gen_res, train_steps=a.steps))
                    got += 1
                made += got
                assert got == want, f'{sample}: {want}장 필요, {got}장뿐 (generate 다시 실행)'
            print(f'  {pcb} {t}: {made}장', flush=True)
    print(writer.close(), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['prepare', 'train', 'generate', 'export', 'all'])
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000, help='공식 기본값은 1,000,000. K=1로 먼저 시간을 재고 정한다')
    p.add_argument('--res', type=int, default=512, help='결함 주변 크롭 크기(원본 픽셀)')
    p.add_argument('--gen-res', type=int, default=512, help='생성 해상도. 공식 기본 256')
    p.add_argument('--per-type', type=int, default=40)
    p.add_argument('--n-normal', type=int, default=30, help='샘플마다 넣을 정상 바탕 크롭 수')
    p.add_argument('--n-masks', type=int, default=60, help='샘플마다 만들 생성용 마스크 수')
    p.add_argument('--jitter', type=int, default=None, help='기본: 종류별 (bent·missing 5px, melt·scratch 30px, copy-paste와 같음)')
    p.add_argument('--data-root')
    p.add_argument('--ad-python', default='conda run --no-capture-output -n Anomalydiffusion python',
                   help='공식 코드를 돌릴 파이썬 (setup.sh가 만드는 conda 환경)')
    p.add_argument('--force', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    stages = ['prepare', 'train', 'generate', 'export'] if a.stage == 'all' else [a.stage]
    for s in stages:
        print(f'== {s} K{a.k}_seed{a.seed}', flush=True)
        globals()[s](a)
