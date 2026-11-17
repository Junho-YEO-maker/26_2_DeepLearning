"""생성 샘플 선별 실험: DevNet에서 생성 증강이 손해인 문제 (13.5절 뒤, 11/16).

가설: 생성 160장(PCB당) 중 상당수가 AD 특징 공간에서 자기 바탕 정상과 거의 같아서, "불량 라벨이 붙은 정상"처럼
작용해 DevNet의 정상 점수를 끌어올리고 FPR 1% 임계값을 높인다(→ FNR 증가).

선별 기준(실험 전에 고정): edit = 1 − cos(생성 임베딩, 자기 바탕 정상 임베딩).
  결함이 특징 공간에서 얼마나 보이는지만 재며, train_normal과 생성 이미지만 쓴다(시험셋·val 미사용).
비교(생성 장수는 PCB당):
  none    실제 K장만
  all     생성 전부(160)
  rand40  종류별 무작위 10장(40)  ← 장수만 줄인 대조군
  top40   종류별 edit 상위 10장(40) ← 제안 방법
임베딩은 photonorm(이미지마다 채널 평균을 그 PCB train_normal 평균으로 맞춤, shortcut_check와 같은 계산).
헤드(DevNet·MLP) 학습 시드 5개의 FNR@FPR1%를 평균한다(K 샘플 seed는 하나).

    python -m AnomalyDetection.ad.gen_select --augs sd_lora sd_lora_v3 --k 1 --seed 0 [--data-root ...]
결과: AnomalyDetection/results/gen_select_K{k}_seed{s}.csv (행 = PCB × aug × 변형 × 모델 × 헤드 시드)
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, TYPES, Data
from .eval import evaluate
from .features import PatchFeatures, device, embed, to_tensor
from .models import EmbeddingHead

VARIANTS = ('all', 'rand40', 'top40')


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def photonorm_embed(data, ex, dev, items, ref, res, bs=16):
    out = []
    for i in range(0, len(items), bs):
        arr = []
        for it in items[i:i + bs]:
            a = data.load(it.image, res).astype(np.float32)
            arr.append(np.clip(a - a.mean((0, 1)) + ref, 0, 255).astype(np.uint8))
        out.append(embed(ex(to_tensor(np.stack(arr)).to(dev))).float().cpu().numpy())
    return np.concatenate(out)


def select(variant, types, edit, per_type=10, seed=0):
    idx = np.arange(len(types))
    if variant == 'all':
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for t in TYPES:
        ti = idx[types == t]
        if variant == 'rand40':
            keep += list(rng.choice(ti, min(per_type, len(ti)), replace=False))
        else:
            keep += list(ti[np.argsort(-edit[ti])[:per_type]])
    return np.sort(np.array(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--augs', nargs='+', default=['sd_lora_v3'])
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--prep', default='photonorm', help='임베딩 전처리 (prep.py: none, photonorm, canon)')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(8)
    from .prep import embedder

    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    rows = []

    def fit_eval(pcb, aug, variant, Xn, Xa, Xt, Xv, test, n_gen, mean_edit):
        for kind in ('devnet', 'mlp'):
            for hs in range(a.head_seeds):
                h = EmbeddingHead(kind, seed=hs).fit(Xn, Xa)
                m = evaluate(h.score(Xt), [x.label for x in test], [x.defect_type for x in test], h.score(Xv))
                rows.append(dict(pcb=pcb, aug=aug, variant=variant, model=kind, head_seed=hs, n_gen=n_gen,
                                 mean_edit=mean_edit, fnr_at_fpr1=m['fnr_at_fpr1'], auroc=m['auroc']))

    for pcb in a.pcbs:
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real = data.train_defects(pcb, a.k, a.seed)
        emb_items, _ = embedder(data, ex, dev, pcb, a.res, a.prep)
        Xt, Xv, Xn, Xr = (emb_items(x) for x in (test, val, nl, real))
        fit_eval(pcb, 'none', 'none', Xn, Xr, Xt, Xv, test, 0, float('nan'))
        nidx = {it.image: i for i, it in enumerate(nl)}
        for aug in a.augs:
            gen = data.generated_defects(aug, pcb, a.k, a.seed)
            with open(data.generated / aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv', newline='', encoding='utf-8-sig') as f:
                base_of = {r['image']: r['base_normal_image'] for r in csv.DictReader(f)}
            Xg = emb_items(gen)
            b = np.array([nidx[base_of[g.image]] for g in gen])
            edit = 1 - (unit(Xg) * unit(Xn[b])).sum(1)
            types = np.array([g.defect_type for g in gen])
            for v in VARIANTS:
                s = select(v, types, edit)
                fit_eval(pcb, aug, v, Xn, np.concatenate([Xr, Xg[s]]), Xt, Xv, test, len(s), float(edit[s].mean()))
                print(f'{pcb} {aug:14s} {v:6s} n={len(s):3d} edit={edit[s].mean():.4f}  ' + '  '.join(
                    f"{k}={np.mean([r['fnr_at_fpr1'] for r in rows if r['pcb'] == pcb and r['aug'] == aug and r['variant'] == v and r['model'] == k]):.3f}"
                    for k in ('devnet', 'mlp')), flush=True)
        print(f'{pcb} none           none   n=  0              ' + '  '.join(
            f"{k}={np.mean([r['fnr_at_fpr1'] for r in rows if r['pcb'] == pcb and r['variant'] == 'none' and r['model'] == k]):.3f}"
            for k in ('devnet', 'mlp')), flush=True)

    suffix = '' if a.prep == 'photonorm' else f'_{a.prep}'      # photonorm은 예전 결과 파일 이름 그대로
    out = ROOT / 'AnomalyDetection/results' / f'gen_select_K{a.k}_seed{a.seed}{suffix}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f'\nFNR@FPR1% (PCB {len(a.pcbs)}개 평균, 헤드 시드 {a.head_seeds}개 평균. 괄호는 PCB별)')
    for aug, v in [('none', 'none')] + [(g, v) for g in a.augs for v in VARIANTS]:
        line = f'  {aug:14s} {v:6s}'
        for k in ('devnet', 'mlp'):
            per = [np.mean([r['fnr_at_fpr1'] for r in rows if r['pcb'] == p and r['aug'] == aug and r['variant'] == v
                            and r['model'] == k]) for p in a.pcbs]
            line += f'  {k}={np.mean(per):.3f} (' + '/'.join(f'{x:.2f}' for x in per) + ')'
        print(line)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
