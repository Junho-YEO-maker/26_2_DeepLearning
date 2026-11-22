"""우리 모델을 SegAD 규칙(segad_protocol)으로 돌린다 (11/20, 2·3단계). 공개 방법(external_eval)과 같은 seed에서 같은 학습 불량 10장.

모델 (모두 고정 WRN50 위, 512)
  patchcore  정상만 (seed 무관, 한 번 학습 → seed마다 남은 시험으로 잼)
  devnet     우리 단순판: 사진 전체 임베딩(평균 + 최대) 위 deviation 머리
  seg        위치 학습 (focal ×10 + dice, 2000 step)
  seg_aug    seg + 촬영 조건 무작위화
지표: 공식 시험(남은 불량 90 + 정상) 이미지 AUROC · 픽셀 AUROC · AUPRO, 공정 시험셋 fair_ad.metrics.
SegAD 논문 표(PCB별 이미지 AUROC, seed 10개 평균): DRA 92.1/91.6/77.7, DevNet 94.4/89.5/73.4, EfficientAD+SegAD 99.5/99.7/98.4.

    python -m AnomalyDetection.ad.segad_eval --seeds 0 1 2 3 4
결과: AnomalyDetection/results/official/segad_protocol.csv, 점수 segad_{pcb}.npz
"""
import argparse
import csv
import time

import numpy as np
import torch

from . import fair_ad
from . import segad_protocol as sp
from .data import PCBS, ROOT
from .features import PatchFeatures, device, embed, to_tensor
from .local_ad import Backbone, SegModel, aggregate
from .models import EmbeddingHead
from .official_eval import EVAL, OUT, RES, CanonOff, PatchCoreMap, image_auroc, mask_geom, pixel_metrics, to_eval
from .patch_ad import load_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+', default=['patchcore', 'devnet', 'seg', 'seg_aug'])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = sp.data()
    maps = sp.build_layout(d)
    dev = device()
    ex = PatchFeatures().to(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, scores = [], {}
    for pcb in a.pcbs:
        t0 = time.time()
        full = d.test(pcb)
        nl = d.train_normal(pcb)
        load = lambda items: np.stack([d.load(x.image, RES) for x in items])  # noqa: E731
        An, Af = load(nl), load(full)
        gt_full = np.stack([mask_geom(d, x, RES, 0, EVAL) for x in full])
        idx_of = {x.image: i for i, x in enumerate(full)}
        print(f'{pcb}: 정상 {len(nl)}, 시험 {len(full)} ({time.time() - t0:.0f}s)', flush=True)

        def add(model, seed, test, st, fs, sf, mt=None):
            labels = [x.label for x in test]
            y = np.asarray(labels)
            r = dict(pcb=pcb, model=model, seed=seed, image_auroc=image_auroc(st, labels), pixel_auroc=np.nan, aupro=np.nan)
            if mt is not None:
                ii = [idx_of[x.image] for x in test]
                r['pixel_auroc'], r['aupro'] = pixel_metrics(mt, gt_full[ii])
            r.update(fair_ad.metrics({'test': st, **sf}, st[y == 0], labels, [x.defect_type for x in test]))
            rows.append(r)
            scores[f'{pcb}|{model}|{seed}|test'] = np.asarray(st, np.float32)
            for p, v in sf.items():
                scores[f'{pcb}|{model}|{seed}|{p}'] = np.asarray(v, np.float32)
            print(f'  {pcb} {model} seed {seed}: 이미지 AUROC {r["image_auroc"]:.3f} 픽셀 {r["pixel_auroc"]:.3f} '
                  f'AUPRO {r["aupro"]:.3f} | 공정@5 {r["fair_fnr5"]:.3f} 공정 AUROC {r["fair_auroc"]:.3f} '
                  f'| 결함 지운 사진 불량 판정 {r["v_DR"]:.2f}, 붙인 결함 검출 {r["v_ND"]:.2f}', flush=True)

        splits = {s: sp.split(d, maps, pcb, s) for s in a.seeds}
        fs_full = None
        if 'patchcore' in a.models:
            pc = PatchCoreMap(ex, dev, coreset_ratio=0.01).fit(An)
            mt_full = pc.maps(Af)
            st_full = mt_full.reshape(len(mt_full), -1).max(1)
            fair_ad.CACHE = sp.FAIR_CACHE
            fs_full = fair_ad.fair_set(d, pcb)
            sf_full = {p: pc.maps(fs_full[p]).reshape(len(fs_full[p]), -1).max(1) for p in fair_ad.PARTS}
            ev_full = to_eval(mt_full)
            for s, (train_def, test, fs) in splits.items():
                ii = [idx_of[x.image] for x in test]
                keep = np.isin(fs_full['image'], fs['image'])
                add('patchcore', s, test, st_full[ii], fs, {p: v[keep] for p, v in sf_full.items()}, ev_full[ii])
            del pc
            torch.cuda.empty_cache()

        if 'devnet' in a.models:
            c = CanonOff(d, pcb, '1cls')

            def cemb(arrs):
                out = []
                for i in range(0, len(arrs), 16):
                    x = to_tensor(np.stack([c.fix(v) for v in arrs[i:i + 16]])).to(dev)
                    out.append(embed(ex(x)).float().cpu().numpy())
                return np.concatenate(out)

            Xn = cemb([c.item(x.image) for x in nl])
            Xf_all = cemb([c.item(x.image) for x in full])
            for s, (train_def, test, fs) in splits.items():
                Xr = Xf_all[[idx_of[x.image] for x in train_def]]
                h = EmbeddingHead('devnet', seed=s).fit(Xn, Xr)
                ii = [idx_of[x.image] for x in test]
                add('devnet', s, test, h.score(Xf_all[ii]), fs, {p: h.score(cemb(fs[p + '_canon'])) for p in fair_ad.PARTS})

        segs = [m for m in ('seg', 'seg_aug') if m in a.models]
        if segs:
            bb = Backbone(ex).to(dev).eval()
            ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
            for s, (train_def, test, fs) in splits.items():
                Ar = Af[[idx_of[x.image] for x in train_def]]
                Mr = np.stack([load_mask(d, x, {}) for x in train_def])
                ii = [idx_of[x.image] for x in test]
                for model in segs:
                    t1 = time.time()
                    sm = SegModel(bb, ref, dev, seed=s, steps=a.steps, aug=model == 'seg_aug').fit(An, Ar, Mr)
                    mt = sm.score(Af[ii])
                    sf = {p: aggregate(sm.score(fs[p]), 'top10') for p in fair_ad.PARTS}
                    add(model, s, test, aggregate(mt, 'top10'), fs, sf, to_eval(mt))
                    print(f'    ({time.time() - t1:.0f}s)', flush=True)
                    del sm
            torch.cuda.empty_cache()
        np.savez_compressed(OUT / f'segad_{pcb}.npz', **{k: v for k, v in scores.items() if k.startswith(pcb + '|')})

    out = OUT / 'segad_protocol.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\nPCB별 (seed 평균)')
    for model in dict.fromkeys(r['model'] for r in rows):
        line = ' '.join(f'{p} {np.mean([r["image_auroc"] for r in rows if (r["model"], r["pcb"]) == (model, p)]):.3f}'
                        for p in a.pcbs)
        rs = [r for r in rows if r['model'] == model]
        print(f'  {model:9s} 이미지 AUROC {line} | 공정@5 {np.mean([r["fair_fnr5"] for r in rs]):.3f} '
              f'결함 지운 사진 불량 {np.mean([r["v_DR"] for r in rs]):.2f} 붙인 결함 검출 {np.mean([r["v_ND"] for r in rs]):.2f}')
    print('저장:', out)


if __name__ == '__main__':
    main()
