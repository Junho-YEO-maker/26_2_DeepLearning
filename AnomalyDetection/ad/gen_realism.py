"""생성 결함이 실제 결함을 얼마나 닮았나: 결함 종류별 패치 특징 정밀도·재현율 (11/16).

생성 모델에서 무엇을 바꿔야 하는지 보려고, AD가 쓰는 패치 특징(WideResNet50, 32×32 격자, photonorm) 공간에서 잰다.
  실제 결함 패치   pool 불량 중 K장 학습 사진을 뺀 나머지(시험셋 아님)의 결함 패치
  정상 패치        train_normal 100장에서 뽑은 20000개
  생성 결함 패치   생성 이미지의 마스크 안 패치
  (패치 라벨은 patch_ad.cell_labels: 결함이 패치의 5% 넘게 덮으면 결함)
지표 (종류별, 특징은 정상 패치 평균·표준편차로 표준화한 거리)
  정밀도  생성 결함 패치 중, 가장 가까운 실제 결함 패치(같은 종류)가 가장 가까운 정상 패치보다 가까운 비율
          → 낮으면 "생성 결함이 실제 결함보다 정상 쪽에 가깝다"(약하거나 엉뚱한 모양)
  재현율  실제 결함 패치 중, 가장 가까운 생성 결함 패치(같은 종류)가 가장 가까운 정상 패치보다 가까운 비율
          → 낮으면 "생성이 실제 결함의 모양 범위를 못 덮는다"
  real    실제 결함 패치끼리(다른 사진) 같은 기준으로 잰 값 = 상한 참고치

    python -m AnomalyDetection.ad.gen_realism --augs sd_lora sd_lora_v3 sd_lora_v3c1024 anomalydiffusion [--data-root ...]
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, TYPES, Data, Item, read_csv
from .features import PatchFeatures, device
from .patch_ad import RES, cell_labels, load_mask, patch_feats


def cells(P, items, data, gen_masks):
    """양성 패치와 그 출처(사진 번호, 종류)."""
    feats, img_id, typ = [], [], []
    for i, (p, it) in enumerate(zip(P, items)):
        lab = cell_labels(load_mask(data, it, gen_masks)).reshape(-1)
        idx = np.where(lab == 1)[0]
        feats.append(p[idx])
        img_id += [i] * len(idx)
        typ += [it.defect_type] * len(idx)
    return np.concatenate(feats).astype(np.float32), np.array(img_id), np.array(typ)


@torch.no_grad()
def nn_dist(Q, B, dev, excl_q=None, excl_b=None):
    """Q 각 행의 B 최근접 거리. excl이 주어지면 같은 사진끼리는 뺀다."""
    if len(B) == 0:
        return np.full(len(Q), np.inf)
    Bt = torch.as_tensor(B, device=dev)
    out = []
    for s in range(0, len(Q), 2048):
        d = torch.cdist(torch.as_tensor(Q[s:s + 2048], device=dev), Bt)
        if excl_q is not None:
            same = torch.as_tensor(excl_q[s:s + 2048], device=dev)[:, None] == torch.as_tensor(excl_b, device=dev)[None]
            d[same] = float('inf')
        out.append(d.min(1).values.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--augs', nargs='+', default=['sd_lora', 'sd_lora_v3', 'sd_lora_v3c1024', 'anomalydiffusion'])
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--data-root')
    a = ap.parse_args()
    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    pool = read_csv(data.splits / 'pool.csv')
    rows = []
    for pcb in a.pcbs:
        nl = data.train_normal(pcb)
        kimgs = {x.image for x in data.train_defects(pcb, a.k, a.seed)}
        real = [Item(r['image'], 1, r['defect_type'], 'real') for r in pool
                if r['category'] == pcb and r['image'] not in kimgs]
        ref = np.mean([data.load(x.image, RES).astype(np.float32).mean((0, 1)) for x in nl], 0)
        Pn = patch_feats(data, ex, dev, nl[::5], ref).reshape(-1, 1024).astype(np.float32)
        rng = np.random.default_rng(0)
        Pn = Pn[rng.choice(len(Pn), 20000, replace=False)]
        mu, sd = Pn.mean(0), Pn.std(0) + 1e-6
        z = lambda X: (X - mu) / sd                                           # noqa: E731
        Bn = z(Pn)
        R, R_img, R_t = cells(patch_feats(data, ex, dev, real, ref), real, data, {})
        R = z(R)
        dn_R = nn_dist(R, Bn, dev)
        for t in TYPES:
            m = R_t == t
            if m.sum() == 0:
                continue
            d_rr = nn_dist(R[m], R[m], dev, R_img[m], R_img[m])
            rows.append(dict(pcb=pcb, aug='real', type=t, n=int(m.sum()), precision=float(np.mean(d_rr < dn_R[m])),
                             recall=float(np.mean(d_rr < dn_R[m]))))
        for aug in a.augs:
            gen_masks = {r['image']: r['mask'] for r in
                         read_csv(data.generated / aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv')}
            gen = data.generated_defects(aug, pcb, a.k, a.seed)
            G, _, G_t = cells(patch_feats(data, ex, dev, gen, ref), gen, data, gen_masks)
            G = z(G)
            dn_G = nn_dist(G, Bn, dev)
            for t in TYPES:
                mg, mr = G_t == t, R_t == t
                if mg.sum() == 0 or mr.sum() == 0:
                    continue
                prec = float(np.mean(nn_dist(G[mg], R[mr], dev) < dn_G[mg]))
                rec = float(np.mean(nn_dist(R[mr], G[mg], dev) < dn_R[mr]))
                rows.append(dict(pcb=pcb, aug=aug, type=t, n=int(mg.sum()), precision=prec, recall=rec))
        print(f'{pcb}: 끝 (실제 결함 패치 {len(R)}개)', flush=True)

    out = ROOT / 'AnomalyDetection/results' / f'gen_realism_K{a.k}_seed{a.seed}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n종류별 정밀도 / 재현율 (PCB 평균). 정밀도: 생성 결함이 정상보다 실제 결함에 가까운 비율, '
          '재현율: 실제 결함이 정상보다 생성 결함에 가까운 비율. real = 실제끼리(상한 참고)')
    print(f'  {"":18s}' + ''.join(f'{t:>16s}' for t in TYPES))
    for aug in ['real'] + a.augs:
        line = f'  {aug:18s}'
        for t in TYPES:
            rs = [r for r in rows if r['aug'] == aug and r['type'] == t]
            line += f'      {np.mean([r["precision"] for r in rs]):.2f} / {np.mean([r["recall"] for r in rs]):.2f}' if rs else ' ' * 16
        print(line, flush=True)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
