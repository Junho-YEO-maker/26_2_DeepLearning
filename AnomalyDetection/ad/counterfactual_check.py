"""지름길 해결 A: 결함을 지운 실제 K장 불량 사진을 "정상" 반례로 학습에 넣는다 (11/16).

defect_removal_check에서 AD가 불량 사진의 촬영 조건(지름길)으로 불량을 맞힌다는 것을 확인했다.
여기서는 학습용 실제 K장 불량 사진 D마다, 결함 영역(마스크 + bent·missing 원래 핀 자리, 40px 넓힘)을
정렬한 train_normal 픽셀로 덮은 사진 CF를 M장 만들어 정상 라벨로 추가한다.
"불량 촬영 분위기 자체는 불량이 아니다"를 직접 보여 주는 반례다. 학습 데이터(train_K, train_normal)만 쓴다.

조건 (정상 쪽 × 불량 쪽)
  정상: train_normal  |  + CF(자연 비율, PCB당 4K·M장)  |  + CF 25%(정상 배치의 약 25%가 되도록 복제)
  불량: 실제 K장  |  실제 K장 + 생성(--aug, 전부)
평가
  원래 시험셋 FNR@FPR1% (전체, 종류별)
  결함 제거 진단 판정 비율(pool 불량, defect_removal_check 캐시 재사용): D / DR(지름길) / ND(결함 자체)
전처리는 canon(prep.py), 헤드 시드 5개 평균.

    python -m AnomalyDetection.ad.counterfactual_check --aug sd_lora_v3 --k 1 --seed 0 [--data-root ...]
"""
import argparse
import csv

import numpy as np
import torch

from .data import PCBS, ROOT, TYPES, Data
from .defect_removal_check import region_in
from .eval import evaluate, threshold_at_fpr
from .features import PatchFeatures, device
from .models import EmbeddingHead
from .prep import embedder

import sys
sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import TYPE_ID, KShot, paste_back  # noqa: E402

MODELS = ('devnet', 'mlp')


def counterfactuals(ks, pcb, m_per, px, rng):
    """실제 K장 불량마다 결함을 지운 사진 m_per장 (서로 다른 train_normal로 덮음)."""
    out, src = [], []
    normals = ks.normals(pcb)
    for t in TYPES:
        for row in ks.defects(pcb, t):
            D, ids = ks.load(row)
            m = ids == TYPE_ID[t]
            W, H = D.size
            for i in rng.choice(len(normals), m_per, replace=False):
                N, _ = ks.load(normals[i])
                reg, N_al, _ = region_in(D, N, m, t, px)
                out.append(paste_back(D, N_al, (0, 0, W, H), reg, feather=3))
                src.append(row['image'])
    return out, src


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prep', default='canon')
    ap.add_argument('--m-per', type=int, default=5, help='실제 불량 1장당 반례 수')
    ap.add_argument('--remove-dilate', type=int, default=40)
    ap.add_argument('--diag-tag', default='_canon_dil40', help='defect_removal_check 캐시 접미사 (같은 prep이어야 함)')
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--data-root')
    a = ap.parse_args()
    assert a.diag_tag.startswith(f'_{a.prep}'), '진단 캐시와 prep이 같아야 한다'
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ks = KShot(a.k, a.seed, a.data_root)
    ex = PatchFeatures().to(dev)
    rows, n_cf = [], {}
    for pcb in a.pcbs:
        emb_items, emb_imgs = embedder(data, ex, dev, pcb, 512, a.prep)
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        Xt, Xv, Xn, Xr, Xg = (emb_items(x) for x in (test, val, nl, real, gen))
        cf_imgs, _ = counterfactuals(ks, pcb, a.m_per, a.remove_dilate, np.random.default_rng(7017))
        Xcf = emb_imgs(cf_imgs)
        n_cf[pcb] = len(Xcf)
        rep = max(int(round(len(Xn) / 3 / len(Xcf))), 1)          # CF가 정상 풀의 약 25%
        z = np.load(ROOT / 'Dataset/cache/defect_removal' / f'{pcb}_{a.aug}_K{a.k}_seed{a.seed}{a.diag_tag}.npz')
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]
        normals = {'normal': Xn, '+CF': np.concatenate([Xn, Xcf]), '+CF25%': np.concatenate([Xn, np.repeat(Xcf, rep, 0)])}
        defects = {'real': Xr, f'real+{a.aug}': np.concatenate([Xr, Xg])}
        for m in MODELS:
            for nk, XN in normals.items():
                for dk, XA in defects.items():
                    for hs in range(a.head_seeds):
                        h = EmbeddingHead(m, seed=hs).fit(XN, XA)
                        r = evaluate(h.score(Xt), labels, dtypes, h.score(Xv))
                        thr = threshold_at_fpr(h.score(Xv), 0.01)
                        diag = {k_: float(np.mean(h.score(z[k_]) > thr)) for k_ in ('D', 'DR', 'ND', 'NR')}
                        rows.append(dict(pcb=pcb, model=m, normal=nk, defect=dk, head_seed=hs, fnr=r['fnr_at_fpr1'],
                                         auroc=r['auroc'], **{f'fnr_{t}': r['per_type'][t][1] for t in TYPES},
                                         **{f'n_{t}': r['per_type'][t][0] for t in TYPES}, **diag))
        print(f'{pcb}: 반례 {len(Xcf)}장 (25% 조건은 {rep}배 복제)', flush=True)

    out = ROOT / 'AnomalyDetection/results' / f'counterfactual_{a.aug}_K{a.k}_seed{a.seed}_{a.prep}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wmean(rs, t):     # 종류별 FNR: PCB 합침(불량 수 가중), 시드 평균
        num = sum(r[f'n_{t}'] * r[f'fnr_{t}'] for r in rs if r[f'n_{t}'])
        den = sum(r[f'n_{t}'] for r in rs if r[f'n_{t}'])
        return num / max(den, 1)

    print(f'\n시험셋 FNR@FPR1% (PCB 평균) | 종류별 FNR (PCB 합침) | 진단 판정 비율 D / DR(지름길) / ND(결함 자체) / NR(붙이기 대조)'
          f'\nprep={a.prep}, 헤드 시드 {a.head_seeds}개 평균. FNR은 낮을수록, ND는 높을수록, DR은 낮을수록 좋다')
    for m in MODELS:
        for nk in ('normal', '+CF', '+CF25%'):
            for dk in ('real', f'real+{a.aug}'):
                rs = [r for r in rows if r['model'] == m and r['normal'] == nk and r['defect'] == dk]
                fnr = np.mean([np.mean([r['fnr'] for r in rs if r['pcb'] == p]) for p in a.pcbs])
                print(f'  {m:6s} 정상={nk:7s} 불량={dk:18s} FNR {fnr:.3f} | ' +
                      ' '.join(f'{t} {wmean(rs, t):.2f}' for t in TYPES) + ' | ' +
                      ' '.join(f"{k_} {np.mean([r[k_] for r in rs]):.2f}" for k_ in ('D', 'DR', 'ND', 'NR')), flush=True)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
