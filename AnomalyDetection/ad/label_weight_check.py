"""실제 레이블은 강하게, 생성 레이블은 약하게: DevNet/MLP에 생성 불량을 보조 신호로 넣는다 (11/16).

지금까지는 생성 불량(PCB당 약 160장)을 실제 K장(PCB당 4장)과 같은 무게로 섞어서, 불량 배치의 약 97%가 생성이었다.
여기서는 불량 절반 배치 안에서 생성 비율 r을 고정하고(나머지는 실제 K장), 생성 불량의 여유값(margin)도 따로 둔다.
  r      0(실제만) · 0.05 · 0.1 · 0.25 · 0.5 · 자연 비율(지금까지 방식)
  margin 생성 불량을 정상보다 몇 표준편차 위로 밀지 (DevNet만): 5(실제와 같음) · 2(약하게)
         MLP는 margin 대신 생성 불량의 BCE 목표값을 1(같음) · 0.7(약하게)로 둔다.
평가: 원래 시험셋 FNR@FPR1%·5%, 종류별 FNR, 결함 제거 진단 D/DR(지름길)/ND(결함 자체) (canon, 40px 캐시 재사용).
전처리 canon, 헤드 시드 5개.

    python -m AnomalyDetection.ad.label_weight_check --aug sd_lora_v3 [--data-root ...]
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F

from .data import PCBS, ROOT, TYPES, Data
from .eval import evaluate, threshold_at_fpr
from .features import PatchFeatures, device
from .prep import embedder

RATIOS = (0.0, 0.05, 0.1, 0.25, 0.5, 'natural')


class MixedHead:
    """EmbeddingHead와 같은 망·표준화·학습 설정. 불량 절반 배치를 실제/생성으로 나눠 뽑는다."""

    def __init__(self, kind, seed=0, steps=600, batch=64, lr=1e-3):
        self.kind, self.seed, self.steps, self.batch, self.lr = kind, seed, steps, batch, lr

    def fit(self, Xn, Xr, Xg, r, weak):
        Xn = torch.as_tensor(Xn, dtype=torch.float32)
        self.mu, self.sd = Xn.mean(0), Xn.std(0) + 1e-6
        Zn = (Xn - self.mu) / self.sd
        Zr = (torch.as_tensor(Xr, dtype=torch.float32) - self.mu) / self.sd
        Zg = (torch.as_tensor(Xg, dtype=torch.float32) - self.mu) / self.sd
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(Zn.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(256, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        half = self.batch // 2
        for _ in range(self.steps):
            if r == 'natural':                       # 지금까지 방식: 실제+생성을 한 풀로
                n_g = int((torch.randint(len(Zr) + len(Zg), (half,), generator=g) >= len(Zr)).sum())
            else:
                n_g = int(round(r * half))
            n_r = half - n_g
            x = torch.cat([Zn[torch.randint(len(Zn), (half,), generator=g)],
                           Zr[torch.randint(len(Zr), (n_r,), generator=g)],
                           Zg[torch.randint(len(Zg), (n_g,), generator=g)]])
            is_g = torch.cat([torch.zeros(half + n_r), torch.ones(n_g)])
            y = torch.cat([torch.zeros(half), torch.ones(half)])
            out = self.net(x).squeeze(1)
            if self.kind == 'mlp':
                tgt = torch.where(is_g > 0, torch.full_like(y, 0.7 if weak else 1.0), y)
                loss = F.binary_cross_entropy_with_logits(out, tgt)
            else:
                ref = torch.randn(5000, generator=g)
                d = (out - ref.mean()) / ref.std()
                margin = torch.where(is_g > 0, torch.full_like(y, 2.0 if weak else 5.0), torch.full_like(y, 5.0))
                loss = ((1 - y) * d.abs() + y * F.relu(margin - d)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X):
        Z = (torch.as_tensor(X, dtype=torch.float32) - self.mu) / self.sd
        return self.net(Z).squeeze(1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prep', default='canon')
    ap.add_argument('--diag-tag', default='_canon_dil40')
    ap.add_argument('--diag-aug', help='진단 캐시(D/DR/ND/NR)를 만든 aug (기본 = --aug). 진단 사진은 aug와 무관해서 v3 캐시를 그대로 써도 된다')
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--data-root')
    a = ap.parse_args()
    assert a.diag_tag.startswith(f'_{a.prep}')
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ex = PatchFeatures().to(dev)
    rows = []
    conds = [(r, w) for r in RATIOS for w in ((False,) if r == 0.0 else (False, True))]
    for pcb in a.pcbs:
        emb_items, _ = embedder(data, ex, dev, pcb, 512, a.prep)
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real, gen = data.train_defects(pcb, a.k, a.seed), data.generated_defects(a.aug, pcb, a.k, a.seed)
        Xt, Xv, Xn, Xr, Xg = (emb_items(x) for x in (test, val, nl, real, gen))
        z = np.load(ROOT / 'Dataset/cache/defect_removal' / f'{pcb}_{a.diag_aug or a.aug}_K{a.k}_seed{a.seed}{a.diag_tag}.npz')
        labels, dtypes = [x.label for x in test], [x.defect_type for x in test]
        y = np.array(labels)
        for m in ('devnet', 'mlp'):
            for r, weak in conds:
                for hs in range(a.head_seeds):
                    h = MixedHead(m, seed=hs).fit(Xn, Xr, Xg, r, weak)
                    st, sv = h.score(Xt), h.score(Xv)
                    ev = evaluate(st, labels, dtypes, sv)
                    thr5 = threshold_at_fpr(st[y == 0], 0.05)
                    thr = threshold_at_fpr(sv, 0.01)
                    diag = {k_: float(np.mean(h.score(z[k_]) > thr)) for k_ in ('D', 'DR', 'ND', 'NR')}
                    rows.append(dict(pcb=pcb, model=m, ratio=r, weak=int(weak), head_seed=hs,
                                     fnr=ev['fnr_at_fpr1'], fnr5=float(np.mean(st[y == 1] <= thr5)), auroc=ev['auroc'],
                                     **{f'fnr_{t}': ev['per_type'][t][1] for t in TYPES},
                                     **{f'n_{t}': ev['per_type'][t][0] for t in TYPES}, **diag))
        print(f'{pcb}: 끝 (실제 {len(Xr)}장, 생성 {len(Xg)}장)', flush=True)

    out = ROOT / 'AnomalyDetection/results' / f'label_weight_{a.aug}_K{a.k}_seed{a.seed}_{a.prep}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wmean(rs, t):
        num = sum(r[f'n_{t}'] * r[f'fnr_{t}'] for r in rs if r[f'n_{t}'])
        den = sum(r[f'n_{t}'] for r in rs if r[f'n_{t}'])
        return num / max(den, 1)

    print(f'\n생성 비율 r(불량 배치 중 생성 몫) × 약하게(weak: DevNet margin 2, MLP 목표 0.7). prep={a.prep}, 시드 {a.head_seeds}개 평균'
          '\nFNR@1%·5%는 PCB 평균(낮을수록 좋음), 종류별은 PCB 합침, DR 낮을수록·ND 높을수록 좋음')
    for m in ('devnet', 'mlp'):
        for r, weak in conds:
            rs = [x for x in rows if x['model'] == m and x['ratio'] == r and x['weak'] == int(weak)]
            f1 = np.mean([np.mean([x['fnr'] for x in rs if x['pcb'] == p]) for p in a.pcbs])
            f5 = np.mean([np.mean([x['fnr5'] for x in rs if x['pcb'] == p]) for p in a.pcbs])
            print(f'  {m:6s} r={str(r):7s} {"weak" if weak else "same"} | FNR@1 {f1:.3f} @5 {f5:.3f} | ' +
                  ' '.join(f'{t} {wmean(rs, t):.2f}' for t in TYPES) + ' | ' +
                  ' '.join(f"{k_} {np.mean([x[k_] for x in rs]):.2f}" for k_ in ('D', 'DR', 'ND')), flush=True)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
