"""공개 지도 학습 방법을 공식 코드 그대로 돌려 우리 공정한 시험에 넣는다 (11/20, 3단계).

방법 (공식 코드, /home/junho/external 아래에 내려받음. 우리 쪽에서 코드를 고치지 않고 불러 쓴다)
  dra     DRA (Ding et al., CVPR 2022)              github.com/Choubo/DRA                     기본값: 448, ResNet18, 30 epoch
  devnet  DevNet 이미지판 (Pang et al., 상위 K 패치)  github.com/Choubo/deviation-network-image  기본값: 448, ResNet18, 50 epoch
  (SegAD 논문이 같은 두 방법을 "공식 코드, n_anomaly 10, 30·50 epoch"로 VisA PCB에 돌린 수치가 있다 → 재현 점검)
규칙: segad_protocol (공식 1cls + 시험 불량 10장을 학습으로, 공식 코드와 같은 고르기). seed 0–4.
점수: 공식 코드 eval과 같은 방식 (DRA = −정상 머리 + 본 불량 머리 + 가짜 불량 머리 + 기준 비교 머리, 기준 정상 5장은 시험 배치마다 새로 뽑음;
      DevNet = 점수망 출력). 입력은 우리 모델과 같은 512 배열을 공식 전처리(448로 줄이기, ImageNet 정규화)에 넣는다.
지표: 공식 시험(남은 불량 90 + 정상) 이미지 AUROC, 공정 시험셋 fair_ad.metrics (v_* 임계값은 시험 정상으로).

    python -m AnomalyDetection.ad.external_eval --method dra --seeds 0 1 2 3 4
결과: AnomalyDetection/results/external/{method}.csv, 점수 {method}_{pcb}_seed{s}.npz
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from . import fair_ad
from . import segad_protocol as sp
from .data import PCBS, ROOT
from .official_eval import image_auroc

REPOS = {'dra': Path('/home/junho/external/DRA'), 'devnet': Path('/home/junho/external/deviation-network-image')}
OUT = ROOT / 'AnomalyDetection/results/external'


def make_args(method, pcb, seed, exp):
    c = dict(batch_size=48, steps_per_epoch=20, ramdn_seed=seed, workers=4, dataset_root=str(sp.LAYOUT),
             experiment_dir=str(exp), classname=pcb, img_size=448, n_scales=2, backbone='resnet18', criterion='deviation',
             topk=0.1, cuda=True, no_cuda=False)
    if method == 'dra':
        return SimpleNamespace(**c, epochs=30, cont_rate=0.0, test_threshold=0, test_rate=0.0, dataset='mvtecad',
                               savename='model.pkl', nAnomaly=10, know_class=None, pretrain_dir=None, total_heads=4, nRef=5,
                               outlier_root=None)
    return SimpleNamespace(**c, epochs=50, weight_name='model.pkl', n_anomaly=10)


@torch.no_grad()
def scorer(method, trainer, transform):
    model = trainer.model.eval()

    @torch.no_grad()
    def score(arrs, bs=32):
        out = []
        for i in range(0, len(arrs), bs):
            x = torch.stack([transform(Image.fromarray(a)) for a in arrs[i:i + bs]]).cuda()
            if method == 'devnet':
                out.append(model(x.float()).view(-1).cpu().numpy())
                continue
            try:
                ref = next(trainer.ref)['image']
            except StopIteration:
                trainer.ref = iter(trainer.ref_loader)
                ref = next(trainer.ref)['image']
            o = model(torch.cat([ref.cuda(), x]), None)
            out.append((-o[0] + o[1] + o[2] + o[3]).cpu().numpy())
        return np.concatenate(out)
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, choices=list(REPOS))
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, help='점검용으로 줄일 때만')
    a = ap.parse_args()
    sys.path.insert(0, str(REPOS[a.method]))
    # 호환: 공식 코드는 PyTorch 1.x 기준이라 Sampler.__init__(dataset)을 부른다. 2.x에는 그 인자가 없어 무시하게만 한다(동작 같음)
    torch.utils.data.Sampler.__init__ = lambda self, *args, **kw: None
    import train as T                                  # noqa: E402  (공식 코드의 train.py)
    d = sp.data()
    maps = sp.build_layout(d)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pcb in a.pcbs:
        for seed in a.seeds:
            t0 = time.time()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_def, test, fs = sp.split(d, maps, pcb, seed)
            exp = OUT / 'runs' / f'{a.method}_{pcb}_seed{seed}'
            exp.mkdir(parents=True, exist_ok=True)
            args = make_args(a.method, pcb, seed, exp)
            if a.epochs:
                args.epochs = a.epochs
            trainer = T.Trainer(args)
            ds = trainer.train_loader.dataset
            got = {maps[pcb][ds.images[i]] for i in ds.outlier_idx}
            assert got == {x.image for x in train_def}, '공식 코드가 고른 10장이 우리 고르기와 다름'
            if a.method == 'dra':
                trainer.model = trainer.model.to('cuda')
                trainer.criterion = trainer.criterion.to('cuda')
                for ep in range(args.epochs):
                    trainer.training(ep)
            else:
                for ep in range(args.epochs):
                    trainer.train(ep)
            score = scorer(a.method, trainer, trainer.test_loader.dataset.transform)
            At = np.stack([d.load(x.image, 512) for x in test])
            labels = [x.label for x in test]
            st = score(At)
            sf = {p: score(fs[p]) for p in fair_ad.PARTS}
            y = np.asarray(labels)
            m = fair_ad.metrics({'test': st, **sf}, st[y == 0], labels, [x.defect_type for x in test])
            r = dict(method=a.method, pcb=pcb, seed=seed, image_auroc=image_auroc(st, labels), **m)
            rows.append(r)
            np.savez_compressed(OUT / f'{a.method}_{pcb}_seed{seed}.npz', test=st, **sf,
                                images=np.array([x.image for x in test]), train=np.array(sorted(got)))
            print(f'{a.method} {pcb} seed {seed}: 이미지 AUROC {r["image_auroc"]:.3f} | 공정@5 {r["fair_fnr5"]:.3f} '
                  f'공정 AUROC {r["fair_auroc"]:.3f} | 결함 지운 사진 불량 판정 {r["v_DR"]:.2f}, 붙인 결함 검출 {r["v_ND"]:.2f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            del trainer
            torch.cuda.empty_cache()
    out = OUT / f'{a.method}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for pcb in a.pcbs:
        rs = [r for r in rows if r['pcb'] == pcb]
        print(f'{a.method} {pcb}: 이미지 AUROC {np.mean([r["image_auroc"] for r in rs]):.3f} ± {np.std([r["image_auroc"] for r in rs]):.3f}, '
              f'공정@5 {np.mean([r["fair_fnr5"] for r in rs]):.3f}')
    print('저장:', out)


if __name__ == '__main__':
    main()
