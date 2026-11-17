"""결함 제거 진단: AD(특징 임베딩 + DevNet)가 실제 불량 사진에서 "결함"을 보는가, "사진 전체의 차이"를 보는가.

실제 불량 사진 D 하나마다 같은 PCB의 train_normal N 하나를 기판 정렬해서 두 가지를 만든다.
  DR (결함 제거)   D에서 결함 영역만 정렬한 N의 픽셀로 바꾼 것. 결함은 없고 D의 촬영 조건(밝기·위치·초점…)은 그대로.
  ND (정상+결함)   N에 D의 결함 영역 픽셀만 그대로 붙인 것. 결함은 실제 그대로, 촬영 조건은 정상.
결함 영역 = 결함 마스크(+ bent·missing은 원래 핀 자리, common/kshot.change_region)를 8px 넓힌 것.
그리고 photonorm 임베딩에서
  novelty  가장 가까운 train_normal까지의 거리 / 정상끼리 최근접 거리(중앙값)   (ND는 자기 바탕 N 제외)
  DevNet   실제 K장(train_K{k}_seed{s})으로 학습한 DevNet이 불량으로 판정하는 비율(임계값: val_normal FPR 1%)
을 D, DR, ND, 그리고 생성(--aug)에 대해 잰다.
  DR이 D만큼 "불량"이면 → AD는 결함이 아니라 사진 전체 차이(지름길)를 본다.
  ND가 정상처럼 보이면 → 진짜 결함을 정상 사진에 붙여도 AD가 못 본다(생성이 아무리 좋아도 DevNet 이득이 어렵다).
진단 대상 D는 pool.csv의 불량 중 학습 K장을 뺀 것(test·val 미사용).

    python -m AnomalyDetection.ad.defect_removal_check --k 1 --seed 0 --aug sd_lora_v3 [--data-root ...]
"""
import argparse
import csv
import sys

import numpy as np
import torch
from PIL import Image

from .data import PCBS, ROOT, Data, read_csv
from .eval import threshold_at_fpr
from .features import PatchFeatures, device
from .gen_select import unit
from .models import EmbeddingHead
from .prep import PREPS, embedder

sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import (STRUCTURE_TYPES, TYPE_ID, KShot, aligned_donor, board_transform,  # noqa: E402
                          change_region, dilate, paste_back)


def region_in(target_img, other_img, mask, defect_type, px=8):
    """target 좌표의 결함 영역. other를 target에 정렬해 bent·missing의 원래 핀 자리까지 넣고 px만큼 넓힌다.
    돌아가는 값: (영역 bool, target 좌표로 옮긴 other 이미지)."""
    tf = board_transform(other_img, target_img)
    info = {'align': {'rot180': bool(tf['rot180'])}, 'applied_shift_xy': [int(tf['dx']), int(tf['dy'])]}
    other_al = aligned_donor(other_img, target_img, info)
    reg = change_region(other_img, target_img, mask, info) if defect_type in STRUCTURE_TYPES else mask
    return dilate(reg, px), other_al, tf['score']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aug', default='sd_lora_v3')
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--head-seeds', type=int, default=5)
    ap.add_argument('--save-examples', type=int, default=2, help='PCB·종류마다 D/DR/N/ND 예시 몇 장 저장')
    ap.add_argument('--prep', default='photonorm', choices=PREPS, help='임베딩 전처리 (prep.py). canon = 기판 정렬 + 밝기·대비')
    ap.add_argument('--remove-dilate', type=int, default=8,
                    help='결함을 지우고(DR)·붙이는(ND) 영역을 마스크에서 몇 px 넓힐지. 크게 하면 주석 밖의 결함 흔적까지 지운다')
    ap.add_argument('--data-root')
    a = ap.parse_args()
    tag = '' if a.remove_dilate == 8 else f'_dil{a.remove_dilate}'
    torch.set_num_threads(8)

    data, dev = Data(a.data_root), device()
    ks = KShot(a.k, a.seed, a.data_root)
    ex = PatchFeatures().to(dev)
    pool = read_csv(data.splits / 'pool.csv')
    out_rows, ex_dir = [], ROOT / 'AnomalyDetection/results/defect_removal_examples'
    ex_dir.mkdir(parents=True, exist_ok=True)
    summary, mix_rows = [], []
    for pcb in a.pcbs:
        nl, val = data.train_normal(pcb), data.val_normal(pcb)
        real = data.train_defects(pcb, a.k, a.seed)
        emb_items, emb_imgs = embedder(data, ex, dev, pcb, a.res, a.prep)
        Xn, Xv, Xr = (emb_items(x) for x in (nl, val, real))
        N_ = unit(Xn)
        sim = N_ @ N_.T
        np.fill_diagonal(sim, -1)
        scale = float(np.median(1 - sim.max(1)))
        nidx = {it.image: i for i, it in enumerate(nl)}
        k_imgs = {it.image for it in real}
        normals = ks.normals(pcb)
        rng = np.random.default_rng(7017)

        arrs = {'D': [], 'DR': [], 'N': [], 'ND': [], 'NR': []}
        meta, saved = [], {}
        for row in [r for r in pool if r['category'] == pcb and r['image'] not in k_imgs]:
            t = row['defect_type'].split('|')[0]
            D, ids = ks.load(row)
            m_D = ids == TYPE_ID[t]
            if not m_D.any():
                continue
            cands = [normals[i] for i in rng.choice(len(normals), 5, replace=False)]
            ranked = sorted(cands, key=lambda r: -board_transform(ks.load(r)[0], D)['score'])
            best = ranked[0]
            N, _ = ks.load(best)
            # DR: D 좌표에서 결함 영역을 정렬한 N으로 덮는다
            reg_D, N_al, score = region_in(D, N, m_D, t, a.remove_dilate)
            W, H = D.size
            DR = paste_back(D, N_al, (0, 0, W, H), reg_D, feather=3)
            # ND: N 좌표로 D를 옮겨 결함 영역만 N에 붙인다
            tf = board_transform(D, N)
            info = {'align': {'rot180': bool(tf['rot180'])}, 'applied_shift_xy': [int(tf['dx']), int(tf['dy'])]}
            D_al = aligned_donor(D, N, info)
            m_N = np.asarray(aligned_donor(Image.fromarray(m_D.astype(np.uint8) * 255), N, info)) > 127
            reg_N = dilate(change_region(D, N, m_N, info) if t in STRUCTURE_TYPES else m_N, a.remove_dilate)
            Wn, Hn = N.size
            ND = paste_back(N, D_al, (0, 0, Wn, Hn), reg_N, feather=3)
            # NR (대조군): N의 같은 영역을 다른 정상 N2로 덮는다. 붙이기 자체가 만드는 흔적이 불량으로 보이는지 확인
            N2, _ = ks.load(ranked[1])
            tf2 = board_transform(N2, N)
            info2 = {'align': {'rot180': bool(tf2['rot180'])}, 'applied_shift_xy': [int(tf2['dx']), int(tf2['dy'])]}
            NR = paste_back(N, aligned_donor(N2, N, info2), (0, 0, Wn, Hn), reg_N, feather=3)
            for key, im in (('D', D), ('DR', DR), ('N', N), ('ND', ND), ('NR', NR)):
                arrs[key].append(im)
            meta.append(dict(image=row['image'], defect_type=t, base=best['image'], align_score=round(score, 3),
                             region_px=int(reg_D.sum()), image_px=W * H))
            if saved.get(t, 0) < a.save_examples:
                saved[t] = saved.get(t, 0) + 1
                tiles = [im.resize((W // 4, H // 4)) for im in (D, DR, N, ND)]
                sheet = Image.new('RGB', (sum(x.width for x in tiles) + 30, tiles[0].height), 'white')
                x = 0
                for tile in tiles:
                    sheet.paste(tile, (x, 0))
                    x += tile.width + 10
                sheet.save(ex_dir / f'{pcb}_{t}_{saved[t]}_D-DR-N-ND.jpg', quality=85)

        E = {k_: emb_imgs(v) for k_, v in arrs.items()}
        base_i = np.array([nidx[m['base']] for m in meta])

        def novelty(X, exclude=None):
            s = unit(X) @ N_.T
            if exclude is not None:
                s[np.arange(len(X)), exclude] = -1
            return (1 - s.max(1)) / scale

        nov = {'D': novelty(E['D']), 'DR': novelty(E['DR']), 'ND': novelty(E['ND'], base_i)}
        # 생성 결과도 같은 잣대로
        gen = data.generated_defects(a.aug, pcb, a.k, a.seed)
        Xg = emb_items(gen)
        with open(data.generated / a.aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv', newline='', encoding='utf-8-sig') as f:
            base_of = {r['image']: r['base_normal_image'] for r in csv.DictReader(f)}
        nov['GEN'] = novelty(Xg, np.array([nidx[base_of[g.image]] for g in gen]))

        det = {k_: [] for k_ in ('D', 'DR', 'N', 'ND', 'NR', 'GEN')}
        for hs in range(a.head_seeds):
            h = EmbeddingHead('devnet', seed=hs).fit(Xn, Xr)
            thr = threshold_at_fpr(h.score(Xv), 0.01)
            for k_, X in (('D', E['D']), ('DR', E['DR']), ('N', E['N']), ('ND', E['ND']), ('NR', E['NR']), ('GEN', Xg)):
                det[k_].append(float(np.mean(h.score(X) > thr)))
        cache = ROOT / 'Dataset/cache/defect_removal' / f'{pcb}_{a.aug}_K{a.k}_seed{a.seed}_{a.prep}{tag}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Xn=Xn, Xv=Xv, Xr=Xr, Xg=Xg, **E)

        # 생성을 넣어 학습한 헤드는 결함 자체(ND)를 더 잘 잡는가? 지름길(DR)은 덜 쓰는가?
        gen_types = np.array([g.defect_type for g in gen])
        pick = np.concatenate([np.random.default_rng(0).choice(np.where(gen_types == t)[0], 10, replace=False)
                               for t in np.unique(gen_types)])
        mixes = {'none': Xr, f'+{a.aug} 160': np.concatenate([Xr, Xg]), f'+{a.aug} 40': np.concatenate([Xr, Xg[pick]])}
        for kind in ('devnet', 'mlp'):
            for name, Xa in mixes.items():
                rates = {k_: [] for k_ in ('D', 'DR', 'ND', 'NR')}
                for hs in range(a.head_seeds):
                    h = EmbeddingHead(kind, seed=hs).fit(Xn, Xa)
                    thr = threshold_at_fpr(h.score(Xv), 0.01)
                    for k_ in rates:
                        rates[k_].append(float(np.mean(h.score(E[k_]) > thr)))
                mix_rows.append(dict(pcb=pcb, model=kind, train=name, **{k_: float(np.mean(v)) for k_, v in rates.items()}))
                print(f'  {pcb} [{kind:6s} {name:18s}] 불량 판정 비율  D {np.mean(rates["D"]):.2f} | DR {np.mean(rates["DR"]):.2f} | '
                      f'ND {np.mean(rates["ND"]):.2f} | NR {np.mean(rates["NR"]):.2f}', flush=True)

        row = dict(pcb=pcb, n=len(meta), **{f'nov_{k_}': float(np.mean(v)) for k_, v in nov.items()},
                   **{f'devnet_{k_}': float(np.mean(v)) for k_, v in det.items()},
                   region_frac=float(np.mean([m['region_px'] / m['image_px'] for m in meta])))
        summary.append(row)
        print(f"{pcb} (n={len(meta)}, 결함 영역 평균 {row['region_frac'] * 100:.2f}%)", flush=True)
        print(f"  novelty   D {row['nov_D']:.2f} | DR(결함 제거) {row['nov_DR']:.2f} | ND(정상+결함) {row['nov_ND']:.2f} | "
              f"생성 {row['nov_GEN']:.2f}   (정상끼리 = 1)", flush=True)
        print(f"  DevNet 불량 판정 비율  D {row['devnet_D']:.2f} | DR {row['devnet_DR']:.2f} | N(바탕 정상) "
              f"{row['devnet_N']:.2f} | ND {row['devnet_ND']:.2f} | NR(정상+정상 붙이기) {row['devnet_NR']:.2f} | "
              f"생성 {row['devnet_GEN']:.2f}", flush=True)
        for m, d, dr, nd in zip(meta, nov['D'], nov['DR'], nov['ND']):
            out_rows.append(dict(pcb=pcb, **m, nov_D=round(float(d), 3), nov_DR=round(float(dr), 3), nov_ND=round(float(nd), 3)))

    out = ROOT / 'AnomalyDetection/results' / f'defect_removal_K{a.k}_seed{a.seed}_{a.prep}{tag}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print('\n평균 (PCB 3개)')
    for key in summary[0]:
        if key != 'pcb':
            print(f'  {key:12s} {np.mean([r[key] for r in summary]):.3f}')
    print('\n학습 조건별 불량 판정 비율 (val_normal FPR 1% 임계값, PCB 평균)  D=실제 불량 | DR=결함 제거 | ND=정상+실제 결함')
    for kind in ('devnet', 'mlp'):
        for name in dict.fromkeys(r['train'] for r in mix_rows):
            rs = [r for r in mix_rows if r['model'] == kind and r['train'] == name]
            print(f'  {kind:6s} {name:18s} D {np.mean([r["D"] for r in rs]):.2f} | DR {np.mean([r["DR"] for r in rs]):.2f} | '
                  f'ND {np.mean([r["ND"] for r in rs]):.2f} | NR {np.mean([r["NR"] for r in rs]):.2f}')
    with open(ROOT / 'AnomalyDetection/results' / f'defect_removal_mix_K{a.k}_seed{a.seed}_{a.prep}{tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(mix_rows[0]))
        w.writeheader()
        w.writerows(mix_rows)
    print(f'저장: {out}\n예시: {ex_dir}')


if __name__ == '__main__':
    main()
