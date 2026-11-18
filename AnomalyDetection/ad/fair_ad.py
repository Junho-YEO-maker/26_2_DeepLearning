"""촬영 조건을 뺀 시험셋(fair test)으로 모델과 증강을 다시 줄 세운다 (11/18).

왜: VisA 불량 사진은 정상 사진과 촬영 조건(밝기, 기판 위치 등)이 달라서, 기존 시험셋 FNR은 촬영 조건으로 맞히는 모델을
    더 높게 친다. 인라인 검사처럼 촬영 조건이 섞이는 현장에서는 그 이득이 없다. 목표는 증강을 넣고도 촬영 조건이 섞인
    라인에서 잘 되는 모델을 찾는 것이다.

시험셋(test.csv) 불량 D마다 같은 PCB의 시험 정상 N(후보 5장 중 기판 정렬 점수가 가장 높은 것)과 짝지어 세 장을 더 만든다.
  DR  D의 결함 영역만 N의 픽셀로 지운 것            음성, 촬영 조건 B(불량 사진 쪽)
  ND  N에 D의 결함 영역만 옮겨 붙인 것              양성, 촬영 조건 A(정상 사진 쪽)
  NR  N에 다른 시험 정상 N2의 같은 영역을 붙인 것    음성, 촬영 조건 A, 붙인 경계만 있음(대조군)
결함 영역 = 결함 마스크 전체(bent·missing은 + 원래 핀 자리) 40px 넓힘 (local_ad 진단과 같은 규칙).
시험셋 이미지만 쓴다(학습·검증 이미지 없음). 분할 seed와 무관하므로 PCB마다 한 번 만들어 캐시한다.

지표: 촬영 조건 A·B가 반반 섞인 라인을 흉내 낸다 (임계값도 그 라인의 정상, 즉 A·B가 섞인 음성으로 정한다)
  양성 = D(B) + ND(A), 음성 = N + NR(A) + DR(B). 클래스마다 A와 B가 무게 절반씩.
  fair_fnr5              (무게) 과검률 5% 임계값에서 놓친 비율(무게)              ← 주 지표, 낮을수록 좋음
  fair_auroc             무게 AUROC                                               ← 보조 지표
  fair_fnr               과검률 1%. B 음성(DR)이 PCB당 약 60장뿐이라 1%면 DR 한 장이 임계값을 정한다 → 참고만
  fair_miss_D, fair_miss_ND, fair_fa_A, fair_fa_DR   주 임계값에서 부분별 놓침·과검
  fair_fnr_{종류}         같은 임계값에서 종류별 (D, ND 반반)
  v_N, v_D, v_ND, v_DR, v_NR   val_normal 과검률 1% 임계값에서 불량 판정 비율 (해석용)
  fnr, fnr5              기존 시험셋 FNR@FPR1%·5% (참고)
모델 (K장 = train_K{k}_seed{s})
  patchcore    정상만, 원본 512 (--patchcore일 때, seed 무관)
  devnet, mlp  canon 임베딩 + 헤드 (헤드 시드 5, local_ad img와 같음)
  seg          local_ad seg, 집계 top10 (헤드 시드 3). none과 --augs 첫 번째만
  seg_aug      seg + capture_aug(학습 이미지 촬영 조건 무작위화). none과 --augs 전부
  학습 = none(실제 K장만) / +{aug}(Dataset/generated/{aug}/K{k}_seed{s}에 이미지가 있을 때만)
이미지별 점수: AnomalyDetection/results/fair_scores/K{k}_seed{s}_{pcb}.npz (키 '모델|학습|헤드시드|부분')

    python -m AnomalyDetection.ad.fair_ad --build                  # 시험셋만 만들기 (CPU 병렬, 처음 한 번)
    python -m AnomalyDetection.ad.fair_ad --seed 0 --patchcore --augs sd_lora_v3 sd_lora anomalydiffusion
    python -m AnomalyDetection.ad.fair_ad --seed 1 --pcbs pcb2      # PCB 하나 (병렬로 돌릴 때)
    python -m AnomalyDetection.ad.fair_ad --k 4 --augs copypaste --sets seg:none seg_aug:none seg_aug:copypaste img:none \
        --seg-seeds 2 --img-seeds 2 --img-kinds devnet               # K 곡선 (results/logs/run_fair_kcurve.sh)
결과: AnomalyDetection/results/fair_ad_K{k}_seed{s}[_{pcb}].csv
"""
import argparse
import csv
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .data import PCBS, ROOT, TYPES, Data, read_csv
from .eval import threshold_at_fpr
from .features import PatchFeatures, device, embed, to_tensor
from .local_ad import Backbone, SegModel, aggregate
from .models import EmbeddingHead, PatchCore
from .patch_ad import load_mask
from .prep import Canon

sys.path.insert(0, str(ROOT / 'Augmentation'))
from common.kshot import STRUCTURE_TYPES, aligned_donor, board_transform, change_region, paste_back  # noqa: E402

RES, PX = 512, 40
PARTS = ('DR', 'ND', 'NR')
CACHE = ROOT / 'Dataset/cache/fair'


# ---------------- 시험셋 만들기 ----------------
def dilate(mask, px):
    """kshot.dilate와 같은 정사각 최대 필터. PIL MaxFilter(81)는 원본 해상도에서 수십 초라 scipy로 바꿨다."""
    return ndimage.maximum_filter(mask.astype(np.uint8), size=2 * px + 1) > 0


def _info(tf):
    return {'align': {'rot180': bool(tf['rot180'])}, 'applied_shift_xy': [int(tf['dx']), int(tf['dy'])]}


def _open(p):
    with Image.open(p) as im:
        return im.convert('RGB').copy()


def _build_one(job):
    d_path, m_path, dtype, cand_paths, cand_rels, ref_path = job
    D = _open(d_path)
    m_D = np.asarray(Image.open(m_path)) > 0
    types = dtype.split('|')
    t = next((x for x in types if x in STRUCTURE_TYPES), types[0])
    cands = [_open(p) for p in cand_paths]
    order = sorted(range(len(cands)), key=lambda j: -board_transform(cands[j], D)['score'])
    N, N2 = cands[order[0]], cands[order[1]]
    W, H = D.size
    info = _info(board_transform(N, D))                                      # N → D 좌표
    reg_D = dilate(change_region(N, D, m_D, info) if t in STRUCTURE_TYPES else m_D, PX)
    DR = paste_back(D, aligned_donor(N, D, info), (0, 0, W, H), reg_D, feather=3)
    info = _info(board_transform(D, N))                                      # D → N 좌표
    m_N = np.asarray(aligned_donor(Image.fromarray(m_D.astype(np.uint8) * 255), N, info)) > 127
    reg_N = dilate(change_region(D, N, m_N, info) if t in STRUCTURE_TYPES else m_N, PX)
    Wn, Hn = N.size
    ND = paste_back(N, aligned_donor(D, N, info), (0, 0, Wn, Hn), reg_N, feather=3)
    NR = paste_back(N, aligned_donor(N2, N, _info(board_transform(N2, N))), (0, 0, Wn, Hn), reg_N, feather=3)
    c = Canon.__new__(Canon)            # Canon.align만 쓴다(기준 = 그 PCB train_normal 첫 장)
    c.ref, c.res = _open(ref_path), RES
    out = {'bg': cand_rels[order[0]],
           'mask_ND': np.asarray(Image.fromarray(m_N.astype(np.uint8) * 255).resize((RES, RES), Image.NEAREST)) > 127}
    for k, im in (('DR', DR), ('ND', ND), ('NR', NR)):
        out[k] = np.asarray(im.resize((RES, RES), Image.BILINEAR))          # data.load와 같은 줄이기
        out[k + '_canon'] = c.align(im)
    return out


def fair_set(data, pcb, workers=16):
    cp = CACHE / f'{pcb}_dil{PX}.npz'
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    rows = [r for r in read_csv(data.splits / 'test.csv') if r['category'] == pcb]
    normals = [r for r in rows if r['label'] == '0']
    defects = [r for r in rows if r['label'] == '1']
    rng = np.random.default_rng([7017, PCBS.index(pcb)])
    ref = str(data.path(data.train_normal(pcb)[0].image))
    jobs = []
    for r in defects:
        cand = [normals[i] for i in rng.choice(len(normals), 5, replace=False)]
        jobs.append((str(data.path(r['image'])), str(data.path(r['mask'])), r['defect_type'],
                     [str(data.path(c['image'])) for c in cand], [c['image'] for c in cand], ref))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
        res = list(pool.map(_build_one, jobs))
    out = {k: np.stack([r[k] for r in res]) for k in res[0] if k != 'bg'}
    out.update(image=np.array([r['image'] for r in defects]), type=np.array([r['defect_type'] for r in defects]),
               bg=np.array([r['bg'] for r in res]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, **out)
    print(f'{pcb}: 공정 시험셋 {len(defects)}쌍 만듦 ({time.time() - t0:.0f}s) → {cp}', flush=True)
    return out


# ---------------- 지표 ----------------
def wthr(s, w, fpr):
    """무게 과검률이 fpr 이하가 되는 가장 낮은 임계값 (threshold_at_fpr의 무게 버전, 판정은 score > thr)."""
    o = np.argsort(-s, kind='stable')
    cw = np.cumsum(w[o]) / w.sum()
    k = int(np.searchsorted(cw, fpr + 1e-12, side='right'))
    return s[o][k] if k < len(s) else -np.inf


def metrics(sc, sv, labels, types):
    """sc = {'test': 시험셋 점수, 'DR', 'ND', 'NR'}, sv = val_normal 점수."""
    y = np.asarray(labels)
    N, D = sc['test'][y == 0], sc['test'][y == 1]
    ND, DR, NR = sc['ND'], sc['DR'], sc['NR']
    dtypes = np.asarray(types)[y == 1]
    negA = np.concatenate([N, NR])
    neg_s = np.concatenate([negA, DR])
    neg_w = np.concatenate([np.full(len(negA), .5 / len(negA)), np.full(len(DR), .5 / len(DR))])
    pos_s = np.concatenate([D, ND])
    pos_w = np.concatenate([np.full(len(D), .5 / len(D)), np.full(len(ND), .5 / len(ND))])
    out = {}
    for name, f in (('fair_fnr', .01), ('fair_fnr5', .05)):
        thr = wthr(neg_s, neg_w, f)
        out[name] = float((pos_w * (pos_s <= thr)).sum() / pos_w.sum())
        if f == .01:
            thr1 = thr
    out['fair_auroc'] = float(roc_auc_score(np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))], np.r_[pos_s, neg_s],
                                            sample_weight=np.r_[pos_w, neg_w]))
    out.update(fair_miss_D=float(np.mean(D <= thr1)), fair_miss_ND=float(np.mean(ND <= thr1)),
               fair_fa_A=float(np.mean(negA > thr1)), fair_fa_DR=float(np.mean(DR > thr1)))
    for t in TYPES:
        i = dtypes == t
        out[f'fair_fnr_{t}'] = float(.5 * np.mean(D[i] <= thr1) + .5 * np.mean(ND[i] <= thr1)) if i.any() else float('nan')
        out[f'n_{t}'] = int(i.sum())
    thr_v = threshold_at_fpr(sv, .01)
    for k, s in (('N', N), ('D', D), ('ND', ND), ('DR', DR), ('NR', NR)):
        out[f'v_{k}'] = float(np.mean(s > thr_v))
    out['fnr'] = float(np.mean(D <= threshold_at_fpr(N, .01)))
    out['fnr5'] = float(np.mean(D <= threshold_at_fpr(N, .05)))
    return out


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', action='store_true', help='공정 시험셋만 만들고 끝낸다')
    ap.add_argument('--augs', nargs='*', default=['sd_lora_v3'], help='생성 불량 (Dataset/generated 아래 이름). 이미지가 없으면 건너뜀')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--patchcore', action='store_true', help='PatchCore(정상만)도 (seed 무관이라 한 번만)')
    ap.add_argument('--models', nargs='+', default=['img', 'seg', 'seg_aug'], choices=['img', 'seg', 'seg_aug'])
    ap.add_argument('--img-seeds', type=int, default=5)
    ap.add_argument('--seg-seeds', type=int, default=3)
    ap.add_argument('--seg-steps', type=int, default=2000)
    ap.add_argument('--sets', nargs='*', help='기본 계획 대신 "모델:학습" 목록. 예: seg:none seg_aug:none seg_aug:copypaste img:none '
                    '(학습 = none 또는 --augs 안의 이름, 생성이 없으면 건너뜀)')
    ap.add_argument('--img-kinds', nargs='+', default=['devnet', 'mlp'], choices=['devnet', 'mlp'])
    ap.add_argument('--tag', default='', help='결과·점수 파일 이름 뒤에 붙임')
    ap.add_argument('--workers', type=int, default=16, help='시험셋 만들기 프로세스 수')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--data-root')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = Data(a.data_root)
    for pcb in a.pcbs:                   # CUDA를 켜기 전에 (CPU 병렬)
        fair_set(data, pcb, a.workers)
    if a.build:
        return

    dev = device()
    ex = PatchFeatures().to(dev)
    augs = []
    for aug in a.augs:
        man = data.generated / aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv'
        if man.exists() and data.path(read_csv(man)[0]['image']).exists():
            augs.append(aug)
        else:
            print(f'생성 없음, 건너뜀: {aug} K{a.k}_seed{a.seed}', flush=True)
    gen_masks = {r['image']: r['mask'] for aug in augs
                 for r in read_csv(data.generated / aug / f'K{a.k}_seed{a.seed}' / 'manifest.csv')}
    if a.sets:                           # 모델 → 학습 목록
        want = {}
        for s in a.sets:
            m, t = s.split(':')
            want.setdefault(m, []).append(t)
    else:
        want = {m: v for m, v in {'img': ['none'] + augs, 'seg': ['none'] + augs[:1], 'seg_aug': ['none'] + augs}.items()
                if m in a.models}
    want = {m: [t for t in v if t == 'none' or t in augs] for m, v in want.items()}
    print('계획:', want, flush=True)
    rows = []
    for pcb in a.pcbs:
        t0 = time.time()
        fs = fair_set(data, pcb)
        test, val, nl = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)
        real = data.train_defects(pcb, a.k, a.seed)
        gens = {aug: data.generated_defects(aug, pcb, a.k, a.seed) for aug in augs}
        labels, types = [x.label for x in test], [x.defect_type for x in test]
        assert list(fs['image']) == [x.image for x in test if x.label == 1], '공정 시험셋과 test.csv 순서가 다름'

        scores = {}

        def add(model, train, hs, sc, sv):
            m = metrics(sc, sv, labels, types)
            rows.append(dict(pcb=pcb, k=a.k, seed=a.seed, model=model, train=train, head_seed=hs, **m))
            for part, s in {**sc, 'val': sv}.items():
                scores[f'{model}|{train}|{hs}|{part}'] = np.asarray(s, np.float32)
            return m

        load = lambda items: np.stack([data.load(x.image, RES) for x in items])  # noqa: E731
        At, Av, An, Ar = (load(x) for x in (test, val, nl, real))
        print(f'{pcb}: 시험 {len(test)}, 공정 {len(fs["ND"])}쌍, 정상 {len(nl)}, 실제 {len(real)}, '
              + ', '.join(f'{g} {len(v)}' for g, v in gens.items()) + f' ({time.time() - t0:.0f}s)', flush=True)

        if a.patchcore:
            t1 = time.time()
            pc = PatchCore(ex, dev).fit(An)
            m = add('patchcore', 'none', 0, {'test': pc.score(At), **{p: pc.score(fs[p]) for p in PARTS}}, pc.score(Av))
            print(f'  {pcb} patchcore: 공정@5 {m["fair_fnr5"]:.3f} 표준 {m["fnr"]:.3f} ({time.time() - t1:.0f}s)', flush=True)
            del pc
            torch.cuda.empty_cache()

        if want.get('img'):
            c = Canon(data, pcb, RES)

            def cemb(arrs):
                out = []
                for i in range(0, len(arrs), 16):
                    x = to_tensor(np.stack([c.fix(v) for v in arrs[i:i + 16]])).to(dev)
                    out.append(embed(ex(x)).float().cpu().numpy())
                return np.concatenate(out)

            item = lambda items: cemb([c.item(x.image) for x in items])  # noqa: E731
            Xt, Xv, Xn, Xr = (item(x) for x in (test, val, nl, real))
            Xf = {p: cemb(fs[p + '_canon']) for p in PARTS}
            sets = {('none' if t == 'none' else f'+{t}'): (Xr if t == 'none' else np.concatenate([Xr, item(gens[t])]))
                    for t in want['img']}
            for name, Xa in sets.items():
                for kind in a.img_kinds:
                    for hs in range(a.img_seeds):
                        h = EmbeddingHead(kind, seed=hs).fit(Xn, Xa)
                        add(kind, name, hs, {'test': h.score(Xt), **{p: h.score(Xf[p]) for p in PARTS}}, h.score(Xv))
            print(f'  {pcb} img 끝 ({time.time() - t0:.0f}s)', flush=True)

        if want.get('seg') or want.get('seg_aug'):
            bb = Backbone(ex).to(dev).eval()
            ref = torch.as_tensor(An.reshape(-1, 3).mean(0, dtype=np.float64), device=dev).float()
            Mr = np.stack([load_mask(data, x, gen_masks) for x in real])
            used = {t for m in ('seg', 'seg_aug') for t in want.get(m, []) if t != 'none'}
            G = {g: (load(gens[g]), np.stack([load_mask(data, x, gen_masks) for x in gens[g]])) for g in used}
            plan = [(m, 'none' if t == 'none' else f'+{t}', () if t == 'none' else G[t])
                    for m in ('seg', 'seg_aug') for t in want.get(m, [])]
            for model, name, extra in plan:
                for hs in range(a.seg_seeds):
                    t1 = time.time()
                    sm = SegModel(bb, ref, dev, seed=hs, steps=a.seg_steps, aug=model == 'seg_aug').fit(An, Ar, Mr, *extra)
                    s = lambda arr: aggregate(sm.score(arr), 'top10')  # noqa: E731
                    m = add(model, name, hs, {'test': s(At), **{p: s(fs[p]) for p in PARTS}}, s(Av))
                    print(f'  {pcb} {model} {name} 시드 {hs}: 공정@5 {m["fair_fnr5"]:.3f} 표준 {m["fnr"]:.3f} '
                          f'({time.time() - t1:.0f}s)', flush=True)
                    del sm
            torch.cuda.empty_cache()
        sp = ROOT / 'AnomalyDetection/results/fair_scores' / f'K{a.k}_seed{a.seed}{a.tag}_{pcb}.npz'
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp, **scores)

    tag = a.tag + ('' if a.pcbs == list(PCBS) else '_' + '-'.join(a.pcbs))
    out = ROOT / 'AnomalyDetection/results' / f'fair_ad_K{a.k}_seed{a.seed}{tag}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\nPCB 평균, 헤드 시드 평균. 공정@5 = 촬영 조건 반반 라인, 과검 5%에서 놓친 비율 (낮을수록 좋음), 표준 = 기존 시험셋 FNR@1%')
    for model, train in dict.fromkeys((r['model'], r['train']) for r in rows):
        rs = [r for r in rows if r['model'] == model and r['train'] == train]
        per = lambda key: np.mean([np.mean([r[key] for r in rs if r['pcb'] == p]) for p in a.pcbs])  # noqa: E731
        print(f'  {model:9s} {train:20s} 공정@5 {per("fair_fnr5"):.3f} AUROC {per("fair_auroc"):.3f} @1 {per("fair_fnr"):.3f} '
              f'| val 임계값: D {per("v_D"):.2f} ND {per("v_ND"):.2f} DR {per("v_DR"):.2f} NR {per("v_NR"):.2f} '
              f'| 표준 {per("fnr"):.3f}', flush=True)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
