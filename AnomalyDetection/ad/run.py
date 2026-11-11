"""실험 실행기. 결과는 AnomalyDetection/results/results.csv 에 추가된다(이미 있는 실행은 건너뜀).

예 (프로젝트 루트에서):
  # 파일럿: PatchCore K=0, 해상도 256 vs 512
  python -m AnomalyDetection.ad.run --models patchcore --res 256 512
  # baseline: 임베딩 모델 × K × seed × 증강
  python -m AnomalyDetection.ad.run --models knn_mlp devnet --ks 0 1 2 4 8 --seeds 0 1 2 3 4 --augs none classic copypaste
  # 원본이 다른 곳에 있으면
  python -m AnomalyDetection.ad.run ... --data-root /content/VisA_PCB

규칙
  - K=0 은 정상만 쓰는 모델(patchcore, knn_mlp의 kNN)만 돈다. K와 seed에 무관하므로 seed 0 한 줄만 기록한다.
  - devnet, classifier 는 K>0 에서만 돈다.
  - 생성 증강(copypaste 등)은 실제 K장 + 해당 manifest의 생성 불량(PCB당 160장)을 함께 쓴다.
  - classic 은 불량마다 10개, 정상마다 1개의 증강본을 더한다(임베딩 모델). 분류기는 학습 중에 매번 건다.
"""
import argparse
import hashlib
import time

import numpy as np
import torch

from .augment import classic
from .data import PCBS, Data
from .eval import done_keys, evaluate, record
from .features import PatchFeatures, device, embed, to_tensor
from .models import Classifier, EmbeddingHead, PatchCore

GEN_METHODS = ('copypaste', 'ctransgan', 'sd_lora')
CLASSIC_COPIES_DEFECT, CLASSIC_COPIES_NORMAL = 10, 1


class Embedder:
    """이미지 임베딩 계산기. 증강하지 않은 이미지는 디스크에 캐시한다."""

    def __init__(self, data, ex, dev, res):
        self.data, self.ex, self.dev, self.res = data, ex, dev, res
        self.dir = data.cache / f'emb{res}'
        self.dir.mkdir(parents=True, exist_ok=True)

    def _forward(self, arrays, bs=16):
        out = []
        for i in range(0, len(arrays), bs):
            m = self.ex(to_tensor(np.stack(arrays[i:i + bs])).to(self.dev))
            out.append(embed(m).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2 * self.ex.dim), np.float32)

    def items(self, items):
        res, todo = [None] * len(items), []
        for i, it in enumerate(items):
            p = self.dir / (hashlib.sha1(it.image.encode()).hexdigest()[:16] + '.npy')
            if p.exists():
                res[i] = np.load(p)
            else:
                todo.append((i, p))
        for j in range(0, len(todo), 64):
            chunk = todo[j:j + 64]
            embs = self._forward([self.data.load(items[i].image, self.res) for i, _ in chunk])
            for (i, p), e in zip(chunk, embs):
                np.save(p, e)
                res[i] = e
        return np.stack(res) if res else np.zeros((0, 2 * self.ex.dim), np.float32)

    def augmented(self, items, copies, rng):
        arrays = [classic(self.data.load(it.image, self.res), rng) for it in items for _ in range(copies)]
        return self._forward(arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['patchcore'],
                    choices=['patchcore', 'knn_mlp', 'devnet', 'classifier'])
    ap.add_argument('--pcbs', nargs='+', default=list(PCBS))
    ap.add_argument('--ks', nargs='+', type=int, default=[0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--augs', nargs='+', default=['none'], choices=['none', 'classic', *GEN_METHODS])
    ap.add_argument('--res', nargs='+', type=int, default=[256])
    ap.add_argument('--data-root', help='원본 pcb 폴더들의 부모 (기본: Dataset/visa/raw)')
    ap.add_argument('--clf-steps', type=int, default=400)
    a = ap.parse_args()

    data, dev = Data(a.data_root), device()
    print(f'device={dev}', flush=True)
    ex = PatchFeatures().to(dev)
    done = done_keys()

    def todo(model, pcb, k, seed, aug, res):
        return (model, pcb, str(k), str(seed), aug, str(res)) not in done

    def log(model, pcb, k, seed, aug, res, n_gen, scores_test, scores_val, test, t0):
        m = evaluate(scores_test, [it.label for it in test], [it.defect_type for it in test], scores_val)
        key = dict(model=model, regime='main', category=pcb, K=k, seed=seed, aug=aug, n_generated=n_gen, res=res)
        record(key, m)
        done.add((model, pcb, str(k), str(seed), aug, str(res)))
        print(f'{model:10s} {pcb} K={k} s={seed} {aug:9s} res={res}  FNR@1%={m["fnr_at_fpr1"]:.3f} '
              f'macro={m["macro_fnr_at_fpr1"]:.3f} AUROC={m["auroc"]:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    for res in a.res:
        for pcb in a.pcbs:
            test, val, normals = data.test(pcb), data.val_normal(pcb), data.train_normal(pcb)

            if 'patchcore' in a.models and todo('patchcore', pcb, 0, 0, 'none', res):
                t0 = time.time()
                pc = PatchCore(ex, dev).fit([data.load(it.image, res) for it in normals])
                st = pc.score([data.load(it.image, res) for it in test])
                sv = pc.score([data.load(it.image, res) for it in val])
                log('patchcore', pcb, 0, 0, 'none', res, 0, st, sv, test, t0)
                del pc
                if dev.type == 'mps':
                    torch.mps.empty_cache()

            emb_models = [m for m in ('knn_mlp', 'devnet') if m in a.models]
            if emb_models:
                E = Embedder(data, ex, dev, res)
                Xt, Xv, Xn = E.items(test), E.items(val), E.items(normals)
            for k in a.ks:
                for seed in (a.seeds if k > 0 else [0]):
                    for aug in (a.augs if k > 0 else ['none']):
                        defects = data.train_defects(pcb, k, seed)
                        gen = data.generated_defects(aug, pcb, k, seed) if aug in GEN_METHODS else []
                        # ---- 임베딩 모델
                        heads = [m for m in emb_models if todo(m, pcb, k, seed, aug, res) and (k > 0 or m == 'knn_mlp')]
                        if heads:
                            t0 = time.time()
                            Xa, Xn_fit = (E.items(defects + gen) if k > 0 else None), Xn
                            if aug == 'classic':
                                rng = np.random.default_rng([seed, k, int(pcb[-1])])
                                Xa = np.concatenate([Xa, E.augmented(defects, CLASSIC_COPIES_DEFECT, rng)])
                                Xn_fit = np.concatenate([Xn, E.augmented(normals, CLASSIC_COPIES_NORMAL, rng)])
                            for m in heads:
                                kind = 'knn' if k == 0 else ('mlp' if m == 'knn_mlp' else 'devnet')
                                h = EmbeddingHead(kind, seed=seed).fit(Xn_fit, Xa)
                                log(m, pcb, k, seed, aug, res, len(gen), h.score(Xt), h.score(Xv), test, t0)
                        # ---- 미세조정 분류기
                        if 'classifier' in a.models and k > 0 and todo('classifier', pcb, k, seed, aug, res):
                            t0 = time.time()
                            clf = Classifier(dev, seed=seed, steps=a.clf_steps,
                                             augment=classic if aug == 'classic' else None)
                            clf.fit([data.load(it.image, res) for it in normals],
                                    [data.load(it.image, res) for it in defects + gen])
                            st = clf.score([data.load(it.image, res) for it in test])
                            sv = clf.score([data.load(it.image, res) for it in val])
                            log('classifier', pcb, k, seed, aug, res, len(gen), st, sv, test, t0)
                            del clf
                            if dev.type == 'mps':
                                torch.mps.empty_cache()


if __name__ == '__main__':
    main()
