"""AnomalyDetection 팀 모델 (README 4.2절).

| 이름        | 학습 데이터         | 원리                                                                 |
| ----------- | ------------------- | -------------------------------------------------------------------- |
| patchcore   | 정상만 (K 무관)     | 정상 패치 특징 메모리(coreset) → 시험 패치의 최근접 거리 최대값     |
| knn_mlp     | K=0: 정상           | 이미지 임베딩의 kNN(5) 평균 거리                                     |
|             | K>0: 정상 + 불량    | 같은 임베딩 위에 MLP(정상/불량), 균형 배치                           |
| devnet      | 정상 + 불량 (K>0)   | 같은 임베딩 위의 점수망, deviation loss (Pang et al., KDD 2019)      |
| classifier  | 정상 + 불량 (K>0)   | ImageNet ResNet-18 전체 미세조정, 균형 배치                          |

knn_mlp와 devnet은 고정 특징(features.PatchFeatures)의 이미지 임베딩을 공유한다. 모델 차이는 머리(head)뿐이다.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .features import to_tensor


# ---------------------------------------------------------------- PatchCore
def greedy_coreset(feats, n_select, gen, proj_dim=128):
    """k-center greedy (PatchCore). feats: (N, C) 텐서. 무작위 투영으로 거리 계산을 줄인다."""
    n = feats.shape[0]
    if n_select >= n:
        return torch.arange(n, device=feats.device)
    proj = torch.randn(feats.shape[1], proj_dim, generator=gen).to(feats.device) / proj_dim ** 0.5
    z = feats.float() @ proj
    selected = [int(torch.randint(n, (1,), generator=gen))]
    min_d = torch.cdist(z, z[selected[-1]][None]).squeeze(1)
    for _ in range(n_select - 1):
        i = int(torch.argmax(min_d))
        selected.append(i)
        min_d = torch.minimum(min_d, torch.cdist(z, z[i][None]).squeeze(1))
    return torch.tensor(selected, device=feats.device)


class PatchCore:
    def __init__(self, extractor, dev, coreset_ratio=0.01, max_patches=400_000, seed=0):
        self.ex, self.dev = extractor, dev
        self.ratio, self.max_patches, self.seed = coreset_ratio, max_patches, seed

    def fit(self, normal_arrays, bs=8):
        """정상 패치 특징을 모아 coreset 메모리를 만든다. 메모리를 아끼려고 이미지마다 패치를 먼저 무작위로 줄인다
        (전체 ≤ max_patches). coreset 크기는 줄이기 전 전체 패치 수의 coreset_ratio."""
        gen = torch.Generator().manual_seed(self.seed)
        per_img, feats, total = None, [], 0
        for i in range(0, len(normal_arrays), bs):
            m = self.ex(to_tensor(np.stack(normal_arrays[i:i + bs])).to(self.dev))
            b, c, h, w = m.shape
            total += b * h * w
            if per_img is None:
                per_img = min(h * w, self.max_patches // len(normal_arrays))
            p = m.permute(0, 2, 3, 1).reshape(b, h * w, c)
            for j in range(b):
                idx = torch.randperm(h * w, generator=gen)[:per_img].to(self.dev)
                feats.append(p[j, idx].half())
        feats = torch.cat(feats)
        n_sel = max(1000, int(self.ratio * total))
        self.bank = feats[greedy_coreset(feats, n_sel, gen)].float()
        del feats
        return self

    @torch.no_grad()
    def score(self, arrays, bs=4, chunk=8192):
        scores = []
        for i in range(0, len(arrays), bs):
            m = self.ex(to_tensor(np.stack(arrays[i:i + bs])).to(self.dev)).float()
            b, c, h, w = m.shape
            p = m.permute(0, 2, 3, 1).reshape(-1, c)
            d = torch.cat([torch.cdist(p[j:j + chunk], self.bank).min(1).values for j in range(0, len(p), chunk)])
            scores.append(d.reshape(b, h * w).max(1).values.cpu())
        return torch.cat(scores).numpy()


# ---------------------------------------------------------------- 임베딩 위 모델
class EmbeddingHead:
    """kind: 'knn' (K=0), 'mlp', 'devnet'. X는 정상 학습 임베딩 기준으로 표준화한다."""

    def __init__(self, kind, seed=0, steps=600, batch=64, lr=1e-3, knn=5):
        self.kind, self.seed, self.steps, self.batch, self.lr, self.knn = kind, seed, steps, batch, lr, knn

    def fit(self, Xn, Xa=None):
        Xn = torch.as_tensor(Xn, dtype=torch.float32)
        self.mu, self.sd = Xn.mean(0), Xn.std(0) + 1e-6
        Zn = (Xn - self.mu) / self.sd
        if self.kind == 'knn':
            self.bank = Zn
            return self
        assert Xa is not None and len(Xa), 'mlp/devnet은 불량이 필요합니다(K>0)'
        Za = (torch.as_tensor(Xa, dtype=torch.float32) - self.mu) / self.sd
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        d = Zn.shape[1]
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                       torch.nn.Linear(256, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        half = self.batch // 2
        for _ in range(self.steps):       # 균형 배치: 정상 절반 + 불량 절반(복원 추출)
            xn = Zn[torch.randint(len(Zn), (half,), generator=g)]
            xa = Za[torch.randint(len(Za), (half,), generator=g)]
            x = torch.cat([xn, xa])
            y = torch.cat([torch.zeros(half), torch.ones(half)])
            out = self.net(x).squeeze(1)
            if self.kind == 'mlp':
                loss = F.binary_cross_entropy_with_logits(out, y)
            else:                          # deviation loss, 기준 분포 N(0,1), 여유 a=5
                ref = torch.randn(5000, generator=g)
                dev = (out - ref.mean()) / ref.std()
                loss = ((1 - y) * dev.abs() + y * F.relu(5.0 - dev)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X):
        Z = (torch.as_tensor(X, dtype=torch.float32) - self.mu) / self.sd
        if self.kind == 'knn':
            return torch.cdist(Z, self.bank).topk(self.knn, largest=False).values.mean(1).numpy()
        return self.net(Z).squeeze(1).numpy()


# ---------------------------------------------------------------- 미세조정 분류기
class Classifier:
    def __init__(self, dev, seed=0, steps=400, batch=16, lr=1e-4, augment=None):
        self.dev, self.seed, self.steps, self.batch, self.lr = dev, seed, steps, batch, lr
        self.augment = augment            # 학습 중 적용할 함수(arr, rng) → arr 또는 None

    def fit(self, normal_arrays, defect_arrays):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
        self.net = net.to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        half = self.batch // 2
        self.net.train()
        for _ in range(self.steps):
            xs = [normal_arrays[i] for i in rng.integers(len(normal_arrays), size=half)]
            xs += [defect_arrays[i] for i in rng.integers(len(defect_arrays), size=half)]
            if self.augment is not None:
                xs = [self.augment(x, rng) for x in xs]
            x = to_tensor(np.stack(xs)).to(self.dev)
            y = torch.cat([torch.zeros(half), torch.ones(half)]).to(self.dev)
            loss = F.binary_cross_entropy_with_logits(self.net(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, arrays, bs=16):
        out = []
        for i in range(0, len(arrays), bs):
            out.append(self.net(to_tensor(np.stack(arrays[i:i + bs])).to(self.dev)).squeeze(1).float().cpu())
        return torch.cat(out).numpy()
