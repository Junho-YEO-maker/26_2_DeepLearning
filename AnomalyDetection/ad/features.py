"""고정 특징 추출기 (PatchCore 방식).

WideResNet50(ImageNet)의 layer2·layer3 출력을 3×3 평균으로 주변 정보를 섞고,
layer3을 layer2 크기로 키워 이어 붙인 뒤 채널을 1024로 줄인다 → 패치 특징 (N, 1024, H/8, W/8).
이미지 임베딩은 패치 특징의 공간 평균과 공간 최대를 이어 붙인 2048차원 벡터다.
최대값을 넣는 이유: 결함이 이미지 면적의 0.2–0.5%라 평균만 쓰면 결함 신호가 묻힌다.
"""
import torch
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_tensor(batch_uint8):
    """(B, H, W, 3) uint8 numpy → 정규화된 (B, 3, H, W) float 텐서."""
    x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


class PatchFeatures(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        net = torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.dim = dim
        # adaptive_avg_pool1d(1536 → dim)과 똑같은 선형 변환을 행렬로 만들어 둔다(MPS는 나누어떨어지지 않는 pool 미지원)
        pool = F.adaptive_avg_pool1d(torch.eye(1536).unsqueeze(1), dim).squeeze(1)   # (1536, dim)
        self.register_buffer('pool', pool)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f2, f3], 1)                       # (B, 1536, h, w)
        f = torch.einsum('bchw,cd->bdhw', f, self.pool)  # 채널 1536 → 1024 (PatchCore 원 구현과 같은 평균 풀링)
        return f


@torch.no_grad()
def embed(patch_maps):
    """(B, C, h, w) → (B, 2C) 이미지 임베딩: [공간 평균, 공간 최대]."""
    return torch.cat([patch_maps.mean((2, 3)), patch_maps.amax((2, 3))], 1)
