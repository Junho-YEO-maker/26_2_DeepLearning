"""cTransGAN 네트워크 (Wang et al. 2023 축소 구현). Transformer 분류기·생성기·판별기.

분류기 특징의 종류별 평균을 조건 임베딩으로 쓰고, 생성기는 잡음에 그 임베딩을 이어 붙여 받고,
판별기는 그 임베딩을 추가 토큰으로 받는다.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Blocks(nn.Module):
    def __init__(self, width, heads, depth, checkpointing):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=4*width,
                                       dropout=0., activation='relu', batch_first=True)
            for _ in range(depth)])
        self.checkpointing = checkpointing

    def forward(self, tokens):
        for layer in self.layers:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        return tokens


class PatchEncoder(nn.Module):
    def __init__(self, image_size, patch_size, width, heads, depth, checkpointing):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        count = (image_size//patch_size)**2
        self.input = nn.Linear(3*patch_size**2, width)
        self.cls = nn.Parameter(torch.zeros(1,1,width))
        # Additional token reserved for the condition in discriminator mode.
        self.positions = nn.Parameter(torch.randn(1,count+2,width)*.02)
        self.blocks = Blocks(width, heads, depth, checkpointing)
        self.norm = nn.LayerNorm(width)

    def forward(self, image, condition_token=None):
        batch, channels, height, width = image.shape
        size = self.patch_size
        patches = image.reshape(batch,channels,height//size,size,width//size,size)
        patches = patches.permute(0,2,4,1,3,5).reshape(batch,-1,channels*size*size)
        tokens = self.input(patches)
        tokens = torch.cat([self.cls.expand(batch,-1,-1), tokens], dim=1)
        if condition_token is not None:
            tokens = torch.cat([tokens, condition_token[:,None,:]], dim=1)
        tokens = tokens+self.positions[:,:tokens.shape[1]]
        return self.norm(self.blocks(tokens)[:,0])


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes, width):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(width,n_classes)

    def forward(self, image):
        return self.head(self.encoder(image))


class Discriminator(nn.Module):
    def __init__(self, encoder, width):
        super().__init__()
        self.encoder = encoder
        self.condition = nn.Linear(width,width)
        self.head = nn.Linear(width,1)

    def forward(self, image, embedding):
        return self.head(self.encoder(image,self.condition(embedding))).flatten()


class Generator(nn.Module):
    """잡음 + 조건 임베딩 → 8×8 토큰 → (블록 → pixel shuffle ×2) 두 번 → 32×32 토큰
    → 토큰마다 겹치는 2p×2p 픽셀 패치로 펼쳐 32p×32p RGB (p=8이면 256×256).

    TransGAN처럼 pixel shuffle로 토큰 격자를 키운다(폭은 단계마다 1/4). 32×32 RGB에서 멈추지 않고
    마지막 토큰을 픽셀 패치로 펼치는 것이 논문·이전 버전과 다른 점이다(이전 버전은 32×32 출력을 키워 써서 흐렸다)."""

    def __init__(self, noise_dim, embedding_dim, base_width, depths, heads, checkpointing, patch=8):
        super().__init__()
        assert base_width % 16 == 0 and len(depths) == 3 and patch % 2 == 0
        widths = [base_width, base_width // 4, base_width // 16]
        assert all(width % heads == 0 for width in widths)
        self.noise_dim = noise_dim
        self.input = nn.Linear(noise_dim + embedding_dim, 8 * 8 * base_width)
        self.positions = nn.ParameterList([nn.Parameter(torch.randn(1, (8 * 2 ** i) ** 2, w) * .02)
                                           for i, w in enumerate(widths)])
        self.stages = nn.ModuleList([Blocks(w, heads, d, checkpointing) for w, d in zip(widths, depths)])
        self.norm = nn.LayerNorm(widths[-1])
        self.output = nn.ConvTranspose2d(widths[-1], 3, 2 * patch, stride=patch, padding=patch // 2)

    def forward(self, noise, embedding):
        batch = noise.shape[0]
        tokens = self.input(torch.cat([noise, embedding], dim=1)).reshape(batch, 64, -1)
        for stage, (position, blocks) in enumerate(zip(self.positions, self.stages)):
            tokens = blocks(tokens + position)
            if stage < 2:
                side = math.isqrt(tokens.shape[1])
                grid = nn.functional.pixel_shuffle(tokens.transpose(1, 2).reshape(batch, -1, side, side), 2)
                tokens = grid.flatten(2).transpose(1, 2)
        grid = self.norm(tokens).transpose(1, 2).reshape(batch, -1, 32, 32)
        return self.output(grid).tanh()


def build_models(config, n_classes):
    """(분류기, 생성기, 판별기). 분류기·판별기는 32×32 입력을 4×4 패치 토큰으로 본다."""
    width = config['encoder_width']

    def encoder():
        return PatchEncoder(32, 4, width, config['heads'], config['encoder_depth'], config['gradient_checkpointing'])

    classifier = Classifier(encoder(), n_classes, width)
    discriminator = Discriminator(encoder(), width)
    generator = Generator(config['noise_dim'], width, config['generator_width'], config['generator_depths'],
                          config['heads'], config['gradient_checkpointing'], patch=config['patch'])
    return classifier, generator, discriminator
