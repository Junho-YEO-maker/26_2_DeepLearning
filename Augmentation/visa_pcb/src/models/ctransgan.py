"""Compact paper-guided cTransGAN with classifier-derived class embeddings."""
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
    def __init__(self, noise_dim, embedding_dim, base_width, depths, heads, checkpointing):
        super().__init__()
        assert base_width%16 == 0 and len(depths)==3
        widths = [base_width,base_width//4,base_width//16]
        assert all(width%heads == 0 for width in widths)
        self.noise_dim = noise_dim
        self.input = nn.Linear(noise_dim+embedding_dim,8*8*base_width)
        self.positions = nn.ParameterList([nn.Parameter(torch.randn(1,(8*2**i)**2,w)*.02) for i,w in enumerate(widths)])
        self.stages = nn.ModuleList([Blocks(w,heads,d,checkpointing) for w,d in zip(widths,depths)])
        self.norm = nn.LayerNorm(widths[-1])
        self.output = nn.Linear(widths[-1],3)

    def forward(self, noise, embedding):
        batch = noise.shape[0]
        tokens = self.input(torch.cat([noise,embedding],dim=1)).reshape(batch,64,-1)
        for stage, (position, blocks) in enumerate(zip(self.positions,self.stages)):
            tokens = blocks(tokens+position)
            if stage<2:
                side = math.isqrt(tokens.shape[1])
                grid = tokens.transpose(1,2).reshape(batch,-1,side,side)
                tokens = nn.functional.pixel_shuffle(grid,2).flatten(2).transpose(1,2)
        rgb = self.output(self.norm(tokens)).tanh()
        return rgb.transpose(1,2).reshape(batch,3,32,32)


def build_models(config, n_classes):
    width = config['encoder_width']
    def encoder():
        return PatchEncoder(32,4,width,config['heads'],config['encoder_depth'],config['gradient_checkpointing'])
    classifier = Classifier(encoder(),n_classes,width)
    discriminator = Discriminator(encoder(),width)
    generator = Generator(config['noise_dim'],width,config['generator_width'],
                          config['generator_depths'],config['heads'],config['gradient_checkpointing'])
    return classifier,generator,discriminator
