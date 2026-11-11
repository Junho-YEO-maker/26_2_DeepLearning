"""Classifier/embedding/GAN training with complete optimizer and RNG resumes."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from src.data.gan_patches import PatchDataset
from src.data.prepare import PCBS, sha
from src.models.copy_paste import atomic_json
from src.models.ctransgan import build_models


def implementation_signature():
    root = Path(__file__).resolve().parents[1]
    files = ['data/gan_patches.py','models/ctransgan.py','train/ctransgan.py']
    return hashlib.sha256(''.join(sha(root/p) for p in files).encode()).hexdigest()


def training_signature(config, dataset):
    selected = {key: config[key] for key in ('seed','network','training','patch_size','context_pixels')}
    selected['data'] = dataset.fingerprint
    selected['implementation'] = implementation_signature()
    return hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()


def device_for(config):
    if config['require_cuda'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this training profile. Select a Colab GPU runtime.')
    requested = config.get('device','auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else torch.device(requested)
    if config['require_cuda'] and device.type!='cuda':
        raise RuntimeError('Training profile requires a CUDA device')
    return device


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def trusted_checkpoint(path):
    # Only load locally generated project checkpoints, including optimizer/RNG objects.
    return torch.load(path,map_location='cpu',weights_only=False)


def train(config, pcb, stop_after=None):
    assert config['seed']==7017 and config['patch_size']==32
    assert config['training']['classifier_steps']>0 and config['training']['gan_steps']>0
    device = device_for(config)
    random.seed(7017)
    np.random.seed(7017)
    torch.manual_seed(7017)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(7017)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    dataset = PatchDataset(config['data_root'],config['splits'],pcb,32,config['context_pixels'])
    signature = training_signature(config,dataset)
    directory = Path(config['checkpoints'])/pcb
    directory.mkdir(parents=True,exist_ok=True)
    latest = directory/'latest.pt'
    models = build_models(config['network'],len(dataset.class_ids))
    classifier,generator,discriminator = [model.to(device) for model in models]
    settings = config['training']
    def optimizer(model):
        return torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],betas=tuple(settings['betas']))
    optimizers = {'classifier':optimizer(classifier),'generator':optimizer(generator),'discriminator':optimizer(discriminator)}
    enabled = settings['amp'] and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda',enabled=enabled)
    state = {'schema':1,'pcb':pcb,'signature':signature,'config':config,'dataset':dataset.describe(),
             'classifier_step':0,'gan_step':0,'embeddings':None,'losses':[],
             'training_seconds':{'classifier':0.,'gan':0.},'gpu_peak_bytes':None,
             'device':str(device),'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None}
    if latest.exists():
        state = trusted_checkpoint(latest)
        assert state['signature']==signature and state['pcb']==pcb, 'Checkpoint config/data/code mismatch'
        for name,model in zip(('classifier','generator','discriminator'),models):
            model.load_state_dict(state['models'][name])
            optimizers[name].load_state_dict(state['optimizers'][name])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
    else:
        atomic_json(directory/'dataset.json',dataset.describe())

    def save():
        state['models'] = {name:model.state_dict() for name,model in zip(('classifier','generator','discriminator'),models)}
        state['optimizers'] = {name:opt.state_dict() for name,opt in optimizers.items()}
        state['scaler'] = scaler.state_dict()
        state['rng'] = rng_state()
        if device.type=='cuda':
            state['gpu_peak_bytes'] = max(state.get('gpu_peak_bytes') or 0,torch.cuda.max_memory_allocated())
        tmp = latest.with_suffix('.tmp')
        torch.save(state,tmp)
        tmp.replace(latest)
        atomic_json(directory/'progress.json',{key:state[key] for key in ('pcb','classifier_step','gan_step','training_seconds','gpu_peak_bytes','device','dataset')})

    def autocast():
        return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled)

    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize()

    def finite(loss):
        assert torch.isfinite(loss).all(), 'Nonfinite training loss; previous checkpoint is preserved'

    updates = 0
    accumulation = settings['gradient_accumulation']
    for step in range(state['classifier_step'],settings['classifier_steps']):
        sync()
        start = time.perf_counter()
        classifier.train()
        opt = optimizers['classifier']
        opt.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(accumulation):
            images,targets = dataset.batch(settings['batch_size'],device)
            with autocast():
                loss = nn.functional.cross_entropy(classifier(images),targets)/accumulation
            finite(loss)
            scaler.scale(loss).backward()
            total += float(loss.detach())
        scaler.step(opt)
        scaler.update()
        sync()
        state['training_seconds']['classifier'] += time.perf_counter()-start
        state['classifier_step'] = step+1
        state['losses'].append({'stage':'classifier','step':step+1,'loss':total})
        updates += 1
        if (step+1)%settings['checkpoint_every']==0 or step+1==settings['classifier_steps']:
            save()
        if stop_after is not None and updates>=stop_after:
            save()
            return state
    if state['embeddings'] is None:
        classifier.eval()
        features = []
        with torch.no_grad():
            for start in range(0,len(dataset.records),settings['batch_size']):
                images = dataset.images[start:start+settings['batch_size']].to(device)
                features.append(classifier.encoder(images).cpu())
        features = torch.cat(features)
        embeddings = torch.stack([features[dataset.targets==i].mean(0) for i in range(len(dataset.class_ids))])
        assert torch.isfinite(embeddings).all()
        state['embeddings'] = embeddings
        discriminator.encoder.load_state_dict(classifier.encoder.state_dict())
        save()
    embeddings = state['embeddings'].to(device)
    bce = nn.BCEWithLogitsLoss()
    for step in range(state['gan_step'],settings['gan_steps']):
        sync()
        start = time.perf_counter()
        generator.train()
        discriminator.train()
        opt_d,opt_g = optimizers['discriminator'],optimizers['generator']
        opt_d.zero_grad(set_to_none=True)
        d_total,g_total = 0.,0.
        for _ in range(accumulation):
            real,targets = dataset.batch(settings['batch_size'],device)
            condition = embeddings[targets]
            noise = torch.randn(len(real),config['network']['noise_dim'],device=device)
            with autocast():
                fake = generator(noise,condition).detach()
                real_logits = discriminator(real,condition)
                fake_logits = discriminator(fake,condition)
                loss = (bce(real_logits,torch.ones_like(real_logits))+bce(fake_logits,torch.zeros_like(fake_logits)))/(2*accumulation)
            finite(loss)
            scaler.scale(loss).backward()
            d_total += float(loss.detach())
        scaler.step(opt_d)
        scaler.update()
        opt_g.zero_grad(set_to_none=True)
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            for _ in range(accumulation):
                _,targets = dataset.batch(settings['batch_size'],device)
                condition = embeddings[targets]
                noise = torch.randn(len(targets),config['network']['noise_dim'],device=device)
                with autocast():
                    logits = discriminator(generator(noise,condition),condition)
                    loss = bce(logits,torch.ones_like(logits))/accumulation
                finite(loss)
                scaler.scale(loss).backward()
                g_total += float(loss.detach())
            scaler.step(opt_g)
            scaler.update()
        finally:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
        sync()
        state['training_seconds']['gan'] += time.perf_counter()-start
        state['gan_step'] = step+1
        state['losses'].append({'stage':'gan','step':step+1,'d_loss':d_total,'g_loss':g_total})
        updates += 1
        if (step+1)%settings['checkpoint_every']==0 or step+1==settings['gan_steps']:
            save()
        if (step+1)%settings['log_every']==0:
            print(pcb,step+1,'D',d_total,'G',g_total,flush=True)
        if stop_after is not None and updates>=stop_after:
            save()
            return state
    save()
    return state


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/ctransgan_t4.yaml')
    parser.add_argument('--pcb',choices=PCBS)
    args = parser.parse_args()
    configuration = yaml.safe_load(Path(args.config).read_text())
    for pcb in ([args.pcb] if args.pcb else PCBS):
        result = train(configuration,pcb)
        print(pcb,'completed GAN steps',result['gan_step'],flush=True)
