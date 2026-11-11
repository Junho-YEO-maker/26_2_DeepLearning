"""Small complete training/generation/evaluation and interruption regression test."""
import copy
import json
from pathlib import Path

import torch
import yaml

from tests.smoke_stage1 import run, must_fail
from src.data.gan_patches import PatchDataset
from src.data.prepare import PCBS, read_csv
from src.generate.ctransgan import ConditionalTransGAN
from src.generate.run import model_from_config
from src.metrics.evaluate import evaluate, validate_outputs
from src.train.ctransgan import train, trusted_checkpoint

REPORT = {}


def exercise(root, splits):
    base = yaml.safe_load(Path('configs/ctransgan_t4.yaml').read_text())
    base.update(profile='dummy_smoke',require_cuda=False,device='cpu',data_root=str(root),
                splits=str(splits),checkpoints=str(root.parent/'checkpoints'),
                output_root=str(root.parent/'outputs'),long_side=32)
    base['network'].update(noise_dim=8,encoder_width=16,encoder_depth=1,generator_width=64,
                           generator_depths=[1,1,1],heads=2,gradient_checkpointing=True)
    base['training'].update(classifier_steps=2,gan_steps=3,batch_size=2,gradient_accumulation=1,
                            amp=False,checkpoint_every=1,log_every=100)
    assert isinstance(model_from_config(base),ConditionalTransGAN)
    for pcb in PCBS:
        reference = copy.deepcopy(base)
        reference['checkpoints'] = str(root.parent/'reference')
        full = train(reference,pcb)
        partial = train(base,pcb,stop_after=3)
        assert partial['classifier_step']==2 and partial['gan_step']==1
        must_fail(lambda: ConditionalTransGAN(base).generate(pcb,4,7017))
        resumed = train(base,pcb)
        for model in ('classifier','generator','discriminator'):
            for key in full['models'][model]:
                assert torch.equal(full['models'][model][key],resumed['models'][model][key]), (pcb,model,key)
        assert full['losses']==resumed['losses']
        assert torch.equal(full['embeddings'],resumed['embeddings'])
        data = PatchDataset(root,splits,pcb,32,2)
        approved = set(data.by_path)
        assert all(record['source_anomaly'] in approved for record in data.records)
        out = ConditionalTransGAN(base).generate(pcb,4,7017)
        old = read_csv(out/'meta.csv')
        ConditionalTransGAN(base).generate(pcb,4,7017)
        assert old==read_csv(out/'meta.csv')
        validate_outputs(splits,out,pcb,4)
        missing = copy.deepcopy(base)
        missing['checkpoints'] = str(root.parent/'missing')
        must_fail(lambda: ConditionalTransGAN(missing).generate(pcb,4,7017))
        REPORT[pcb] = {'classifier_steps':2,'gan_steps':3,'images':4,'checkpoint_resume':'bit-identical',
                       'patches':len(data.records),'parameters':sum(v.numel() for v in full['models']['generator'].values())}
        if pcb=='pcb1':
            metrics = dict(base['metrics'],device='cpu',cpu_threads=2,batch_size=2,
                           kid_subset_size=3,kid_subsets=3,lpips_pairs=4)
            evaluated = evaluate(root,splits,out,pcb,metrics,4,root.parent/'results/ctransgan',root.parent/'figures/ctransgan')
            assert evaluated['validation_status']=='PASS' and evaluated['outside_mask_pixel_changes']==0
    cuda_required = copy.deepcopy(base)
    cuda_required['require_cuda']=True
    cuda_required['device']='cpu'
    try:
        train(cuda_required,'pcb1')
    except RuntimeError:
        pass
    else:
        raise AssertionError('CUDA guard did not reject CPU profile')


if __name__=='__main__':
    torch.set_num_threads(2)
    run(exercise)
    print(json.dumps({'status':'PASS','pcb':REPORT,'checks':[
        'frozen split and heldout access gates','class-specific connected-component training crops',
        'classifier training and data-derived class embeddings','G/D gradient updates with checkpointing',
        'bit-identical interrupted/resumed and continuous CPU training',
        'missing/incomplete checkpoints rejected before generation','deterministic resumed generation',
        'paired output masks and metadata','pretrained KID/FID/LPIPS and structural metrics on dummy data',
        'CUDA-required training rejects CPU environment']},indent=2))
