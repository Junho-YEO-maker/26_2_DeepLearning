"""Generate annotated composites from a completed per-PCB cTransGAN checkpoint."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import yaml

from src.data.access import a_rows, open_a_pair
from src.data.gan_patches import PatchDataset
from src.data.prepare import PCBS, read_csv, sha
from src.models.copy_paste import FIELDS, atomic_json, size_for
from src.models.ctransgan import build_models
from src.train.ctransgan import trusted_checkpoint, training_signature


class ConditionalTransGAN:
    name = 'ctransgan'

    def __init__(self, config):
        assert config['feather_pixels']>0
        self.config = config

    def generate(self, pcb, n=500, seed=7017):
        assert seed==7017 and n>0
        config = self.config
        checkpoint_path = Path(config['checkpoints'])/pcb/'latest.pt'
        assert checkpoint_path.is_file(), f'Train {pcb} before generation'
        state = trusted_checkpoint(checkpoint_path)
        plan = config['training']
        assert state['classifier_step']==plan['classifier_steps'] and state['gan_step']==plan['gan_steps'], 'Training plan is incomplete'
        dataset = PatchDataset(config['data_root'],config['splits'],pcb,32,config['context_pixels'])
        assert state['signature']==training_signature(config,dataset), 'Checkpoint config/data/code mismatch'
        _,generator,_ = build_models(config['network'],len(dataset.class_ids))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator.load_state_dict(state['models']['generator'])
        generator = generator.to(device).eval()
        embeddings = state['embeddings'].to(device)
        normal = a_rows(config['splits'],pcb,'generate_condition','normal')
        out = Path(config['output_root'])/self.name/pcb
        for folder in ('images','masks','records'):
            (out/folder).mkdir(parents=True,exist_ok=True)
        signature = {'checkpoint_sha256':sha(checkpoint_path),'split_sha256':sha(Path(config['splits'])/f'{pcb}_split.csv'),
                     'seed':seed,'long_side':config['long_side'],'feather_pixels':config['feather_pixels'],
                     'implementation_sha256':sha(Path(__file__))}
        if (out/'run.json').exists():
            assert json.loads((out/'run.json').read_text())==signature, 'Use a fresh output directory for changed generation settings'
        else:
            atomic_json(out/'run.json',signature)
        assert all(int(p.stem)<n for p in (out/'records').glob('*.json')), 'Refusing to shrink output'
        rows = []
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats()
        for index in range(n):
            name = f'{index:06d}.png'
            record_path = out/'records'/f'{index:06d}.json'
            if record_path.exists():
                row = json.loads(record_path.read_text())
                assert sha(out/'images'/name)==row['image_sha256'] and sha(out/'masks'/name)==row['mask_sha256']
                rows.append(row)
                continue
            if device.type=='cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence([seed,int(pcb[-1]),index]))
            nr = normal[int(rng.integers(len(normal)))]
            target,_ = open_a_pair(config['data_root'],config['splits'],pcb,nr,'generate_condition')
            original = target.size
            target = target.resize(size_for(original,config['long_side']),Image.Resampling.LANCZOS)
            for attempt in range(64):
                source = dataset.records[int(rng.integers(len(dataset.records)))]
                mask = dataset.component_mask(source)
                source_row = dataset.by_path[source['source_anomaly']]
                assert (int(source_row['width']),int(source_row['height']))==original
                left,top,right,bottom = source['box']
                box = [round(left*target.width/original[0]),round(top*target.height/original[1]),
                       min(target.width,max(1,round(right*target.width/original[0]))),
                       min(target.height,max(1,round(bottom*target.height/original[1])))]
                box[0] = min(box[0],target.width-1)
                box[1] = min(box[1],target.height-1)
                box[2] = max(box[0]+1,box[2])
                box[3] = max(box[1]+1,box[3])
                support = np.asarray(mask.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.NEAREST))>0
                if support.any():
                    break
            else:
                raise RuntimeError('No defect mask survives output resizing')
            cls = dataset.class_ids.index(source['class_id'])
            random = torch.Generator(device='cpu').manual_seed(seed+100000*int(pcb[-1])+index)
            noise = torch.randn(1,config['network']['noise_dim'],generator=random).to(device)
            with torch.inference_mode():
                prediction = generator(noise,embeddings[cls:cls+1])[0]
            assert torch.isfinite(prediction).all()
            pixels = ((prediction.cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype('uint8')
            patch = Image.fromarray(pixels).resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
            blend = np.minimum(1,distance_transform_edt(np.pad(support,1))[1:-1,1:-1]/config['feather_pixels'])*support
            before = np.asarray(target)
            result = before.copy()
            region = np.s_[box[1]:box[3],box[0]:box[2]]
            result[region] = np.rint(before[region]*(1-blend[...,None])+np.asarray(patch)*blend[...,None]).clip(0,255).astype('uint8')
            mask = np.zeros((target.height,target.width),dtype='uint8')
            mask[region] = support.astype('uint8')*255
            assert np.array_equal(before[mask==0],result[mask==0]) and np.any(result!=before)
            condition = {'class_id':source['class_id'],'class_name':source['class_name'],
                         'embedding':'mean penultimate classifier features',
                         'source_box':source['box'],'output_box':box,'component_id':source['component_id'],
                         'location':'source annotation coordinates on a gen_train normal',
                         'mask_semantics':'synthetic insertion support','semantic_mask_verified':False,
                         'checkpoint_sha256':signature['checkpoint_sha256']}
            for folder,image in [('images',Image.fromarray(result)),('masks',Image.fromarray(mask))]:
                tmp = out/folder/(name+'.tmp')
                image.save(tmp,format='PNG')
                tmp.replace(out/folder/name)
            image_digest = sha(out/'images'/name)
            mask_digest = sha(out/'masks'/name)
            if device.type=='cuda':
                torch.cuda.synchronize()
            row = {'filename':name,'source_normal':nr['file_path'],'condition':json.dumps(condition,sort_keys=True),
                   'seed':seed,'source_anomaly':source['source_anomaly'],'source_mask':source['source_mask'],
                   'source_defect_type':source['class_name'],'original_width':original[0],'original_height':original[1],
                   'output_width':target.width,'output_height':target.height,'sample_index':index,
                   'resize':json.dumps({'long_side':config['long_side'],'image':'LANCZOS','mask':'NEAREST','crop':False,
                                        'generated_patch':'32x32 RGB upsampled to annotated crop'}),
                   'generation_seconds':time.perf_counter()-start,
                   'image_sha256':image_digest,'mask_sha256':mask_digest}
            atomic_json(record_path,row)
            rows.append(row)
        temporary = out/'meta.csv.tmp'
        with temporary.open('w',newline='',encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file,fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out/'meta.csv')
        atomic_json(out/'cost.json',{'training_status':'completed configured step budget; quality requires evaluation',
                    'training_seconds':state['training_seconds'],'training_gpu_peak_bytes':state['gpu_peak_bytes'],
                    'generation_seconds_sum':sum(float(r['generation_seconds']) for r in rows),
                    'seconds_per_image':sum(float(r['generation_seconds']) for r in rows)/n,'n':n,'device':str(device),
                    'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None})
        return out


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/ctransgan_t4.yaml')
    parser.add_argument('--pcb',choices=PCBS)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model = ConditionalTransGAN(config)
    for pcb in ([args.pcb] if args.pcb else PCBS):
        print(model.generate(pcb,config['n_per_pcb'],7017),flush=True)
