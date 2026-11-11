"""Technical real-input check only. Temporary one-step models are never exported."""
import copy
import json
import tempfile
from pathlib import Path

import torch
import yaml

from src.data.prepare import PCBS
from src.train.ctransgan import train


if __name__=='__main__':
    torch.set_num_threads(2)
    base = yaml.safe_load(Path('configs/ctransgan_t4.yaml').read_text())
    base.update(data_root='../raw/VisA',require_cuda=False,device='cpu',profile='real_input_functional_check')
    base['training'].update(classifier_steps=1,gan_steps=1,batch_size=1,
                            gradient_accumulation=1,amp=False,checkpoint_every=1)
    report = {'purpose':'technical real-input forward/backward check; not quality training',
              'device':'CPU','exported_images':0,'production_training':'미실행','pcb':{}}
    with tempfile.TemporaryDirectory() as temporary:
        for pcb in PCBS:
            config = copy.deepcopy(base)
            config['checkpoints'] = temporary
            state = train(config,pcb)
            report['pcb'][pcb] = {'dataset':state['dataset'],'classifier_updates':state['classifier_step'],
                                  'gan_updates':state['gan_step'],'finite_loss':all(
                                      all(torch.isfinite(torch.tensor(value)) for key,value in row.items() if isinstance(value,float))
                                      for row in state['losses']),
                                  'training_seconds':state['training_seconds']}
            print(pcb,state['dataset'],flush=True)
    report['status'] = 'PASS'
    Path('reports/real_stage3_functional.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
