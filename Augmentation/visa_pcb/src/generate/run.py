import argparse
from pathlib import Path
import yaml
from src.models.copy_paste import CopyPaste
from src.data.prepare import PCBS


def model_from_config(config):
    if config.get('model','copy_paste')=='ctransgan':
        from src.generate.ctransgan import ConditionalTransGAN
        return ConditionalTransGAN(config)
    if config.get('model','copy_paste')!='copy_paste':
        raise ValueError('Unknown generation model')
    return CopyPaste(config['data_root'], config['splits'], config['output_root'],
                     config['long_side'], config['rotation_degrees'], config['feather_pixels'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage2.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--pcb', choices=PCBS)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['data_root'] = args.data_root or cfg['data_root']
    model = model_from_config(cfg)
    for pcb in ([args.pcb] if args.pcb else PCBS):
        print(model.generate(pcb, cfg['n_per_pcb'], cfg['seed']), flush=True)
