"""Combine completed metrics and explicit unexecuted cTransGAN status."""
import csv
import json
from pathlib import Path
from src.data.prepare import PCBS


def summarize():
    rows = []
    for model in ('copy_paste','ctransgan'):
        for pcb in PCBS:
            path = Path('results')/(f'{pcb}_metrics.json' if model=='copy_paste' else f'ctransgan/{pcb}_metrics.json')
            row = {'model':model,'pcb':pcb,'status':'미실행','reason':'GPU 본학습과 500장 생성·평가 대기',
                   'n_generated':'미실행','kid_mean':'미실행','kid_std':'미실행',
                   'fid':'미실행','lpips_mean':'미실행','source_normal_psnr':'미실행',
                   'source_normal_ssim':'미실행','generation_seconds_per_image':'미실행',
                   'psnr_ssim_note':'source_normal과의 구조 보존값; 생성 불량 vs 진짜 불량의 1:1 대응 없음'}
            if path.exists():
                report = json.loads(path.read_text())
                assert report['validation_status']=='PASS' and report['model']==model and report['pcb']==pcb
                assert report['n_generated']==500 and report['n_real_reference']==33, 'Benchmark requires 500 generated / 33 gen_train references'
                row.update(status='실행 완료',reason='',n_generated=report['n_generated'],
                           kid_mean=report['kid_mean'],kid_std=report['kid_std'],fid=report['fid'],
                           lpips_mean=report['lpips_mean'],source_normal_psnr=report['source_normal_psnr_mean'],
                           source_normal_ssim=report['source_normal_ssim_mean'],
                           generation_seconds_per_image=report['generation_cost']['seconds_per_image'])
            rows.append(row)
    with Path('results/stage3_comparison.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote results/stage3_comparison.csv')


if __name__=='__main__':
    summarize()
