"""Write a compact report from completed evaluations."""
import csv
import json
from pathlib import Path
from src.data.prepare import PCBS


def main():
    rows = []
    for pcb in PCBS:
        report = json.loads(Path(f'results/{pcb}_metrics.json').read_text())
        assert report['validation_status'] == 'PASS'
        assert report['n_generated'] == 500 and report['n_real_reference'] == 33
        keys = ('pcb', 'n_generated', 'n_real_reference', 'kid_mean', 'kid_std',
                'fid', 'lpips_mean', 'lpips_pair_std', 'lpips_pairs',
                'source_normal_psnr_mean', 'source_normal_ssim_mean',
                'outside_mask_pixel_changes', 'metric_device', 'evaluation_seconds')
        row = {key: report[key] for key in keys}
        row['generation_seconds_per_image'] = report['generation_cost']['seconds_per_image']
        row['training_status'] = report['generation_cost']['training_status']
        row['generation_gpu_peak_bytes'] = report['generation_cost']['gpu_peak_memory_bytes']
        rows.append(row)
    with Path('results/baseline_summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 단계 2 복사·붙여넣기 평가', '',
             '고정 분할의 gen_train 불량 33장과 생성 500장을 PCB별로 비교했습니다.', '',
             '| PCB | 생성 수 | KID 평균 ± 표준편차 | FID | LPIPS 평균 | CPU 초/장 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['pcb']}|{r['n_generated']}|{r['kid_mean']:.5f} ± {r['kid_std']:.5f}|"
                     f"{r['fid']:.2f}|{r['lpips_mean']:.4f}|{r['generation_seconds_per_image']:.3f}|")
    lines += ['', 'KID는 25장씩 비복원 추출을 100회 반복한 원 단위입니다. LPIPS는 생성 이미지의 무순서 1,000쌍 평균입니다. FID는 참조가 33장인 조건의 보조 지표입니다.',
              '', '| PCB | 원본 정상과의 PSNR (dB) | 원본 정상과의 SSIM | 마스크 밖 변경 픽셀 |',
              '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['pcb']}|{r['source_normal_psnr_mean']:.3f}|{r['source_normal_ssim_mean']:.6f}|0|")
    lines += ['', 'PSNR/SSIM은 생성 이미지와 대응하는 원본 정상 사이의 구조 보존을 측정합니다.',
              '', '이미지와 이진 마스크 각 2,000장에 대해 파일 해시, 해상도, 대응 메타데이터, gen_train 출처를 검사했습니다. 마스크 밖 픽셀은 같은 크기로 변환한 원본 정상과 일치했습니다.',
              '', '배치 위치는 이미지 전체에서 고릅니다. pcb1의 첫 샘플에는 기판 바깥 배경에 놓인 결함이 보입니다. LPIPS에는 정상 배경의 차이도 반영되며, KID/FID의 진짜 참조와 결함 제공 풀이 같습니다.',
              '', 'CPU에서 생성과 평가를 실행했습니다. 생성 시간은 4개 PCB 작업이 동시에 실행된 환경에서 로딩, 변환, PNG 저장과 해시 계산을 포함한 샘플 시간입니다. 학습 시간과 GPU 메모리 항목은 기준선에 해당하지 않습니다.',
              '', 'Colab T4/A100과 cTransGAN, UniPCB, 인페인팅+LoRA는 미실행입니다.',
              '', '세부 설정과 출처는 STAGE2_METHOD.md, B팀 전달 규약은 B_TEAM_HANDOFF.md에서 확인할 수 있습니다.']
    Path('reports/stage2_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:11]))


if __name__ == '__main__':
    main()
