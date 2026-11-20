"""생성 판정 검증 (11/19): 같은 장수(종류당 20장)를 '덜 튀는 것'과 '더 튀는 것'으로 골라 위치 seg를 다시 돌린다.

gen_quality_smt에서 SMT 합성은 마스크 안이 실제 결함보다 훨씬 이상해 보일수록(in_anom 높을수록) 이득이 작았다.
그 점수로 고른 세트가 정말 나은지 본다. 같은 생성 40장 중에서
  {gen}_qlow   종류마다 in_anom 낮은 20장 (덜 튀는 것)
  {gen}_qhigh  종류마다 in_anom 높은 20장 (더 튀는 것)
비교 대상(이미 있음): {gen}:20 = manifest 앞 20장(품질과 무관한 순서), {gen}:40 = 전부, none = 실제만.
manifest만 만든다(이미지는 원래 경로). Dataset/aihub_smt/generated/ 아래라 git에 안 올라간다.

    python -m AnomalyDetection.smt.quality_subset            # 세트 만들기
    python -m AnomalyDetection.smt.quality_subset --compare  # 결과 비교 (seg_ad 실행 뒤)
"""
import argparse
import csv

from ..ad.data import ROOT, read_csv
from .data import GENERATED

RESULTS = ROOT / 'AnomalyDetection/results/smt'
GENS = ('sd_lora_fix', 'ctransgan')
KS = [(k, s) for k in (1, 4, 16) for s in (0, 1, 2)]
N = 20


def make():
    import pandas as pd
    sc = pd.read_csv(RESULTS / 'gen_quality_smt_images.csv').set_index('image').in_anom.to_dict()
    for g in GENS:
        for k, s in KS:
            src = GENERATED / g / f'K{k}_seed{s}' / 'manifest.csv'
            rows = [r for r in read_csv(src) if r['label'] == '1']
            fields = list(rows[0])
            for tag, rev in (('qlow', False), ('qhigh', True)):
                out = []
                for key in sorted({(r['category'], r['defect_type']) for r in rows}):
                    rt = sorted([r for r in rows if (r['category'], r['defect_type']) == key],
                                key=lambda r: sc[r['image']], reverse=rev)
                    out += rt[:N]
                d = GENERATED / f'{g}_{tag}' / f'K{k}_seed{s}'
                d.mkdir(parents=True, exist_ok=True)
                with open(d / 'manifest.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(out)
            print(f'{g} K{k} seed{s}: qlow/qhigh 각 {len(out)}행')


def compare():
    import pandas as pd
    lines = ['# 생성 판정 검증: 같은 20장을 점수로 고르면 나아지나 (quality_subset.py)', '',
             '위치 seg, 표준 시험셋 미검률(과검 1%), 낮을수록 좋음. seed 3개 평균. '
             'qlow = 마스크 안이 덜 튀는 20장, 앞20 = 품질과 무관한 20장, qhigh = 더 튀는 20장.', '']
    allrows = []
    for cat in ('pre', 'solder'):
        d = pd.read_csv(RESULTS / f'seg_ad_{cat}.csv')
        for g in GENS:
            for k, s in KS:
                x = d[(d.K == k) & (d.seed == s)]
                get = lambda c: x[x.cond == c].fnr.iloc[-1] if (x.cond == c).any() else float('nan')   # noqa: E731
                allrows.append(dict(cat=cat, gen=g, K=k, seed=s, none=get('none'), all40=get(f'{g}:40:0.5'),
                                    first20=get(f'{g}:20:0.5'), qlow=get(f'{g}_qlow:20:0.5'), qhigh=get(f'{g}_qhigh:20:0.5')))
    a = pd.DataFrame(allrows)
    a.to_csv(RESULTS / 'quality_subset_compare.csv', index=False)
    cols = ['none', 'qhigh', 'first20', 'qlow', 'all40']
    t = a.groupby(['cat', 'gen', 'K'])[cols].mean()
    lines += ['| 공정 | 생성기 | K | 실제만 | 더 튀는 20 | 앞 20 | **덜 튀는 20** | 전부 40 |', '|---|---|---|---|---|---|---|---|']
    for (cat, g, k), r in t.iterrows():
        lines.append(f'| {cat} | {g} | {k} | ' + ' | '.join(f'{r[c]:.3f}' for c in cols) + ' |')
    ok = a.dropna(subset=['qlow', 'qhigh', 'first20'])
    lines += ['', f'칸 {len(ok)}개 (공정 × 생성기 × K × seed) 중',
              f'- 덜 튀는 20 < 더 튀는 20: {int((ok.qlow < ok.qhigh).sum())}개, 평균 차 {(ok.qhigh - ok.qlow).mean():+.3f}',
              f'- 덜 튀는 20 < 앞 20: {int((ok.qlow < ok.first20).sum())}개, 평균 차 {(ok.first20 - ok.qlow).mean():+.3f}',
              f'- 덜 튀는 20 < 전부 40: {int((ok.qlow < ok.all40).sum())}개, 평균 차 {(ok.all40 - ok.qlow).mean():+.3f}',
              f'- 덜 튀는 20 < 실제만: {int((ok.qlow < ok.none).sum())}개 / 앞 20 < 실제만: {int((ok.first20 < ok.none).sum())}개', '']
    (ROOT / 'AnomalyDetection/results/ppt/quality_subset_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', action='store_true')
    compare() if ap.parse_args().compare else make()
