"""공통 평가 모듈 (README 6절, 12절).

주 지표: 과검률(FPR) 1%에서의 미검률(FNR). 시험 ROC에서 FPR ≤ 1%를 만족하는 가장 낮은 임계값을 쓴다.
보조: 종류별·macro 미검률(같은 임계값), val_normal로 정한 임계값의 실운영 FNR/FPR, AUROC, AP.
results.csv / results_per_type.csv 에는 추가만 한다.
"""
import csv
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ROOT, TYPES

RESULTS = ROOT / 'AnomalyDetection/results'
FIELDS = ['model', 'regime', 'category', 'K', 'seed', 'aug', 'n_generated', 'res',
          'fnr_at_fpr1', 'macro_fnr_at_fpr1', 'fnr_calib', 'fpr_calib', 'auroc', 'ap',
          'git_commit', 'timestamp']
TYPE_FIELDS = ['model', 'regime', 'category', 'K', 'seed', 'aug', 'res', 'defect_type', 'n_test', 'fnr_at_fpr1']


def threshold_at_fpr(normal_scores, fpr=0.01):
    """정상 중 임계값을 '넘는'(>) 비율이 fpr 이하가 되는 가장 낮은 임계값."""
    s = np.sort(np.asarray(normal_scores))[::-1]
    n_allowed = int(np.floor(fpr * len(s)))          # 넘어도 되는 정상 장수
    return s[n_allowed] if n_allowed < len(s) else -np.inf


def evaluate(test_scores, test_labels, test_types, val_scores, fpr=0.01):
    y = np.asarray(test_labels)
    s = np.asarray(test_scores, dtype=float)
    thr = threshold_at_fpr(s[y == 0], fpr)
    pred = s > thr
    fnr = float(np.mean(~pred[y == 1]))
    per_type = {}
    for t in TYPES:
        idx = np.array([y[i] == 1 and t in test_types[i].split('|') for i in range(len(y))])
        per_type[t] = (int(idx.sum()), float(np.mean(~pred[idx])) if idx.any() else float('nan'))
    thr_c = threshold_at_fpr(np.asarray(val_scores, dtype=float), fpr)
    pred_c = s > thr_c
    return dict(
        fnr_at_fpr1=fnr,
        macro_fnr_at_fpr1=float(np.nanmean([v[1] for v in per_type.values()])),
        fnr_calib=float(np.mean(~pred_c[y == 1])),
        fpr_calib=float(np.mean(pred_c[y == 0])),
        auroc=float(roc_auc_score(y, s)),
        ap=float(average_precision_score(y, s)),
        per_type=per_type,
    )


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ''


def _append(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def record(key, metrics, results_dir=RESULTS):
    """key: model, regime, category, K, seed, aug, n_generated, res"""
    row = dict(key, **{k: round(v, 6) for k, v in metrics.items() if k != 'per_type'},
               git_commit=git_commit(), timestamp=dt.datetime.now().isoformat(timespec='seconds'))
    _append(Path(results_dir) / 'results.csv', FIELDS, [row])
    trows = [dict({k: key[k] for k in TYPE_FIELDS if k in key}, defect_type=t, n_test=n, fnr_at_fpr1=round(v, 6))
             for t, (n, v) in metrics['per_type'].items()]
    _append(Path(results_dir) / 'results_per_type.csv', TYPE_FIELDS, trows)
    return row


def done_keys(results_dir=RESULTS):
    """이미 기록된 실행(재실행 건너뛰기용)."""
    p = Path(results_dir) / 'results.csv'
    if not p.exists():
        return set()
    with open(p, newline='', encoding='utf-8') as f:
        return {(r['model'], r['category'], r['K'], r['seed'], r['aug'], r['res']) for r in csv.DictReader(f)}
