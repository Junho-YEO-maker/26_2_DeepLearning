"""SegAD(Baitieva et al., CVPR 2024) 비교 규칙 (11/20). 이 규칙으로 낸 PCB별 공개 수치가 있어서 그대로 맞춘다.

규칙: VisA 공식 1cls 분할에서 시험 불량 10장을 무작위로 골라 학습으로 옮기고, 나머지 시험(불량 90 + 정상 약 100)으로 잰다.
      사진 크기 256(SegAD), DRA·DevNet은 공식 코드 기본값(448)으로 n_anomaly 10. SegAD는 seed 10개, 우리는 5개.
고르는 방법은 DRA·DevNet 공식 코드(datasets/mvtecad.py split_outlier)와 똑같이 한다:
  MVTec 형식 폴더의 test/{good,anomaly}에서 os.listdir 순서로 불량 목록 → np.random.RandomState(seed).shuffle → 앞 10장.
그래서 우리 모델과 공개 방법이 같은 seed에서 같은 10장으로 학습한다.

MVTec 형식 폴더(심볼릭 링크): Dataset/cache/mvtec_layout/visa_1cls/{pcb}/{train/good, test/good, test/anomaly}/*.jpg
공정 시험셋: fair_ad 규칙으로 1cls 시험 불량 100장 전부에 대해 만들고(Dataset/cache/fair_official_1cls), seed마다 학습으로 간 10장의 짝을 뺀다.

    python -m AnomalyDetection.ad.segad_protocol      # 폴더·공정 시험셋 만들기, seed별 10장 출력
"""
import os

import numpy as np

from . import fair_ad
from .data import PCBS, ROOT, Data

SPLITS = ROOT / 'Dataset/visa/splits_official/1cls'
DATA_ROOT = ROOT / 'Dataset/visa/VisA_official/VisA_PCB'
LAYOUT = ROOT / 'Dataset/cache/mvtec_layout/visa_1cls'
FAIR_CACHE = ROOT / 'Dataset/cache/fair_official_1cls'
N_ANOMALY = 10
EXT = ('png', 'PNG', 'jpg', 'npy')


def data():
    return Data(DATA_ROOT, splits=SPLITS)


def _name(rel):
    kind = 'anomaly' if '/Anomaly/' in rel else 'good'
    return kind, os.path.splitext(os.path.basename(rel))[0] + '.jpg'


def build_layout(d):
    """MVTec 형식 심볼릭 링크 폴더. 돌려주는 값: {pcb: {'test/anomaly/011.jpg': rel, ...}}"""
    maps = {}
    for pcb in PCBS:
        m = {}
        for split, items in (('train', d.train_normal(pcb)), ('test', d.test(pcb))):
            for it in items:
                kind, name = _name(it.image)
                p = LAYOUT / pcb / split / kind / name
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.symlink_to(d.path(it.image))
                m[f'{split}/{kind}/{name}'] = it.image
        maps[pcb] = m
    return maps


def selection(pcb, seed, n=N_ANOMALY):
    """DRA·DevNet split_outlier와 같은 순서·같은 섞기. 돌려주는 값: 학습으로 간 'test/anomaly/xxx.jpg' 목록."""
    root = LAYOUT / pcb / 'test'
    outlier = []
    for cl in os.listdir(root):
        if cl == 'good':
            continue
        outlier += ['test/' + cl + '/' + f for f in os.listdir(root / cl) if f[-3:] in EXT]
    np.random.RandomState(seed).shuffle(outlier)
    return outlier[:n]


def split(d, maps, pcb, seed):
    """(학습 불량 Item 목록, 시험 Item 목록, 공정 시험셋 dict(학습으로 간 짝 뺌))."""
    sel = {maps[pcb][k] for k in selection(pcb, seed)}
    test = d.test(pcb)
    train_def = [x for x in test if x.image in sel]
    rest = [x for x in test if x.image not in sel]
    assert len(train_def) == N_ANOMALY
    fair_ad.CACHE = FAIR_CACHE
    fs = fair_ad.fair_set(d, pcb)
    keep = np.array([im not in sel for im in fs['image']])
    fs = {k: v[keep] for k, v in fs.items()}
    assert list(fs['image']) == [x.image for x in rest if x.label == 1]
    return train_def, rest, fs


if __name__ == '__main__':
    d = data()
    maps = build_layout(d)
    fair_ad.CACHE = FAIR_CACHE
    for pcb in PCBS:
        fair_ad.fair_set(d, pcb, 16)
        for s in range(5):
            print(pcb, s, [maps[pcb][k].rsplit('/', 1)[1] for k in selection(pcb, s)])
