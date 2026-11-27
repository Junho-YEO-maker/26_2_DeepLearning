"""발표용 AIHub SMT 예시 사진. 모두 "같은 부품 자리를 찍은 정상 사진과 불량 사진" 짝이다 (불량은 결함 상자를 빨간 선으로).
  - 본편 RQ3 데이터 슬라이드: 공정마다 짝 하나 (smt_{공정}_{normal,defect}.jpg)
  - 부록: 공정 · 결함 유형(18종)마다 짝 하나 (smt_type_{공정}_{유형}_{normal,defect}.jpg)

짝 찾기: AIHub 원본 전체(Dataset/aihub_smt/index.csv)에서 불량 사진마다 사진 지문(dHash, 64비트)이 가장 가까운 정상 사진을
찾고(pair_candidates), 가까운 후보를 눈으로 보고 부품 번호(C93, L17 …)까지 같은 자리인 짝을 PAIRS에 적었다.
그래서 실험 분할표(test.csv 등)에 없는 사진도 섞여 있다. 예시로만 쓴다.

AIHub 약관상 SMT 사진은 제3자에게 줄 수 없어서 ppt_figures/local_only/ 에 만든다 (git 제외, .gitignore).
원본이 있는 서버에서만 돌아간다 (Dataset/aihub_smt/raw, 받는 법은 Dataset/aihub_smt/raw/README.md).
build_deck_v4.py는 이 파일들이 있으면 사진을 넣은 발표용 pptx(…_local.pptx)를 따로 만든다.

    python ppt_build/make_smt_figures.py                 # 그림 만들기
    python ppt_build/make_smt_figures.py --candidates    # 유형마다 짝 후보(지문 거리 순) 보기

모델이 본 곳(SMT CAM) 그림도 여기서 만든다 (cam_panels). 먼저 AnomalyDetection/smt/cam.py를 K=16으로 돌려
Dataset/aihub_smt/cam/K16_seed0_{공정}_types.{jpg,npz}가 있어야 한다. 없으면 건너뛴다.
  - 본편: 공정마다 예시 하나를 [사진 | DevNet Grad-CAM | 위치 모델 지도] 세 장으로 (smt_cam_{공정}_{photo,devnet,seg}.jpg)
  - 부록: 유형 6종씩 [사진 | DevNet | 위치 모델] 그림 (smt_cam_{공정}_K16_types_{1,2}.jpg, 사전 1장 · 납땜 2장)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ppt_figures' / 'local_only'
TYPES = {'pre': {'nm': '미납', 'nb': '납부족', 'ns': '납쇼트', 'pb': '납볼', 'om': '위치 밀림', 'mf': '모양 불량'},
         'solder': {'nm': '미납', 'cc': '냉납', 'ml': '밀림', 'sh': '쇼트', 'ws': '오삽', 'ms': '미삽', 'rs': '역삽', 'fl': '뒤집힘',
                    'st': '일어섬', 'pb': '납볼', 'ph': '핀홀', 'ic': '납고드름'}}
# (공정, 유형): (불량 사진, 같은 자리 정상 사진). 괄호 안은 사진에 보이는 부품 번호
PAIRS = {('pre', 'nm'): ('PR_DEF_NM_A_20250902-160908_01458.jpg', 'PR_NOR_PB_A_20250902-164401_01023.jpg'),     # L22 · R41
         ('pre', 'nb'): ('PR_DEF_NB_A_20250820-125056_00073.JPG', 'PR_NOR_NS_A_20250820-155400_00094.JPG'),     # C135
         ('pre', 'ns'): ('PR_DEF_NS_A_20250902-135336_01370.jpg', 'PR_NOR_NS_A_20250902-102232_00489.jpg'),     # C93
         ('pre', 'pb'): ('PR_DEF_PB_A_20250902-165148_01579.jpg', 'PR_NOR_NB_A_20250902-164918_01089.jpg'),     # C53 · CT1
         ('pre', 'om'): ('PR_DEF_MF_A_20250819-150136_00122.JPG', 'PR_NOR_NM_A_20250818-174506_00121.JPG'),
         ('pre', 'mf'): ('PR_DEF_MF_A_20250902-181342_01598.jpg', 'PR_NOR_NM_A_20250818-101708_00037.JPG'),     # C31
         ('solder', 'nm'): ('SD_DEF_NM_A_20250821-163549_00139.JPG', 'SD_NOR_CC_A_20250821-142503_00072.JPG'),  # C53
         ('solder', 'cc'): ('SD_DEF_CC_A_20250902-090221_00146.JPG', 'SD_NOR_IC_A_20250905-143016_00090.JPG'),
         ('solder', 'ml'): ('SD_DEF_ML_A_20250902-191828_00479.JPG', 'SD_NOR_ST_A_20250904-162225_00209.JPG'),  # C40
         ('solder', 'sh'): ('SD_DEF_SH_A_20250825-121204_00046.JPG', 'SD_NOR_SH_A_20250822-160716_00084.JPG'),  # U9
         ('solder', 'ws'): ('SD_DEF_WS_A_20250827-102058_00021.JPG', 'SD_NOR_MS_A_20250822-181541_00108.JPG'),  # L17
         ('solder', 'ms'): ('SD_DEF_MS_A_20250822-115038_00067.JPG', 'SD_NOR_ST_A_20250822-142514_00065.JPG'),  # C6
         ('solder', 'rs'): ('SD_DEF_RS_A_20250904-093505_00145.JPG', 'SD_NOR_ST_A_20250904-141141_00182.JPG'),  # R12
         ('solder', 'fl'): ('SD_DEF_FL_A_20250822-121827_00073.JPG', 'SD_NOR_ST_A_20250822-152429_00074.JPG'),  # L22 · R41
         ('solder', 'st'): ('SD_DEF_ST_A_20250822-112159_00060.JPG', 'SD_NOR_CC_A_20250821-151214_00083.JPG'),  # L17
         ('solder', 'pb'): ('SD_DEF_PB_A_20250821-113520_00062.JPG', 'SD_NOR_CC_A_20250821-090300_00016.JPG'),  # C15 · R18
         ('solder', 'ph'): ('SD_DEF_PH_A_20250809-164529_00133.JPG', 'SD_NOR_ST_A_20250822-194042_00140.JPG'),  # R28 · R29
         ('solder', 'ic'): ('SD_DEF_IC_A_20250825-092447_00018.JPG', 'SD_NOR_ST_A_20250822-192429_00133.JPG')}  # L5
MAIN = {'pre': 'ns', 'solder': 'ms'}         # 본편 22번: 사전 공정 납쇼트, 납땜 공정 미삽


def load_index():
    with open(ROOT / 'Dataset/aihub_smt/index.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def defect_boxes(label_json):
    j = json.loads((ROOT / label_json).read_text(encoding='utf-8'))
    names = {c['id']: c['name'] for c in j['categories']}
    return [an['bbox'] for an in j['annotations'] if names[an['category_id']].startswith('불량')]


def pair_candidates(rows, cat, code, n=10):
    """불량(그 유형 하나만) 사진마다 지문 거리가 가장 가까운 정상 사진. 거리 순 n개 (불량, 정상, 거리)."""
    use = [r for r in rows if r['category'] == cat and (cat == 'pre' or (r['code'] == 'A' and r['jpeg_q'] == '128'))]
    bits = lambda r: np.array([(int(r['dhash'], 16) >> i) & 1 for i in range(64)], np.int16)  # noqa: E731
    normals = [r for r in use if r['label'] == '0']
    nb = np.stack([bits(r) for r in normals])
    out = []
    for r in use:
        if r['label'] == '1' and r['defect_codes'] == code:
            dist = (bits(r)[None, :] != nb).sum(1)
            j = int(dist.argmin())
            out.append((int(dist[j]), int(r['n_box_defect']), Path(r['image']).name, Path(normals[j]['image']).name))
    return sorted(out)[:n]


def save_pair(by_name, defect, normal, out_defect, out_normal, size=None):
    rd, rn = by_name[defect], by_name[normal]
    assert rd['label'] == '1' and rn['label'] == '0', (defect, normal)
    im = Image.open(ROOT / rd['image']).convert('RGB')
    d = ImageDraw.Draw(im)
    for x, y, w, h in defect_boxes(rd['label_json']):
        d.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline='#FF2A2A', width=5)
    nim = Image.open(ROOT / rn['image']).convert('RGB')
    if size:
        im, nim = im.resize((size, size), Image.LANCZOS), nim.resize((size, size), Image.LANCZOS)
    im.save(out_defect, quality=92)
    nim.save(out_normal, quality=92)


CAM_DIR = ROOT / 'Dataset/aihub_smt/cam'
CAM_MAIN = {'pre': 'nm', 'solder': 'ml'}          # 본편에 쓰는 예시 유형 (시험 불량, cam.py가 유형마다 고른 1장)


def cam_panels():
    """smt/cam.py 결과(K=16, 나누기 0)로 본편용 세 장과 부록용 유형별 그림을 만든다. 지도는 사진마다 자기 최댓값 기준."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT))
    from AnomalyDetection.smt.data import SPLITS, SmtData
    from AnomalyDetection.smt.cam import type_figure
    from AnomalyDetection.smt.seg_ad import RES, defect_mask
    for cat, code in CAM_MAIN.items():
        z = CAM_DIR / f'K16_seed0_{cat}_types.npz'
        if not z.exists():
            print('CAM 결과 없음, 건너뜀:', z.relative_to(ROOT))
            continue
        z = np.load(z)
        with open(SPLITS / 'test.csv', encoding='utf-8') as f:
            defects = [r for r in csv.DictReader(f) if r['category'] == cat and r['label'] == '1']
        names = [defects[i]['defect_type'] for i in z['idx']]
        # 부록: 유형 6종씩 한 그림 (납땜 12종은 두 장). cam.py 유형별 그림과 같은 모양
        with open(ROOT / f'AnomalyDetection/results/smt/cam_K16_seed0_{cat}.csv', encoding='utf-8') as f:
            res = [r for r in csv.DictReader(f) if r['part'] == 'test']
        load = SmtData('none').load
        for c0 in range(0, len(names), 6):
            sel = list(range(c0, min(c0 + 6, len(names))))
            ids = [int(z['idx'][j]) for j in sel]
            cols = []
            for m in ('devnet', 'seg'):
                by = {int(r['idx']): r for r in res if r['model'] == m}
                cols.append((m, z[m][sel].astype(np.float32), np.array([float(by[i]['score']) for i in ids]),
                             np.array([by[i]['flagged'] == '1' for i in ids])))
            type_figure(OUT / f'smt_cam_{cat}_K16_types_{c0 // 6 + 1}.jpg', [names[j] for j in sel],
                        np.stack([load(defects[i]['image'], RES) for i in ids]),
                        np.stack([defect_mask(defects[i], cat, dilate=8) for i in ids]), cols,
                        f'SMT {cat}, K=16, seed 0: test defects, one per type')
        j = names.index(code)
        r = defects[int(z['idx'][j])]
        img, reg = SmtData('none').load(r['image'], RES), defect_mask(r, cat, dilate=8)
        for kind in ('photo', 'devnet', 'seg'):
            fig = plt.figure(figsize=(RES / 100, RES / 100), dpi=100)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis('off')
            ax.imshow(img)
            if kind != 'photo':
                m = z[kind][j].astype(np.float32)
                h = m / max(float(m.max()), 1e-6)
                rgba = plt.cm.jet(h)
                rgba[..., 3] = .7 * h ** .8
                ax.imshow(rgba)
                py, px = np.unravel_index(np.argmax(m), m.shape)
                ax.plot(px, py, 'o', ms=22, mfc='none', mec='magenta', mew=3.5)
            ax.contour(reg, levels=[.5], colors='w', linewidths=2.4)
            ax.set_xlim(-.5, RES - .5)
            ax.set_ylim(RES - .5, -.5)
            fig.savefig(OUT / f'smt_cam_{cat}_{kind}.jpg', dpi=100, pil_kwargs={'quality': 92})
            plt.close(fig)
        print('CAM 예시', cat, TYPES[cat][code], r['image'])


def main():
    rows = load_index()
    if '--candidates' in sys.argv:
        for cat, types in TYPES.items():
            for code, ko in types.items():
                print(cat, code, ko)
                for c in pair_candidates(rows, cat, code):
                    print('   거리 {} 상자 {} | 불량 {} | 정상 {}'.format(*c))
        return
    by_name = {Path(r['image']).name: r for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    for cat, code in MAIN.items():
        save_pair(by_name, *PAIRS[(cat, code)], OUT / f'smt_{cat}_defect.jpg', OUT / f'smt_{cat}_normal.jpg')
        print('본편', cat, TYPES[cat][code], PAIRS[(cat, code)])
    for (cat, code), (defect, normal) in PAIRS.items():
        save_pair(by_name, defect, normal, OUT / f'smt_type_{cat}_{code}_defect.jpg', OUT / f'smt_type_{cat}_{code}_normal.jpg', size=400)
    print('부록 유형별 짝', len(PAIRS), '개 →', OUT.relative_to(ROOT))
    cam_panels()


if __name__ == '__main__':
    main()
