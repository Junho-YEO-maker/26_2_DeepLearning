"""발표 '데이터' 슬라이드용 예시 사진과 장수 표를 만든다 (build_deck_v2.py가 읽는다).

    python ppt_build/make_dataset_figures.py            (아무 위치에서)

VisA (공개 데이터, git에 올려도 됨): Dataset/visa/raw 또는 Dataset/visa/VisA_official/VisA_PCB 원본에서
  ppt_figures/s02_visa_pcb{1,2,3}_{normal,defect}.jpg   PCB별 정상 1장 · 불량 1장(빨간 상자 = 결함)
  ppt_figures/s02_visa_type_{bent,melt,scratch,missing}.jpg   결함 유형별 확대(빨간 선 = 결함 마스크)
  ppt_figures/app_visa_{pcb}_{type}_{normal,defect}.jpg   부록용: PCB · 결함 유형마다 정상 1장과 불량 1장을 같은 자리로 확대
    (정상 사진을 불량 사진에 기판 정렬(Augmentation/common/kshot.board_transform)한 뒤 같은 칸을 자름)
장수: ppt_figures/dataset_counts.json (숫자만 담는다).
AIHub SMT 사진은 약관상 제3자 제공 금지라 여기서 만들지 않는다. 덱의 SMT 사진 칸은 비워 두고 직접 넣는다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'ppt_figures'
VISA_DIRS = [ROOT / 'Dataset/visa/raw', ROOT / 'Dataset/visa/VisA_official/VisA_PCB']
RED = (230, 40, 40)
# 불량 예시로 쓸 유형 (PCB마다 다른 유형을 보여 준다) · 유형 확대에 쓸 PCB
VISA_DEFECT_TYPE = {'pcb1': 'bent', 'pcb2': 'missing', 'pcb3': 'melt'}
VISA_TYPE_PCB = {'bent': 'pcb3', 'melt': 'pcb3', 'scratch': 'pcb2', 'missing': 'pcb1'}


def visa_root():
    for d in VISA_DIRS:
        if (d / 'pcb1' / 'image_anno.csv').exists():
            return d
    return None


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def pick(cands, root, q=0.6):
    """한 유형짜리 불량 중 마스크 면적이 q 분위에 가장 가까운 것 (너무 작지도 크지도 않게)."""
    areas = []
    for _, r in cands.iterrows():
        m = np.array(Image.open(root / r['mask']).convert('L')) > 0
        areas.append(m.sum())
    areas = np.array(areas)
    return cands.iloc[int(np.argmin(np.abs(areas - np.quantile(areas, q))))]


def save_full(img, out, width=900, boxes=(), line=6):
    img = img.convert('RGB')
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle(b, outline=RED, width=line)
    h = round(img.height * width / img.width)
    img.resize((width, h), Image.LANCZOS).save(out, quality=90)


def save_zoom(img, m, out, size=520, scale=2.4, min_side=260):
    x0, y0, x1, y1 = mask_bbox(m)
    side = int(max(x1 - x0, y1 - y0, min_side / scale) * scale)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    l = int(np.clip(cx - side // 2, 0, max(0, img.width - side)))
    t = int(np.clip(cy - side // 2, 0, max(0, img.height - side)))
    box = (l, t, l + side, t + side)
    crop = img.convert('RGB').crop(box).resize((size, size), Image.LANCZOS)
    mk = Image.fromarray((m * 255).astype(np.uint8)).crop(box).resize((size, size), Image.NEAREST)
    # 결함을 가리지 않게 마스크를 조금 넓힌 바깥 테두리만 그린다
    edge = mk.filter(ImageFilter.MaxFilter(17)).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    crop.paste(Image.new('RGB', crop.size, RED), (0, 0), edge.point(lambda v: 255 if v > 0 else 0))
    crop.save(out, quality=90)


def visa(counts):
    root = visa_root()
    if root is None:
        print('VisA 원본이 없어 VisA 그림은 건너뜀')
        return
    test = pd.read_csv(ROOT / 'Dataset/visa/splits/test.csv')
    pool = pd.read_csv(ROOT / 'Dataset/visa/splits/pool.csv')
    meta = json.loads((ROOT / 'Dataset/visa/splits/splits_meta.json').read_text())['counts']
    counts['visa'] = {}
    for pcb in ('pcb1', 'pcb2', 'pcb3'):
        anno = pd.read_csv(root / pcb / 'image_anno.csv')
        anno['image'] = anno['image'].str.replace(f'^{pcb}/', '', regex=True)
        anno['mask'] = anno['mask'].fillna('').str.replace(f'^{pcb}/', '', regex=True)
        lab = anno.label.value_counts()
        single = {t: int(lab.get(t, 0)) for t in ('bent', 'melt', 'scratch', 'missing')}
        counts['visa'][pcb] = dict(normal=int(lab.get('normal', 0)), defect=int((anno.label != 'normal').sum()),
                                   types=single, multi=int(sum(v for k, v in lab.items() if ',' in k)),
                                   train_normal=meta[pcb]['train_normal'], val_normal=meta[pcb]['val_normal'],
                                   test_normal=meta[pcb]['test_normal'], test_defect=meta[pcb]['test_anomaly'],
                                   pool=meta[pcb]['pool'])
        base = root / pcb
        normal = sorted(anno[anno.label == 'normal'].image)[0]
        with Image.open(base / normal) as im:
            save_full(im, FIG / f's02_visa_{pcb}_normal.jpg')
        t = VISA_DEFECT_TYPE[pcb]
        r = pick(anno[anno.label == t], base)
        m = np.array(Image.open(base / r['mask']).convert('L')) > 0
        x0, y0, x1, y1 = mask_bbox(m)
        pad = 40
        with Image.open(base / r['image']) as im:
            save_full(im, FIG / f's02_visa_{pcb}_defect.jpg', boxes=[(x0 - pad, y0 - pad, x1 + pad, y1 + pad)], line=8)
        print(pcb, '정상', normal, '| 불량', t, r['image'])
    for t, pcb in VISA_TYPE_PCB.items():
        anno = pd.read_csv(root / pcb / 'image_anno.csv')
        anno['image'] = anno['image'].str.replace(f'^{pcb}/', '', regex=True)
        anno['mask'] = anno['mask'].fillna('').str.replace(f'^{pcb}/', '', regex=True)
        base = root / pcb
        r = pick(anno[anno.label == t], base, q=0.5)
        m = np.array(Image.open(base / r['mask']).convert('L')) > 0
        with Image.open(base / r['image']) as im:
            save_zoom(im, m, FIG / f's02_visa_type_{t}.jpg')
        print('유형', t, pcb, r['image'])
    del test, pool


def crop_box(m, w, h, scale=2.4, min_side=260):
    x0, y0, x1, y1 = mask_bbox(m)
    side = int(max(x1 - x0, y1 - y0, min_side / scale) * scale)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    l = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    tp = int(np.clip(cy - side // 2, 0, max(0, h - side)))
    return (l, tp, l + side, tp + side)


def visa_type_pairs(n_normals=12, size=420):
    """부록용: PCB · 결함 유형마다 [정상 | 불량]을 같은 자리로 확대. 정상은 정렬이 가장 잘 맞는 시험 정상 사진."""
    import sys
    sys.path.insert(0, str(ROOT / 'Augmentation'))
    from common.kshot import board_transform
    root = visa_root()
    if root is None:
        print('VisA 원본이 없어 부록 유형 그림은 건너뜀')
        return
    test = pd.read_csv(ROOT / 'Dataset/visa/splits/test.csv')
    for pcb in ('pcb1', 'pcb2', 'pcb3'):
        anno = pd.read_csv(root / pcb / 'image_anno.csv')
        anno['image'] = anno['image'].str.replace(f'^{pcb}/', '', regex=True)
        anno['mask'] = anno['mask'].fillna('').str.replace(f'^{pcb}/', '', regex=True)
        base = root / pcb
        names = sorted(Path(p).name for p in test[(test.category == pcb) & (test.label == 0)].image)[:n_normals]
        normals = [base / 'Data/Images/Normal' / n for n in names]
        for t_ in ('bent', 'melt', 'scratch', 'missing'):
            r = pick(anno[anno.label == t_], base, q=0.5)
            m = np.array(Image.open(base / r['mask']).convert('L')) > 0
            with Image.open(base / r['image']) as im:
                dimg = im.convert('RGB')
            best = None
            for npth in normals:                       # 정렬 점수가 가장 높은 정상 사진
                with Image.open(npth) as im:
                    nimg = im.convert('RGB')
                tf = board_transform(nimg, dimg)
                if best is None or tf['score'] > best[0]['score']:
                    best = (tf, nimg, npth.name)
            tf, nimg, nname = best
            if tf['rot180']:
                nimg = nimg.rotate(180)
            aligned = Image.new('RGB', dimg.size, (128, 128, 128))
            aligned.paste(nimg, (int(tf['dx']), int(tf['dy'])))
            box = crop_box(m, dimg.width, dimg.height)
            aligned.crop(box).resize((size, size), Image.LANCZOS).save(FIG / f'app_visa_{pcb}_{t_}_normal.jpg', quality=90)
            crop = dimg.crop(box).resize((size, size), Image.LANCZOS)
            mk = Image.fromarray((m * 255).astype(np.uint8)).crop(box).resize((size, size), Image.NEAREST)
            edge = mk.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
            crop.paste(Image.new('RGB', crop.size, RED), (0, 0), edge.point(lambda v: 255 if v > 0 else 0))
            crop.save(FIG / f'app_visa_{pcb}_{t_}_defect.jpg', quality=90)
            print('부록', pcb, t_, '불량', r['image'], '| 정상', nname, f"(정렬 점수 {tf['score']:.2f})")


if __name__ == '__main__':
    out = FIG / 'dataset_counts.json'
    counts = json.loads(out.read_text()) if out.exists() else {}
    visa(counts)
    visa_type_pairs()
    out.write_text(json.dumps(counts, ensure_ascii=False, indent=1))
    print('저장:', out)
