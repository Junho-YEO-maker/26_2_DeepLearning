"""발표 RQ3 데이터 슬라이드의 AIHub SMT 예시 사진 4장 (공정마다 정상 1장 + 불량 1장, 불량은 결함 상자를 빨간 선으로)
과 부록용 유형별 사진 (공정 · 결함 유형마다 정상 1장 + 불량 1장, smt_type_{공정}_{유형}_{normal,defect}.jpg).

AIHub 약관상 SMT 사진은 제3자에게 줄 수 없어서 ppt_figures/local_only/ 에 만든다 (git 제외, .gitignore).
원본이 있는 서버에서만 돌아간다 (Dataset/aihub_smt/raw, 받는 법은 Dataset/aihub_smt/raw/README.md).
build_deck_v4.py는 이 파일들이 있으면 사진을 넣은 발표용 pptx(…_local.pptx)를 따로 만든다.

    python ppt_build/make_smt_figures.py
"""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ppt_figures' / 'local_only'
# (공정, 정상/불량): 시험셋(test.csv)에서 고른 파일 이름. 불량은 결함 하나가 또렷한 장
PICKS = {('pre', 'normal'): 'PR_NOR_MF_A_20250902-122351_00553.jpg',
         ('pre', 'defect'): 'PR_DEF_NS_A_20250902-102053_00715.jpg',        # 납쇼트 (이웃한 납이 이어짐)
         ('solder', 'normal'): 'SD_NOR_RS_A_20250904-094631_00123.JPG',
         ('solder', 'defect'): 'SD_NOR_MS_A_20250903-193716_00265.JPG'}     # 미삽 (부품 빠짐)


# 유형 코드 → (이름, 정상 폴더 이름에 들어 있는 말). 정상 사진은 AIHub가 유형마다 따로 둔 '정상_N.유형' 폴더에서 고른다
TYPES = {'pre': {'nm': ('미납', '미납'), 'nb': ('납부족', '납부족'), 'ns': ('납쇼트', '납쇼트'), 'pb': ('납볼', '납볼'),
                 'om': ('위치 밀림', '납좌표_밀림'), 'mf': ('모양 불량', '납형성_불량')},
         'solder': {'nm': ('미납', '미납'), 'cc': ('냉납', '냉납'), 'ml': ('밀림', '밀림'), 'sh': ('쇼트', '쇼트'),
                    'ws': ('오삽', '오삽'), 'ms': ('미삽', '미삽'), 'rs': ('역삽', '역삽'), 'fl': ('뒤집힘', '뒤집힘'),
                    'st': ('일어섬', '일어섬'), 'pb': ('납볼', '납볼'), 'ph': ('핀홀', '핀홀'), 'ic': ('납고드름', '납고드름')}}
# 자동으로 고른 장이 흐릿하면 여기서 파일 이름으로 바꾼다
TYPE_OVERRIDE = {('solder', 'ml'): {'defect': 'SD_DEF_ML_A_20250821-120637_00069.JPG'},     # 부품이 패드에서 옆으로 밀린 게 잘 보이는 장
                 ('solder', 'ms'): {'defect': 'SD_NOR_MS_A_20250903-193716_00265.JPG'},     # 본편 22번과 같은 장
                 ('solder', 'st'): {'defect': 'SD_DEF_ST_A_20250822-130023_00082.JPG'}}     # 한쪽이 들린 게 잘 보이는 장


def defect_boxes(r):
    j = json.loads((ROOT / r['label_json']).read_text(encoding='utf-8'))
    names = {c['id']: c['name'] for c in j['categories']}
    return [an['bbox'] for an in j['annotations'] if names[an['category_id']].startswith('불량')]


def type_pairs(size=400):
    """부록용: 공정 · 결함 유형마다 정상 1장과 불량 1장 (불량은 결함 상자 빨간 선). 모두 시험셋(test.csv)에서."""
    rows = list(csv.DictReader(open(ROOT / 'Dataset/aihub_smt/splits/test.csv', encoding='utf-8')))
    for cat, types in TYPES.items():
        for code, (ko, folder_word) in types.items():
            over = TYPE_OVERRIDE.get((cat, code), {})
            cands = [r for r in rows if r['category'] == cat and r['label'] == '1' and r['defect_type'] == code]
            scored = []
            for r in cands:
                bs = defect_boxes(r)
                area = sum(w * h for _, _, w, h in bs)
                scored.append((len(bs) != 1, abs(area - 70 * 70), Path(r['image']).name, r, bs))
            scored.sort(key=lambda x: x[:3])
            pick = next((s for s in scored if s[2] == over.get('defect')), scored[0])
            r, bs = pick[3], pick[4]
            im = Image.open(ROOT / r['image']).convert('RGB')
            d = ImageDraw.Draw(im)
            for x, y, w, h in bs:
                d.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline='#FF2A2A', width=5)
            im.resize((size, size), Image.LANCZOS).save(OUT / f'smt_type_{cat}_{code}_defect.jpg', quality=90)
            normals = sorted((r_ for r_ in rows if r_['category'] == cat and r_['label'] == '0'
                              and f'정상_' in r_['image'] and r_['image'].split('/')[-2].endswith(folder_word)),
                             key=lambda r_: Path(r_['image']).name)
            nr = next((r_ for r_ in normals if Path(r_['image']).name == over.get('normal')), normals[len(normals) // 2])
            Image.open(ROOT / nr['image']).convert('RGB').resize((size, size), Image.LANCZOS).save(
                OUT / f'smt_type_{cat}_{code}_normal.jpg', quality=90)
            print(cat, code, ko, '| 불량', Path(r['image']).name, f'상자 {len(bs)}', '| 정상', Path(nr['image']).name, f'(후보 {len(normals)})')


def main():
    rows = {Path(r['image']).name: r for r in csv.DictReader(open(ROOT / 'Dataset/aihub_smt/splits/test.csv', encoding='utf-8'))}
    OUT.mkdir(parents=True, exist_ok=True)
    for (cat, kind), name in PICKS.items():
        r = rows[name]
        assert r['category'] == cat and r['label'] == ('1' if kind == 'defect' else '0'), (name, r['label'])
        im = Image.open(ROOT / r['image']).convert('RGB')
        if kind == 'defect':
            j = json.loads((ROOT / r['label_json']).read_text(encoding='utf-8'))
            names = {c['id']: c['name'] for c in j['categories']}
            d = ImageDraw.Draw(im)
            for an in j['annotations']:
                if names[an['category_id']].startswith('불량'):
                    x, y, w, h = an['bbox']
                    d.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline='#FF2A2A', width=5)
        out = OUT / f'smt_{cat}_{kind}.jpg'
        im.save(out, quality=92)
        print(out.relative_to(ROOT), im.size, r['defect_type'] or '정상')


if __name__ == '__main__':
    main()
    type_pairs()
