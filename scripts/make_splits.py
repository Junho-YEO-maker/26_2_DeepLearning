"""VisA PCB1–3 공유 분할표 생성 스크립트 (한 번만 실행하고 결과를 git에 고정).

규칙은 AnomalyDetection/README.md 2.2절과 3절을 따른다.
  - 분할 seed 7017 (두 팀 합의). K 샘플링 seed는 0–4.
  - 불량: PCB·결함 종류마다 "그 종류 하나만 있는" 이미지 중 pool = min(10, 장수 − 4)장을 학습 풀로,
          나머지 단일 종류 이미지와 여러 종류가 섞인 이미지는 전부 test.
  - 정상: PCB마다 train_normal 500 / val_normal 200 / 나머지 test.
  - K = PCB 1개에서 결함 종류당 장수. 같은 seed 안에서 K1 ⊂ K2 ⊂ K4 ⊂ K8 (중첩).

만드는 파일 (--out 폴더):
  test.csv                  고정 시험셋(정상 + 불량). 학습·생성·임계값 선택에 사용 금지
  val_normal.csv            임계값·하이퍼파라미터 선택용 정상
  train_normal.csv          정상 학습셋. 생성 시 정상 바탕 이미지도 여기서만 가져온다
  pool.csv                  K장을 뽑는 불량 후보 전체 (참고용, 학습에 통째로 쓰지 않는다)
  train_K{k}_seed{s}.csv    불량 K장 (PCB별 × 종류별 K장)
  splits_meta.json          인자·개수·검사 결과

csv 열: category, image, mask, label, split, defect_type, image_sha256
  - image/mask: 프로젝트 루트 기준 상대경로 (예: Dataset/visa/raw/pcb1/Data/Images/Anomaly/000.JPG)
  - label: 0 = 정상, 1 = 불량
  - defect_type: bent / melt / scratch / missing. 여러 종류면 "melt|scratch". 정상은 빈 문자열
  - 마스크 픽셀값은 종류 ID: 1=bent, 2=melt, 3=scratch, 4=missing (원본 VisA 그대로)

사용 예:
  python scripts/make_splits.py --raw Dataset/visa/raw
  python scripts/make_splits.py --raw Dataset/visa/VisA_official/VisA_PCB   # 원본이 다른 곳에 있을 때
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = ["category", "image", "mask", "label", "split", "defect_type", "image_sha256"]
TYPES = ["bent", "melt", "scratch", "missing"]
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_category(raw: Path, cat: str, prefix: str):
    """image_anno.csv를 읽어 정상/불량 행을 만든다. 마스크 값과 라벨이 맞는지 검사한다."""
    anno = raw / cat / "image_anno.csv"
    if not anno.is_file():
        sys.exit(f"[오류] {anno} 가 없습니다. --raw 가 pcb 폴더들의 부모인지 확인하세요.")
    normals, anomalies = [], []
    with anno.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img = raw / r["image"]
            if not img.is_file():
                sys.exit(f"[오류] 이미지 없음: {img}")
            row = {"category": cat, "image": f"{prefix}/{r['image']}", "mask": "",
                   "label": 0, "defect_type": "", "image_sha256": sha(img)}
            if r["label"] == "normal":
                normals.append(row)
                continue
            types = sorted({t.strip() for t in r["label"].split(",")}, key=TYPE_ID.get)
            assert all(t in TYPE_ID for t in types), f"알 수 없는 종류: {r['label']}"
            mask = raw / r["mask"]
            ids = set(np.unique(np.asarray(Image.open(mask))).tolist()) - {0}
            assert ids == {TYPE_ID[t] for t in types}, f"마스크 값 {ids} ≠ 라벨 {types}: {mask}"
            anomalies.append(dict(row, mask=f"{prefix}/{r['mask']}", label=1, defect_type="|".join(types)))
    return normals, anomalies


def main():
    try:  # Windows 콘솔 한글 깨짐 방지
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="Dataset/visa/raw", help="pcb1, pcb2, pcb3 폴더가 바로 아래 있는 VisA 원본 폴더")
    ap.add_argument("--path-prefix", default="Dataset/visa/raw",
                    help="csv에 적을 경로 앞부분. 원본을 어디서 읽든 csv 경로는 이 값으로 통일한다")
    ap.add_argument("--out", default="Dataset/visa/splits")
    ap.add_argument("--root", default=None, help="프로젝트 루트. 기본값은 이 스크립트의 상위 폴더")
    ap.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--pool-max", type=int, default=10, help="종류별 학습 풀 최대 장수")
    ap.add_argument("--test-min", type=int, default=4, help="종류별로 test에 남길 최소 단일 종류 장수")
    ap.add_argument("--train-normal", type=int, default=500)
    ap.add_argument("--val-normal", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=7017, help="pool/test, 정상 분할용 seed (고정)")
    ap.add_argument("--force", action="store_true", help="기존 분할표 덮어쓰기 (팀 합의 후에만)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    raw = Path(args.raw) if Path(args.raw).is_absolute() else root / args.raw
    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    prefix = args.path_prefix.rstrip("/")
    if out.exists() and any(out.glob("*.csv")) and not args.force:
        sys.exit(f"[중단] {out} 에 분할표가 이미 있습니다. 분할표는 고정이 원칙입니다. "
                 f"정말 다시 만들려면 팀 합의 후 --force 를 붙이세요.")
    out.mkdir(parents=True, exist_ok=True)

    test, val, train_normal, pool_rows = [], [], [], []
    pools = {}   # (cat, type) -> rows
    counts = {}
    for cat in args.categories:
        normals, anomalies = load_category(raw, cat, prefix)

        # 불량: 종류별 단일 종류 이미지에서 풀을 떼고 나머지는 test
        c = {"pool": {}, "test_anomaly": 0}
        for t in TYPES:
            single = [r for r in anomalies if r["defect_type"] == t]
            random.Random(f"{args.split_seed}-{cat}-{t}").shuffle(single)
            n_pool = min(args.pool_max, len(single) - args.test_min)
            if n_pool < max(args.ks):
                sys.exit(f"[오류] {cat} {t}: 풀 {n_pool}장 < 최대 K {max(args.ks)}")
            pools[(cat, t)] = single[:n_pool]
            pool_rows += [dict(r, split="pool") for r in single[:n_pool]]
            test += [dict(r, split="test") for r in single[n_pool:]]
            c["pool"][t] = n_pool
        multi = [r for r in anomalies if "|" in r["defect_type"]]
        test += [dict(r, split="test") for r in multi]
        c["test_anomaly"] = len(anomalies) - sum(c["pool"].values())
        c["test_anomaly_multi"] = len(multi)

        # 정상: 500 / 200 / 나머지
        random.Random(f"{args.split_seed}-{cat}-normal").shuffle(normals)
        a, b = args.train_normal, args.train_normal + args.val_normal
        train_normal += [dict(r, split="train_normal") for r in normals[:a]]
        val += [dict(r, split="val_normal") for r in normals[a:b]]
        test += [dict(r, split="test") for r in normals[b:]]
        c.update(normal_total=len(normals), anomaly_total=len(anomalies),
                 train_normal=a, val_normal=args.val_normal, test_normal=len(normals) - b)
        counts[cat] = c

    write_csv(out / "test.csv", test)
    write_csv(out / "val_normal.csv", val)
    write_csv(out / "train_normal.csv", train_normal)
    write_csv(out / "pool.csv", pool_rows)

    # K장 샘플링: seed마다 (PCB, 종류)별 풀을 한 번 섞고 앞에서 K장 → K끼리 중첩
    k_files = {}
    for s in args.seeds:
        order = {}
        for key, rows in pools.items():
            shuffled = list(rows)
            random.Random(f"{s}-{key[0]}-{key[1]}").shuffle(shuffled)
            order[key] = shuffled
        for k in args.ks:
            name = f"train_K{k}_seed{s}"
            rows = [dict(r, split=name) for cat in args.categories for t in TYPES
                    for r in order[(cat, t)][:k]]
            write_csv(out / f"{name}.csv", rows)
            k_files[name] = [r["image"] for r in rows]

    # 자가검사: 집합끼리 겹치지 않고, 중첩이 성립하는지
    sets = {name: {r["image"] for r in rows} for name, rows in
            [("test", test), ("val_normal", val), ("train_normal", train_normal), ("pool", pool_rows)]}
    names = list(sets)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not sets[x] & sets[y], f"[버그] {x} ∩ {y} 겹침"
    hashes = Counter(r["image_sha256"] for r in test + val + train_normal + pool_rows)
    dup = [h for h, n in hashes.items() if n > 1]
    assert not dup, f"[버그] 같은 파일 내용이 두 번 등장: {dup[:3]}"
    for s in args.seeds:
        for k_small, k_big in zip(args.ks, args.ks[1:]):
            assert set(k_files[f"train_K{k_small}_seed{s}"]) <= set(k_files[f"train_K{k_big}_seed{s}"])
        for k in args.ks:
            assert set(k_files[f"train_K{k}_seed{s}"]) <= sets["pool"]

    meta = {"args": vars(args), "counts": counts,
            "rows": {n: len(v) for n, v in sets.items()},
            "test_anomaly_total": sum(c["test_anomaly"] for c in counts.values()),
            "files": sorted(p.name for p in out.glob("*.csv")),
            "checks": ["disjoint test/val_normal/train_normal/pool", "no duplicate sha256",
                       "K nested within seed", "K subset of pool", "mask ids match labels"]}
    (out / "splits_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 위치: {out}")
    print(f"{'PCB':5} {'pool(B/M/S/Mi)':>16} {'test불량':>7} {'(섞임)':>6} | {'train_N':>7} {'val_N':>5} {'test_N':>6}")
    for cat, c in counts.items():
        p = "/".join(str(c["pool"][t]) for t in TYPES)
        print(f"{cat:5} {p:>16} {c['test_anomaly']:7} {c['test_anomaly_multi']:6} | "
              f"{c['train_normal']:7} {c['val_normal']:5} {c['test_normal']:6}")
    print(f"test 불량 합계 {meta['test_anomaly_total']}장, test.csv {len(test)}행, "
          f"train_K*.csv {len(args.ks) * len(args.seeds)}개")


if __name__ == "__main__":
    main()
