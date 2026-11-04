"""VisA pcb1~4 분할표 생성 스크립트 (한 번만 실행하고 결과를 git에 고정).

만드는 파일 (--out 폴더):
  test.csv                        고정 시험셋. 정상 + 불량. 학습·생성 소스로 사용 금지
  train_normal.csv                정상 학습셋 (K=0 비지도, 지도 학습 공통)
  train_K{k}_seed{s}.csv          불량 K장 (카테고리별 K장). 같은 seed 안에서는 K1 ⊂ K3 ⊂ K5 ⊂ ... (중첩)
  splits_meta.json                생성에 쓴 인자·개수 기록

csv 열: category, image, mask, label, split
  - image/mask 는 프로젝트 루트 기준 상대경로 (예: Dataset/visa/raw/pcb1/Data/Images/Anomaly/000.JPG)
  - label: 0 = 정상, 1 = 불량
  - mask: 정상은 빈 문자열

VisA 원본 폴더 구조 (--raw 아래):
  {category}/Data/Images/Normal/*.JPG
  {category}/Data/Images/Anomaly/*.JPG
  {category}/Data/Masks/Anomaly/*.png

사용 예:
  python scripts/make_splits.py --raw Dataset/visa/raw --out Dataset/visa/splits
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FIELDS = ["category", "image", "mask", "label", "split"]


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def find_mask(mask_dir: Path, image: Path):
    """이미지 stem과 같은 이름의 마스크 파일을 찾는다 (확장자 무관)."""
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".bmp"):
        cand = mask_dir / (image.stem + ext)
        if cand.exists():
            return cand
    return None


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    try:  # Windows 콘솔 한글 깨짐 방지
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="Dataset/visa/raw", help="VisA 원본 폴더 (카테고리 폴더들이 바로 아래)")
    ap.add_argument("--out", default="Dataset/visa/splits", help="분할표 저장 폴더")
    ap.add_argument("--root", default=None, help="프로젝트 루트. 기본값은 이 스크립트의 상위 폴더")
    ap.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 30])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--test-anomaly-frac", type=float, default=0.5,
                    help="카테고리별 불량 중 test로 보낼 비율. 나머지는 K장 샘플링 풀")
    ap.add_argument("--test-normal", type=int, default=100, help="카테고리별 test 정상 장수")
    ap.add_argument("--split-seed", type=int, default=42, help="test/풀 분할용 seed (고정)")
    ap.add_argument("--force", action="store_true", help="기존 분할표 덮어쓰기 허용")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    raw = (root / args.raw) if not Path(args.raw).is_absolute() else Path(args.raw)
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    if not raw.is_dir():
        sys.exit(f"[오류] raw 폴더가 없습니다: {raw}")
    if out.exists() and any(out.glob("*.csv")) and not args.force:
        sys.exit(f"[중단] {out} 에 분할표가 이미 있습니다. 분할표는 고정이 원칙입니다. "
                 f"정말 다시 만들려면 --force 를 붙이세요.")
    out.mkdir(parents=True, exist_ok=True)

    max_k = max(args.ks)
    test_rows, train_normal_rows = [], []
    pools = {}          # category -> 불량 풀 (dict rows)
    counts = {}

    for cat in args.categories:
        cdir = raw / cat
        normals = list_images(cdir / "Data" / "Images" / "Normal")
        anomalies = list_images(cdir / "Data" / "Images" / "Anomaly")
        mask_dir = cdir / "Data" / "Masks" / "Anomaly"
        if not normals or not anomalies:
            sys.exit(f"[오류] {cat}: 정상 {len(normals)}장 / 불량 {len(anomalies)}장. 폴더 구조를 확인하세요: {cdir}")

        anom_rows = []
        for img in anomalies:
            m = find_mask(mask_dir, img)
            if m is None:
                sys.exit(f"[오류] 마스크 없음: {img}")
            anom_rows.append({"category": cat, "image": rel(img, root), "mask": rel(m, root), "label": 1})
        norm_rows = [{"category": cat, "image": rel(img, root), "mask": "", "label": 0} for img in normals]

        # test / 풀 분할 (고정 seed, 카테고리별로 독립)
        rng = random.Random(f"{args.split_seed}-{cat}")
        rng.shuffle(anom_rows)
        rng.shuffle(norm_rows)

        n_test_anom = int(round(len(anom_rows) * args.test_anomaly_frac))
        test_anom, pool = anom_rows[:n_test_anom], anom_rows[n_test_anom:]
        if len(pool) < max_k:
            sys.exit(f"[오류] {cat}: 불량 풀 {len(pool)}장 < 최대 K {max_k}. "
                     f"--test-anomaly-frac 을 낮추거나 --ks 를 줄이세요.")
        if len(norm_rows) <= args.test_normal:
            sys.exit(f"[오류] {cat}: 정상 {len(norm_rows)}장 ≤ test 정상 {args.test_normal}장")
        test_norm, train_norm = norm_rows[:args.test_normal], norm_rows[args.test_normal:]

        test_rows += [dict(r, split="test") for r in test_anom + test_norm]
        train_normal_rows += [dict(r, split="train_normal") for r in train_norm]
        pools[cat] = pool
        counts[cat] = {"normal_total": len(norm_rows), "anomaly_total": len(anom_rows),
                       "test_normal": len(test_norm), "test_anomaly": len(test_anom),
                       "train_normal": len(train_norm), "anomaly_pool": len(pool)}

    write_csv(out / "test.csv", test_rows)
    write_csv(out / "train_normal.csv", train_normal_rows)

    # K장 샘플링: seed마다 풀을 한 번 섞고 앞에서 K장 → 같은 seed 안에서 K가 중첩됨
    for s in args.seeds:
        shuffled = {}
        for cat, pool in pools.items():
            rng = random.Random(f"{s}-{cat}")
            p = list(pool)
            rng.shuffle(p)
            shuffled[cat] = p
        for k in args.ks:
            rows = []
            for cat in args.categories:
                rows += [dict(r, split=f"train_K{k}_seed{s}") for r in shuffled[cat][:k]]
            write_csv(out / f"train_K{k}_seed{s}.csv", rows)

    # 누수 자가검사: train_K 들어간 이미지가 test에 없는지
    test_imgs = {r["image"] for r in test_rows}
    for s in args.seeds:
        for cat, pool in pools.items():
            leaked = [r["image"] for r in pool if r["image"] in test_imgs]
            assert not leaked, f"[버그] test와 풀이 겹침: {leaked[:3]}"

    meta = {"args": vars(args), "root": str(root), "counts": counts,
            "files": sorted(p.name for p in out.glob("*.csv"))}
    (out / "splits_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 위치: {out}")
    print(f"{'category':8} {'normal':>7} {'anomaly':>8} | {'test_N':>6} {'test_A':>6} | {'trainN':>6} {'pool':>5}")
    for cat, c in counts.items():
        print(f"{cat:8} {c['normal_total']:7} {c['anomaly_total']:8} | {c['test_normal']:6} {c['test_anomaly']:6} "
              f"| {c['train_normal']:6} {c['anomaly_pool']:5}")
    print(f"test.csv {len(test_rows)}장, train_normal.csv {len(train_normal_rows)}장, "
          f"train_K*.csv {len(args.ks) * len(args.seeds)}개 (카테고리별 K장, seed 내 중첩)")


if __name__ == "__main__":
    main()
