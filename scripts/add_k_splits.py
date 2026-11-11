"""기존 분할(make_splits.py)을 건드리지 않고 K장 분할표를 더 만든다 (예: K=10).

make_splits.py와 같은 규칙: seed마다 (PCB, 종류)별 풀을 Random(f"{s}-{pcb}-{type}")로 한 번 섞고 앞에서 K장.
풀이 K보다 작으면 풀 전체를 쓴다 (PCB2 bent는 풀 8장 → K=10에서도 8장).
같은 방법으로 기존 K(1,2,4,8)를 다시 만들어 저장된 파일과 똑같은지 먼저 확인한다.

  python scripts/add_k_splits.py --ks 10
"""
import argparse
import csv
import random
from collections import OrderedDict
from pathlib import Path

TYPES = ("bent", "melt", "scratch", "missing")


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def build(pools, cats, s, k, name):
    order = {}
    for key, rows in pools.items():
        sh = list(rows)
        random.Random(f"{s}-{key[0]}-{key[1]}").shuffle(sh)
        order[key] = sh
    return [dict(r, split=name) for c in cats for t in TYPES for r in order[(c, t)][:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="Dataset/visa/splits")
    ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3"])
    a = ap.parse_args()
    out = Path(a.splits)
    fields, pool = read(out / "pool.csv")
    pools = OrderedDict()
    for r in pool:
        pools.setdefault((r["category"], r["defect_type"]), []).append(r)

    for s in a.seeds:                       # 재현 검사
        for k in (1, 2, 4, 8):
            name = f"train_K{k}_seed{s}"
            p = out / f"{name}.csv"
            if p.exists():
                assert build(pools, a.categories, s, k, name) == read(p)[1], f"재현 실패: {p}"
    print("기존 K1/2/4/8 분할 재현: 일치")

    for s in a.seeds:
        for k in a.ks:
            name = f"train_K{k}_seed{s}"
            rows = build(pools, a.categories, s, k, name)
            prev = read(out / f"train_K8_seed{s}.csv")[1]
            assert {r["image"] for r in prev} <= {r["image"] for r in rows}, "K8 ⊄ 새 K"
            with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(name, len(rows))


if __name__ == "__main__":
    main()
