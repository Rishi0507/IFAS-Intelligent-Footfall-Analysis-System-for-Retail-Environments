"""How well do the two annotators agree? (task S2 in docs/research-plan.md)

    python scripts/research/agreement.py --runs outputs/runs/cam1 outputs/runs/cam2 --a rishi --b yash

Cohen's kappa for gender and for age, over people both annotators labelled with
something other than "unsure", and the list of disagreements to resolve
together. Once resolved, save the agreed labels as labels_final.json in each run
folder (same format); every research script prefers that file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research import stats  # noqa: E402
from src.research.data import RESULTS, results_meta, save_results  # noqa: E402


def compare(run_dirs, a, b):
    pairs = {"gender": [], "age": []}
    disagreements = []
    for d in run_dirs:
        d = Path(d)
        docs = []
        for who in (a, b):
            path = d / f"labels_{who}.json"
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            with open(path, encoding="utf-8") as f:
                docs.append(json.load(f)["labels"])
        for track in sorted(set(docs[0]) & set(docs[1]), key=int):
            la, lb = docs[0][track], docs[1][track]
            for field in pairs:
                va, vb = la.get(field), lb.get(field)
                if va in (None, "unsure") or vb in (None, "unsure"):
                    continue
                pairs[field].append((va, vb))
                if va != vb:
                    disagreements.append({"run": d.name, "track": int(track), "field": field, a: va, b: vb})
    rows = []
    for field, ps in pairs.items():
        if not ps:
            continue
        xa, xb = [p[0] for p in ps], [p[1] for p in ps]
        rows.append({"field": field, "n_people": len(ps),
                     "raw_agreement": round(sum(x == y for x, y in ps) / len(ps), 4),
                     "cohens_kappa": round(stats.cohen_kappa(xa, xb), 4)})
    return rows, disagreements


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--a", required=True, help="first annotator name")
    p.add_argument("--b", required=True, help="second annotator name")
    p.add_argument("--out-dir", default=str(RESULTS))
    args = p.parse_args()
    rows, disagreements = compare(args.runs, args.a, args.b)
    save_results(Path(args.out_dir) / "agreement.json", rows,
                 results_meta(annotators=[args.a, args.b], runs=[Path(r).name for r in args.runs],
                              disagreements=disagreements))
    for r in rows:
        print(f"{r['field']:6s} n={r['n_people']}  agreement={r['raw_agreement']:.3f}  "
              f"kappa={r['cohens_kappa']:.3f}")
    print(f"{len(disagreements)} disagreements to resolve (listed in agreement.json)")


if __name__ == "__main__":
    main()
