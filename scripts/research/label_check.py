"""How often is the label a person ends up with right? The product's own
answer, after pooling and after "not sure", against labelled people.

    python scripts/research/label_check.py --run outputs/runs/<run> --annotator <name> [--out ...]

For gender and age it reports, over the labelled people:

* coverage: the share the product gave a label to (the rest it called unsure);
* accuracy and balanced accuracy of those labels, with 95% intervals that
  resample people;
* the confusion between truth and answer, unsure included;
* why people were left unsure (low confidence, or votes that disagreed).

People the annotator could not judge (unsure, two people in the crops, no
person) are left out and counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.data import results_meta  # noqa: E402
from src.research.stats import grouped_bootstrap_ci  # noqa: E402

TASKS = {"gender": ["female", "male"], "age": ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]}


def acc(y, p):
    return sum(a == b for a, b in zip(y, p)) / len(y) if len(y) else float("nan")


def bal(y, p):
    r = []
    for k in set(list(y)):
        sub = [(a, b) for a, b in zip(y, p) if a == k]
        r.append(sum(a == b for a, b in sub) / len(sub))
    return sum(r) / len(r) if r else float("nan")


def ci(fn, y, p, groups):
    if not y:
        return None
    v, lo, hi = grouped_bootstrap_ci(fn, y, p, groups=groups, n_boot=1000)
    return {"value": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(y)}


def check(run, annotator):
    run = Path(run)
    lab = json.loads((run / f"labels_{annotator}.json").read_text(encoding="utf-8"))["labels"]
    doc = json.loads((run / "analytics.json").read_text(encoding="utf-8"))
    tracks = {str(t["id"]): t for t in doc["tracks"]}
    rows = []
    for task, labels in TASKS.items():
        judged = {t: v[task] for t, v in lab.items() if v.get(task) in labels and t in tracks}
        answered = {t: tracks[t][task] for t in judged if tracks[t].get(task)}
        y = [judged[t] for t in answered]
        p = [answered[t] for t in answered]
        g = list(answered)
        confusion = Counter((judged[t], tracks[t].get(task) or "unsure") for t in judged)
        why = Counter(tracks[t].get(f"{task}_unsure") or "no label" for t in judged if not tracks[t].get(task))
        rows.append({
            "run": run.name, "annotator": annotator, "task": task,
            "people_labelled": len(judged),
            "annotator_unsure": sum(1 for v in lab.values() if v.get(task) not in labels),
            "coverage": round(len(answered) / len(judged), 4) if judged else None,
            "accuracy": ci(acc, y, p, g), "balanced": ci(bal, y, p, g),
            "confusion": [{"truth": a, "answer": b, "n": n} for (a, b), n in sorted(confusion.items())],
            "unsure_why": dict(why),
        })
    return rows, doc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--annotator", required=True)
    p.add_argument("--out", default=None, help="results JSON (default: print only)")
    a = p.parse_args()
    rows, doc = check(a.run, a.annotator)
    for r in rows:
        acc_ = r["accuracy"] or {}
        print(f"{r['task']:6s}: labelled {r['people_labelled']}, product answered {100 * (r['coverage'] or 0):.0f}%, "
              f"accuracy {acc_.get('value')} [{acc_.get('lo')}, {acc_.get('hi')}], "
              f"balanced {(r['balanced'] or {}).get('value')}; unsure because {r['unsure_why']}")
    if a.out:
        meta = results_meta(models=[doc["config"].get("gender_model"), doc["config"].get("age_model")],
                            run=Path(a.run).name, annotator=a.annotator, config={
                                k: doc["config"].get(k) for k in ("attributes", "min_gender_conf", "min_age_conf",
                                                                   "vote_spread", "max_votes_per_person", "match_iou",
                                                                   "max_vote_overlap", "detector", "tiles", "stride")})
        Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")
        print(f"written {a.out}")


if __name__ == "__main__":
    main()
