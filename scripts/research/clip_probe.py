"""Train the CLIP gender and age heads on PETA and test them where it matters.

    python scripts/research/clip_probe.py --split data/PETA/split_identity_all.csv \
        [--runs outputs/runs/<run> --annotator <name>] [--heads-out src/inference/assets/clip_heads.json]

1. CLIP ViT-B/32 embeds every PETA image once (cached next to the split).
2. A logistic regression per task is fitted on the identity-disjoint training
   split, its strength C chosen on validation (gender by accuracy, age by
   balanced accuracy, with balanced class weights so the head cannot win by
   always answering the commonest age group).
3. Test identities give accuracy and balanced accuracy with 95% intervals,
   overall and per subset.
4. With --runs, every labelled person in those runs is scored: per crop and
   per person (mean of the crop probabilities), for the CLIP heads and for
   the product's own per-crop predictions in crops.jsonl, which come from the
   fine-tuned ViT and the age SVMs.

Writes the heads to --heads-out and the numbers to paper/results/clip_probe.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.clip_attrs import CLIP_NAME, HEADS, ClipEncoder, LinearHead  # noqa: E402
from src.research.data import results_meta  # noqa: E402
from src.research.stats import grouped_bootstrap_ci  # noqa: E402

AGES = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]
GENDERS = ["female", "male"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def sample_rows(rows, caps, seed=0):
    """Up to caps[split] images per split, the same share from every subset and
    gender (so the sample keeps PETA's mix), with a fixed seed. CLIP runs at a
    few images a second on a laptop CPU; a linear head on 512 numbers does
    not need all 19,000."""
    rng = np.random.default_rng(seed)
    keep = []
    for split, cap in caps.items():
        part = [r for r in rows if r["split"] == split]
        if cap is None or cap >= len(part):
            keep += part
            continue
        strata = defaultdict(list)
        for r in part:
            strata[(r["subset"], r["gender"])].append(r)
        for group in strata.values():
            n = max(1, round(cap * len(group) / len(part)))
            idx = rng.choice(len(group), size=min(n, len(group)), replace=False)
            keep += [group[i] for i in sorted(idx)]
    return keep


def embed_split(rows, root, cache, batch=32):
    """Embeddings for `rows`, reusing and growing a cache keyed by image path."""
    from PIL import Image
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["paths"].tolist(), z["X"]))
    todo = [r["image"] for r in rows if r["image"] not in have]
    if todo:
        enc = ClipEncoder(CLIP_NAME)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for path, x in zip(chunk, enc.embed([Image.open(root / p) for p in chunk])):
                have[path] = x
            if (i // batch) % 10 == 0:
                print(f"  embedded {i + len(chunk)}/{len(todo)} new images", flush=True)
                paths = list(have)
                np.savez_compressed(cache, paths=np.array(paths), X=np.stack([have[p] for p in paths]))
        paths = list(have)
        np.savez_compressed(cache, paths=np.array(paths), X=np.stack([have[p] for p in paths]))
    return np.stack([have[r["image"]] for r in rows])


def fit_head(X, y, labels, Xv, yv, balanced, metric):
    from sklearn.linear_model import LogisticRegression
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z, Zv = (X - mean) / scale, (Xv - mean) / scale
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None)
        m.fit(Z, y)
        pv = m.predict(Zv)
        score = metric(yv, pv)
        if best is None or score > best[0]:
            best = (score, C, m)
    score, C, m = best
    order = [list(m.classes_).index(k) for k in labels] if len(labels) > 2 else None
    if len(labels) == 2:
        # sklearn's binary model has one logit for classes_[1]
        w, b = m.coef_, m.intercept_
        if list(m.classes_) != labels:
            w, b = -w, -b
        head = LinearHead(labels, w, b, mean, scale)
    else:
        head = LinearHead(labels, m.coef_[order], m.intercept_[order], mean, scale)
    return head, C, score


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p))) if len(y) else float("nan")


def balanced(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = [np.mean(p[y == k] == k) for k in np.unique(y)]
    return float(np.mean(r)) if r else float("nan")


def with_ci(y, p, groups, fn):
    if not len(y):
        return {"value": None, "lo": None, "hi": None, "n": 0}
    v, lo, hi = grouped_bootstrap_ci(fn, list(y), list(p), groups=list(groups), n_boot=1000)
    return {"value": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(y)}


def score_runs(runs, annotator, heads, enc):
    from PIL import Image
    out = []
    for run in runs:
        run = Path(run)
        lab = json.loads((run / f"labels_{annotator}.json").read_text(encoding="utf-8"))["labels"]
        crops = [json.loads(l) for l in open(run / "crops.jsonl", encoding="utf-8")]
        crops = [c for c in crops if str(c["track"]) in lab]
        cache = run / "crops.clip_vitb32.npz"
        have = {}
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            have = dict(zip(z["paths"].tolist(), z["X"]))
        todo = [c["crop"] for c in crops if c["crop"] not in have]
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            for path, x in zip(chunk, enc().embed([Image.open(run / p) for p in chunk])):
                have[path] = x
        if todo:
            paths = list(have)
            np.savez_compressed(cache, paths=np.array(paths), X=np.stack([have[p] for p in paths]))
        X = np.stack([have[c["crop"]] for c in crops])
        probs = {t: h.proba(X) for t, h in heads.items()}
        for task, labels in (("gender", GENDERS), ("age", AGES)):
            truth = {t: v[task] for t, v in lab.items() if v.get(task) in labels}
            per_crop = defaultdict(lambda: ([], [], []))
            pooled = defaultdict(lambda: defaultdict(list))
            for i, c in enumerate(crops):
                t = str(c["track"])
                if t not in truth:
                    continue
                for model, p in (("clip", dict(zip(labels, probs[task][i]))), ("product", c["pred"].get(task))):
                    if not p:
                        continue
                    y, pr, g = per_crop[model]
                    y.append(truth[t]); pr.append(max(p, key=p.get)); g.append(t)
                    pooled[model][t].append([p[k] for k in labels])
            for model in per_crop:
                y, pr, g = per_crop[model]
                yp = [truth[t] for t in pooled[model]]
                pp = [labels[int(np.argmax(np.mean(v, axis=0)))] for v in pooled[model].values()]
                gp = list(pooled[model])
                out.append({"run": run.name, "annotator": annotator, "task": task, "model": model,
                            "crop_accuracy": with_ci(y, pr, g, accuracy),
                            "crop_balanced": with_ci(y, pr, g, balanced),
                            "person_accuracy": with_ci(yp, pp, gp, accuracy),
                            "person_balanced": with_ci(yp, pp, gp, balanced),
                            "people": len(yp), "crops": len(y),
                            "truth": {k: yp.count(k) for k in labels},
                            "predicted": {k: pp.count(k) for k in labels}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default=str(ROOT / "data" / "PETA" / "split_identity_all.csv"))
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--annotator", default=None)
    p.add_argument("--max-train", type=int, default=3000)
    p.add_argument("--max-val", type=int, default=800)
    p.add_argument("--max-test", type=int, default=1500)
    p.add_argument("--heads-out", default=str(HEADS))
    p.add_argument("--out", default=str(ROOT / "paper" / "results" / "clip_probe.json"))
    a = p.parse_args()

    split = Path(a.split)
    rows = list(csv.DictReader(open(split, encoding="utf-8")))
    rows = sample_rows(rows, {"train": a.max_train, "val": a.max_val, "test": a.max_test})
    print(f"PETA sample: {len(rows)} images", flush=True)
    X = embed_split(rows, split.parent, split.with_name(split.stem + ".clip_vitb32.npz"))
    part = np.array([r["split"] for r in rows])
    pid = np.array([f"{r['subset']}/{r['pid']}" for r in rows])
    subset = np.array([r["subset"] for r in rows])
    results, heads = [], {}
    for task, labels, bal in (("gender", GENDERS, False), ("age", AGES, True)):
        y = np.array([r[task] for r in rows])
        ok = np.isin(y, labels)
        tr, va, te = ok & (part == "train"), ok & (part == "val"), ok & (part == "test")
        head, C, vscore = fit_head(X[tr], y[tr], labels, X[va], y[va], bal, balanced if bal else accuracy)
        heads[task] = head
        pred = np.array(labels)[head.proba(X[te]).argmax(1)]
        row = {"task": task, "C": C, "val_score": round(vscore, 4), "train": int(tr.sum()), "test": int(te.sum()),
               "accuracy": with_ci(y[te], pred, pid[te], accuracy),
               "balanced_accuracy": with_ci(y[te], pred, pid[te], balanced),
               "by_subset": {}}
        for s in sorted(set(subset[te])):
            m = subset[te] == s
            row["by_subset"][s] = {"n": int(m.sum()), "accuracy": round(accuracy(y[te][m], pred[m]), 4),
                                   "balanced": round(balanced(y[te][m], pred[m]), 4)}
        results.append(row)
        print(f"{task}: C={C}  PETA test accuracy {row['accuracy']['value']:.3f} "
              f"[{row['accuracy']['lo']:.3f}, {row['accuracy']['hi']:.3f}], balanced "
              f"{row['balanced_accuracy']['value']:.3f}  (CUHK {row['by_subset'].get('CUHK', {}).get('accuracy')})")

    run_rows = []
    if a.runs:
        if not a.annotator:
            raise SystemExit("--annotator is needed with --runs")
        made = []
        enc = lambda: made[0] if made else (made.append(ClipEncoder(CLIP_NAME)) or made[0])  # noqa: E731
        run_rows = score_runs(a.runs, a.annotator, heads, enc)
        for r in run_rows:
            print(f"{r['run']} {r['task']:6s} {r['model']:7s}: per person {r['person_accuracy']['value']:.3f} "
                  f"(balanced {r['person_balanced']['value']:.3f}, n={r['people']}), per crop "
                  f"{r['crop_accuracy']['value']:.3f}  predicted {r['predicted']}")

    meta = results_meta(models=[CLIP_NAME], split=split.name, trained_on="PETA identity split (train)",
                        annotator=a.annotator, runs=[Path(r).name for r in a.runs])
    Path(a.heads_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.heads_out).write_text(json.dumps({
        "meta": {"clip": CLIP_NAME, "trained_on": "PETA identity-disjoint train split", "commit": meta["commit"],
                 "date": meta["date"], "test": {r["task"]: {"accuracy": r["accuracy"]["value"],
                                                             "balanced_accuracy": r["balanced_accuracy"]["value"]}
                                                for r in results}},
        "heads": {t: h.to_dict(C=next(r["C"] for r in results if r["task"] == t)) for t, h in heads.items()},
    }), encoding="utf-8")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": results + run_rows}, indent=1), encoding="utf-8")
    print(f"written {a.heads_out} and {a.out}")


if __name__ == "__main__":
    main()
