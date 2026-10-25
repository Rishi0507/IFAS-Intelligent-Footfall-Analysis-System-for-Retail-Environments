"""Which gender and age classifier to trust on CCTV, and when to say "not sure".

    python scripts/research/attribute_study.py --runs outputs/runs/<run> --annotator <name>

Compares, on the same people:

* the fine-tuned ViT (gender) and the colour/texture SVMs (age), from the
  per-crop predictions each run stored in crops.jsonl;
* CLIP zero-shot with the prompts in src/inference/clip_attrs.py;
* their average (gender);

on two kinds of data: PETA test identities (human labels, small standing
pedestrians; CUHK for the ViT, which never trained on it) and labelled runs
(CCTV crops). A person's label is the mean of their crop probabilities. For
the chosen models it also reports accuracy against coverage when people
under a confidence threshold are left as "not sure".

Needs the CLIP embeddings cached by clip_probe.py (PETA sample, and
crops.clip_vitb32.npz in each run); embeds whatever is missing.

Writes paper/results/attributes.json, and a short summary the pipeline
reads into each run's model card (src/inference/assets/attribute_study.json).
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

from src.inference.clip_attrs import AGE_PROMPTS, CLIP_NAME, GENDER_PROMPTS, ClipEncoder, LinearHead  # noqa: E402
from src.research.data import results_meta  # noqa: E402

G = ["female", "male"]
A = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]
SUMMARY = ROOT / "src" / "inference" / "assets" / "attribute_study.json"


def acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return round(float(np.mean(y == p)), 4) if len(y) else None


def bal(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = [np.mean(p[y == k] == k) for k in np.unique(y)]
    return round(float(np.mean(r)), 4) if r else None


def cached(path):
    if not Path(path).exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return dict(zip(z["paths"].tolist(), z["X"]))


def ensure(emb, items, load, enc, path):
    todo = [k for k in items if k not in emb]
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        for k, x in zip(chunk, enc().embed([load(k) for k in chunk])):
            emb[k] = x
    if todo:
        keys = list(emb)
        np.savez_compressed(path, paths=np.array(keys), X=np.stack([emb[k] for k in keys]))
    return emb


def coverage(people, labels, thresholds):
    """people: [(probs over labels, truth)] -> accuracy and share kept per threshold."""
    out = []
    for th in thresholds:
        kept = [(labels[int(np.argmax(p))], y) for p, y in people if max(p) >= th]
        out.append({"min_conf": th, "kept": round(len(kept) / len(people), 4) if people else None,
                    "accuracy": acc([y for _, y in kept], [a for a, _ in kept]),
                    "balanced": bal([y for _, y in kept], [a for a, _ in kept])})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default=str(ROOT / "data" / "PETA" / "split_identity_all.csv"))
    p.add_argument("--peta-frames", default=str(ROOT / "paper" / "results" / "frames_peta.jsonl"),
                   help="the ViT's stored PETA predictions (eval_frames.py)")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--annotator", default=None)
    p.add_argument("--out", default=str(ROOT / "paper" / "results" / "attributes.json"))
    a = p.parse_args()

    made = []
    enc = lambda: made[0] if made else (made.append(ClipEncoder(CLIP_NAME)) or made[0])  # noqa: E731
    gh = LinearHead.zero_shot(enc(), GENDER_PROMPTS)
    ah = LinearHead.zero_shot(enc(), AGE_PROMPTS)
    rows_out = []

    # ---------------------------------------------------------------- PETA
    split = Path(a.split)
    emb = cached(split.with_name(split.stem + ".clip_vitb32.npz"))
    test = [r for r in csv.DictReader(open(split, encoding="utf-8")) if r["split"] == "test" and r["image"] in emb]
    if test:
        X = np.stack([emb[r["image"]] for r in test])
        pg, pa = gh.proba(X), ah.proba(X)
        vit = {}
        if Path(a.peta_frames).exists():
            for line in open(a.peta_frames, encoding="utf-8"):
                r = json.loads(line)
                if r["task"] == "gender" and r["condition"] == "clean":
                    vit[Path(r["item"]).name] = r["probs"]
        y = [r["gender"] for r in test]
        rows_out.append({"data": "PETA test", "task": "gender", "model": "clip", "n": len(y),
                         "accuracy": acc(y, [G[i] for i in pg.argmax(1)]), "balanced": bal(y, [G[i] for i in pg.argmax(1)])})
        cuhk = [i for i, r in enumerate(test) if r["subset"] == "CUHK" and Path(r["image"]).name in vit]
        if cuhk:
            yc = [test[i]["gender"] for i in cuhk]
            for model, probs in (("vit", [[vit[Path(test[i]["image"]).name][k] for k in G] for i in cuhk]),
                                 ("clip", [pg[i] for i in cuhk]),
                                 ("ensemble", [[(pg[i][j] + vit[Path(test[i]["image"]).name][k]) / 2
                                                for j, k in enumerate(G)] for i in cuhk])):
                pr = [G[int(np.argmax(v))] for v in probs]
                rows_out.append({"data": "PETA CUHK test", "task": "gender", "model": model, "n": len(yc),
                                 "accuracy": acc(yc, pr), "balanced": bal(yc, pr)})
        ya = [r["age"] for r in test]
        ok = [i for i, v in enumerate(ya) if v in A]
        pr = [A[int(pa[i].argmax())] for i in ok]
        rows_out.append({"data": "PETA test", "task": "age", "model": "clip", "n": len(ok),
                         "accuracy": acc([ya[i] for i in ok], pr), "balanced": bal([ya[i] for i in ok], pr)})

    # ---------------------------------------------------------------- runs
    from PIL import Image
    for run in a.runs:
        run = Path(run)
        lab = json.loads((run / f"labels_{a.annotator}.json").read_text(encoding="utf-8"))["labels"]
        crops = [json.loads(l) for l in open(run / "crops.jsonl", encoding="utf-8")]
        crops = [c for c in crops if str(c["track"]) in lab]
        cpath = run / "crops.clip_vitb32.npz"
        cemb = ensure(cached(cpath), [c["crop"] for c in crops], lambda k: Image.open(run / k), enc, cpath)
        X = np.stack([cemb[c["crop"]] for c in crops])
        pg, pa = gh.proba(X), ah.proba(X)
        per = defaultdict(lambda: defaultdict(list))
        for c, g, ag in zip(crops, pg, pa):
            t = str(c["track"])
            v = c["pred"].get("gender")
            per["clip_g"][t].append(g)
            if v:
                per["vit_g"][t].append([v[k] for k in G])
                per["ens_g"][t].append([(g[j] + v[k]) / 2 for j, k in enumerate(G)])
            per["clip_a"][t].append(ag)
            if c["pred"].get("age"):
                per["svm_a"][t].append([c["pred"]["age"][k] for k in A])
        for task, labels, models in (("gender", G, [("vit", "vit_g"), ("clip", "clip_g"), ("ensemble", "ens_g")]),
                                     ("age", A, [("svm", "svm_a"), ("clip", "clip_a")])):
            truth = {t: v[task] for t, v in lab.items() if v.get(task) in labels}
            for model, key in models:
                people = [(np.mean(v, axis=0), truth[t]) for t, v in per[key].items() if t in truth]
                if not people:
                    continue
                pr = [labels[int(np.argmax(p_))] for p_, _ in people]
                yy = [y for _, y in people]
                rows_out.append({"data": run.name, "annotator": a.annotator, "task": task, "model": model,
                                 "n": len(people), "accuracy": acc(yy, pr), "balanced": bal(yy, pr),
                                 "predicted": {k: pr.count(k) for k in labels},
                                 "truth": {k: yy.count(k) for k in labels},
                                 "coverage": coverage(people, labels, [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])})

    meta = results_meta(models=[CLIP_NAME, "abhshkp/footfall-analysis-vit-stage1 (PETA)",
                                "crops.jsonl predictions (runs)"],
                        prompts={"gender": GENDER_PROMPTS, "age": AGE_PROMPTS},
                        annotator=a.annotator, runs=[Path(r).name for r in a.runs])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows_out}, indent=1), encoding="utf-8")

    def pick(data, task, model, field="balanced"):
        return next((r[field] for r in rows_out if r["data"] == data and r["task"] == task and r["model"] == model), None)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "commit": meta["commit"], "date": meta["date"],
        "gender": {"ensemble": {"balanced_accuracy": pick("PETA CUHK test", "gender", "ensemble"), "on": "PETA CUHK test identities"},
                   "clip": {"balanced_accuracy": pick("PETA CUHK test", "gender", "clip"), "on": "PETA CUHK test identities"},
                   "vit": {"balanced_accuracy": pick("PETA CUHK test", "gender", "vit"), "on": "PETA CUHK test identities"}},
        "age": {"clip": {"balanced_accuracy": pick("PETA test", "age", "clip"), "on": "PETA test identities, 4 groups"}},
    }, indent=1), encoding="utf-8")
    for r in rows_out:
        print(f"{r['data']:24s} {r['task']:6s} {r['model']:9s} n={r['n']:5d}  accuracy {r['accuracy']}  balanced {r['balanced']}")
    print(f"written {a.out} and {SUMMARY}")


if __name__ == "__main__":
    main()
