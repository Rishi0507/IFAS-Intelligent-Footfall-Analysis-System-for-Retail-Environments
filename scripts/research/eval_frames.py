"""RQ1: how much accuracy is lost from PETA to CCTV, and what drives it (task B4).

    # PETA test split, clean
    python scripts/research/eval_frames.py --domain peta --split-csv data/PETA/split_identity.csv \\
        --gender-repo abhshkp/footfall-analysis-vit-stage1 --age-bundle abhshkp/footfall-analysis-age-svm

    # PETA test split under every controlled degradation (slow on a CPU: use --limit, or Colab)
    python scripts/research/eval_frames.py --domain peta-degraded --split-csv ... --limit 500 ...

    # Labelled retail runs (pipeline --export-crops + the labelling page)
    python scripts/research/eval_frames.py --domain retail --runs outputs/runs/cam1 outputs/runs/cam2 ...

    # MARS, once `scripts/research/get_mars.py index` has built data/mars/tracklets.jsonl
    python scripts/research/eval_frames.py --domain mars --mars-index data/mars/tracklets.jsonl ...

Runs `GenderModel.predict` and `AgeModel.predict` from src/inference/models.py
unchanged, so the numbers are the product's. Writes

    paper/results/frames_<domain>.jsonl   one line per crop and model
    paper/results/rq1_gap.json            accuracy per domain, condition and model,
                                          the radial power spectrum of each domain,
                                          and how well a crop's high-frequency energy
                                          predicts an error

Rows of rq1_gap.json for the same domain and model are replaced on a re-run;
other domains are kept.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research import degrade, spectrum, stats  # noqa: E402
from src.research.data import (AGES, RESULTS, read_mars_index, read_peta_split,  # noqa: E402
                               read_retail, results_meta, save_results, write_jsonl)


# ------------------------------------------------------------------- items
def items_for(a):
    """Every crop to evaluate: {"item", "path", "group", "gender", "age"}."""
    rng = np.random.default_rng(a.seed)
    if a.domain in ("peta", "peta-degraded"):
        rows = read_peta_split(a.split_csv, split=None if a.split == "all" else a.split, root=a.peta_root)
        if a.subsets:
            rows = [r for r in rows if r["item"].split("/")[1] in a.subsets]
        items = [{"item": r["item"], "path": r["image"], "group": r["group"], "track": None,
                  "frame": None, "gender": r["gender"], "age": r["age"]} for r in rows]
    elif a.domain == "retail":
        people, info = read_retail(a.runs, a.annotator)
        print(f"retail: {len(people)} people, unsure {info['unsure']}, "
              f"disagreements left out {info['disagreements']}")
        items = []
        for p in people:
            for c in p["crops"]:
                items.append({"item": f"{p['run']}/{p['track']}/{c['frame']:06d}", "path": c["path"],
                              "group": p["group"], "track": p["track"], "frame": c["frame"],
                              "gender": p["truth"]["gender"], "age": p["truth"]["age"]})
    elif a.domain == "mars":
        items = []
        for r in read_mars_index(a.mars_index):
            if a.split and r.get("split") not in (a.split, None):
                continue
            frames = r["frames"]
            if a.frames_per_tracklet and len(frames) > a.frames_per_tracklet:
                pick = np.linspace(0, len(frames) - 1, a.frames_per_tracklet).round().astype(int)
                frames = [frames[i] for i in pick]
            for f in frames:
                items.append({"item": f"mars/{Path(f).name}", "path": f, "group": r["tracklet"],
                              "track": r["tracklet"], "frame": Path(f).stem,
                              "gender": r.get("gender"), "age": None, "age_coarse": r.get("age_coarse")})
    else:
        raise SystemExit(f"unknown domain {a.domain}")
    if a.limit and len(items) > a.limit:
        # Sample whole persons, so person-level intervals stay meaningful.
        groups = sorted({i["group"] for i in items})
        rng.shuffle(groups)
        keep, n = set(), 0
        for g in groups:
            if n >= a.limit:
                break
            keep.add(g)
            n += sum(1 for i in items if i["group"] == g)
        items = [i for i in items if i["group"] in keep]
    return items


def conditions_for(a):
    """(kind, level) pairs to evaluate: the clean crops, then on degraded PETA
    every cell of src/research/degrade.py's suite, mildest first."""
    if a.domain == "peta-degraded":
        return [("clean", None)] + list(degrade.iter_suite())
    return [("clean", None)]


def condition_id(kind, level):
    """How a condition is named in the results: "clean" or "<kind>@<level>"."""
    return "clean" if kind == "clean" else f"{kind}@{level}"


# ---------------------------------------------------------------- running
def _seed(item):
    return zlib.crc32(item.encode("utf-8"))


def run(a, gender_model, age_model):
    from PIL import Image
    items = items_for(a)
    print(f"{a.domain}: {len(items)} crops")
    rows = []
    spectra = {}
    for name, level in conditions_for(a):
        cond = condition_id(name, level)
        imgs, hf = [], []
        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            if name != "clean":
                img = degrade.apply(img, name, level, seed=_seed(it["item"]))
            imgs.append(img)
            hf.append(spectrum.hf_ratio(img))
        freqs, curve = spectrum.mean_log_spectrum(imgs[: a.spectrum_crops])
        if freqs is not None:
            spectra[cond] = {"freqs": [round(float(f), 5) for f in freqs],
                             "log_power": [round(float(v), 5) for v in curve], "n": min(len(imgs), a.spectrum_crops)}
        outputs = []
        if gender_model is not None:
            outputs.append(("gender", a.gender_name, gender_model.predict(imgs)))
        if age_model is not None:
            outputs.append(("age", a.age_name, age_model.predict(imgs)))
        for task, model_name, preds in outputs:
            for it, probs, h in zip(items, preds, hf):
                label = max(probs, key=probs.get)
                rows.append({"domain": a.domain, "condition": cond, "model": model_name, "task": task,
                             "item": it["item"], "track": it["track"], "frame": it["frame"],
                             "group": it["group"], "probs": {k: round(float(v), 5) for k, v in probs.items()},
                             "label": label, "truth": it["gender"] if task == "gender" else it["age"],
                             "truth_age_coarse": it.get("age_coarse"), "hf_ratio": round(h, 5)})
        print(f"  {cond}: done")
    return rows, spectra


# -------------------------------------------------------------- summarising
def summarise(rows, n_boot=2000):
    """One row per (domain, condition, task, model) with person-level 95% CIs."""
    out = []
    keyed = defaultdict(list)
    for r in rows:
        keyed[(r["domain"], r["condition"], r["task"], r["model"])].append(r)
    for (domain, cond, task, model), rs in sorted(keyed.items()):
        rs_t = [r for r in rs if r["truth"] is not None]
        row = {"domain": domain, "condition": cond, "task": task, "model": model,
               "n_crops": len(rs_t), "n_people": len({r["group"] for r in rs_t})}
        if rs_t:
            t = np.array([r["truth"] for r in rs_t])
            p = np.array([r["label"] for r in rs_t])
            g = np.array([r["group"] for r in rs_t])
            row["accuracy"] = stats.ci(stats.accuracy, t, p, groups=g, n_boot=n_boot)
            row["balanced_accuracy"] = stats.ci(stats.balanced_accuracy, t, p, groups=g, n_boot=n_boot)
            if task == "age":
                per, mean = stats.one_vs_rest_mA(t, p, AGES)
                row["mA"] = {"per_group": per, "mean": mean}
                row.update(stats.ordinal_metrics(t.tolist(), p.tolist()))
            correct = t == p
            hf = np.array([r["hf_ratio"] for r in rs_t])
            row["hf_auc_correct"] = stats.auc(hf, correct)   # > 0.5: sharper crops are right more often
            edges = np.quantile(hf, [0, .2, .4, .6, .8, 1.0])
            bins = np.clip(np.searchsorted(edges, hf, side="right") - 1, 0, 4)
            row["error_by_hf_quintile"] = [
                {"hf_lo": round(float(edges[q]), 5), "hf_hi": round(float(edges[q + 1]), 5),
                 "n": int((bins == q).sum()),
                 "error": round(float(1 - correct[bins == q].mean()), 4) if (bins == q).any() else None}
                for q in range(5)]
        # MARS age: adult vs old only (docs/research-data.md)
        coarse = [r for r in rs if task == "age" and r.get("truth_age_coarse")]
        if coarse:
            t = np.array([r["truth_age_coarse"] for r in coarse])
            p = np.array(["old" if r["label"] == "AgeAbove61" else "adult" for r in coarse])
            row["adult_vs_old"] = stats.ci(stats.balanced_accuracy, t, p,
                                           groups=np.array([r["group"] for r in coarse]),
                                           n_boot=n_boot)
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=["peta", "peta-degraded", "retail", "mars"])
    p.add_argument("--split-csv", help="PETA split from make_peta_split.py")
    p.add_argument("--peta-root", default=None, help="defaults to the split CSV's folder")
    p.add_argument("--split", default="test", help="split to evaluate, or 'all'")
    p.add_argument("--subsets", nargs="*", default=None,
                   help="only these PETA subsets, e.g. CUHK, which the released models never saw")
    p.add_argument("--runs", nargs="*", default=[], help="retail run folders")
    p.add_argument("--annotator", default=None, help="use one annotator's labels (default: consensus)")
    p.add_argument("--mars-index")
    p.add_argument("--frames-per-tracklet", type=int, default=8)
    p.add_argument("--gender-repo")
    p.add_argument("--gender-ckpt")
    p.add_argument("--age-bundle")
    p.add_argument("--gender-name", default=None, help="model tag in the results, e.g. gender-fda-seed1")
    p.add_argument("--age-name", default="age-svm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="evaluate about this many crops, whole people")
    p.add_argument("--spectrum-crops", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out-dir", default=str(RESULTS))
    a = p.parse_args()
    if a.domain.startswith("peta") and not a.split_csv:
        raise SystemExit("--split-csv is required for PETA")
    if a.domain == "retail" and not a.runs:
        raise SystemExit("--runs is required for retail")
    if a.domain == "mars" and not a.mars_index:
        raise SystemExit("--mars-index is required for MARS")

    from src.inference.models import load_age_model, load_gender_model
    gender = load_gender_model(a.gender_repo, a.gender_ckpt, device=a.device)
    age = load_age_model(a.age_bundle)
    if gender is None and age is None:
        raise SystemExit("Pass a gender model, an age model, or both.")
    a.gender_name = a.gender_name or (a.gender_repo or Path(a.gender_ckpt or "gender").name)

    rows, spectra = run(a, gender, age)
    out_dir = Path(a.out_dir)
    write_jsonl(out_dir / f"frames_{a.domain}.jsonl", rows)
    models = [m for m, used in ((a.gender_name, gender), (a.age_name, age)) if used is not None]
    summary = summarise(rows, a.n_boot)
    for r in summary:                                  # say exactly which PETA images these are
        if a.domain.startswith("peta"):
            r.update(peta_split=a.split, peta_subsets=a.subsets or "all")
    spectrum_rows = [{"domain": a.domain, "condition": c, "spectrum": s} for c, s in spectra.items()]
    save_results(out_dir / "rq1_gap.json", summary + spectrum_rows,
                 results_meta(models=models, seeds=[a.seed], domains=[a.domain],
                              peta_split=a.split, peta_subsets=a.subsets, limit=a.limit),
                 replace=lambda r: r.get("domain") == a.domain
                 and (r.get("model") in models or "spectrum" in r))
    for r in summary:
        acc = r.get("accuracy", {})
        print(f"{r['domain']:14s} {r['condition']:22s} {r['task']:6s} n={r['n_people']:5d} people  "
              f"acc={acc.get('value', float('nan')):.3f} [{acc.get('lo', float('nan')):.3f}, "
              f"{acc.get('hi', float('nan')):.3f}]")


if __name__ == "__main__":
    main()
