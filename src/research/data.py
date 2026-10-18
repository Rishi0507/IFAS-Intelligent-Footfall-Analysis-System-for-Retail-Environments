"""Reading the research data and writing results (docs/research-plan.md, Section 3).

Three sources of labelled crops:

* PETA, through a split CSV from scripts/research/make_peta_split.py.
* Retail runs: a run folder holding crops.jsonl (pipeline --export-crops) and one
  labels_<annotator>.json per annotator (the labelling page).
* MARS, through the tracklet index `scripts/research/get_mars.py index` writes
  (src/research/mars.py).

Results go to paper/results/<name>.json as {"meta": {...}, "rows": [...]}, the
format paper/figures/make_figures.py reads under its `experiment` source tag.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results"
GENDERS = ("female", "male")
AGES = ("Age16-30", "Age31-45", "Age46-60", "AgeAbove61")


# ---------------------------------------------------------------------- files
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, append=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def results_meta(models=(), seeds=(), **extra):
    return {"commit": git_commit(), "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "models": list(models), "seeds": list(seeds), **extra}


def save_results(path, rows, meta, replace=None):
    """Write {"meta", "rows"}. With `replace` (a row -> bool test), rows already in
    the file that pass the test are swapped for the new ones and the rest kept,
    so one domain or condition can be re-run without losing the others."""
    path = Path(path)
    old = []
    if replace is not None and path.exists():
        with open(path, encoding="utf-8") as f:
            old = [r for r in json.load(f).get("rows", []) if not replace(r)]
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta, "rows": old + list(rows)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return doc


# ----------------------------------------------------------------------- PETA
def read_peta_split(csv_path, split=None, root=None):
    """Rows of a split CSV: image (absolute), gender, age (or None), group, split."""
    csv_path = Path(csv_path)
    root = Path(root) if root else csv_path.parent
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            img = Path(r["image"])
            out.append({
                "image": str(img if img.is_absolute() else root / img),
                "gender": r.get("gender") or None,
                "age": r.get("age") or None,
                "group": f"{r.get('subset', '')}/{r.get('pid', '')}",
                "split": r["split"],
                "item": f"peta/{r.get('subset', '')}/{img.name}",
            })
    return out


# ----------------------------------------------------------------------- MARS
def read_mars_index(path, root=None):
    """Tracklets from src/research/mars.py's index, in the form the RQ1 and RQ3
    scripts take: {"tracklet", "frames": [absolute paths], "gender",
    "age_coarse", "split"}.

    `root` is the MARS folder the index's frame paths are relative to; by
    default the index file's own folder (data/mars/). MARS age groups do not
    line up with PETA's, so age is kept only as adult or old
    (docs/research-data.md); young and teenager become None.
    """
    path = Path(path)
    root = Path(root) if root else path.parent
    out = []
    for r in read_jsonl(path):
        key = (f"{int(r['pid']):04d}C{int(r['cam'])}T{int(r['tracklet']):04d}" if "pid" in r
               else str(r["tracklet"]))
        out.append({"tracklet": key, "split": r.get("split"), "gender": r.get("gender"),
                    "age_coarse": r.get("age") if r.get("age") in ("adult", "old") else r.get("age_coarse"),
                    "frames": [str(f if Path(f).is_absolute() else root / f) for f in r["frames"]]})
    return out


# --------------------------------------------------------------------- retail
def read_labels(run_dir, annotator=None):
    """Truth per person for one run, and how it was decided.

    `annotator` picks one file. Without it: labels_final.json if the annotators
    have written their resolved labels there, else the only annotator, else the
    people every annotator agrees on (disagreements are left out and counted).
    Returns ({track: {"gender", "age"}}, info).
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("labels_*.json"))
    if annotator:
        files = [run_dir / f"labels_{annotator}.json"]
    elif (run_dir / "labels_final.json").exists():
        files = [run_dir / "labels_final.json"]
    docs = []
    for p in files:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                docs.append(json.load(f))
    if not docs:
        return {}, {"annotators": [], "disagreements": 0}
    truth, disagree = {}, 0
    tracks = set().union(*(d["labels"].keys() for d in docs))
    for t in tracks:
        per = [d["labels"].get(t) for d in docs]
        if any(p is None for p in per):
            continue
        entry = {}
        for field in ("gender", "age"):
            vals = {p.get(field) for p in per}
            if len(vals) == 1:
                entry[field] = next(iter(vals))
            else:                                     # annotators gave different answers
                entry[field] = None
                disagree += 1
        truth[int(t)] = entry
    return truth, {"annotators": [d.get("annotator") for d in docs], "disagreements": disagree}


def read_retail(run_dirs, annotator=None):
    """People of labelled retail runs, each with their crops and their truth.

    Returns (people, info). A person is {"run", "track", "group", "truth":
    {"gender", "age"}, "crops": [crops.jsonl records with an absolute "path"]}.
    'unsure' and missing labels become None, and `info` counts them per field.
    """
    people, info = [], {"runs": [], "unsure": {"gender": 0, "age": 0}, "disagreements": 0}
    for d in run_dirs:
        d = Path(d)
        crops_path = d / "crops.jsonl"
        if not crops_path.exists():
            raise FileNotFoundError(f"{d} has no crops.jsonl; process it with --export-crops")
        truth, linfo = read_labels(d, annotator)
        info["runs"].append({"run": d.name, **linfo})
        info["disagreements"] += linfo["disagreements"]
        by_track = {}
        for r in read_jsonl(crops_path):
            r = dict(r, path=str(d / r["crop"]))
            by_track.setdefault(r["track"], []).append(r)
        for track, crops in sorted(by_track.items()):
            t = truth.get(track, {})
            clean = {}
            for field, valid in (("gender", GENDERS), ("age", AGES)):
                v = t.get(field)
                if v == "unsure":
                    info["unsure"][field] += 1
                clean[field] = v if v in valid else None
            people.append({"run": d.name, "track": track, "group": f"{d.name}/{track}",
                           "truth": clean, "crops": crops})
    return people, info
