"""Frozen per-frame features for the RQ3 attention baseline (task A5).

    # every crop of some retail runs (written next to crops.jsonl)
    python scripts/research/extract_features.py --runs outputs/runs/cam1 outputs/runs/cam2 \\
        --gender-repo abhshkp/footfall-analysis-vit-stage1 --age-bundle abhshkp/footfall-analysis-age-svm

    # a labelled training file from MARS, once `get_mars.py index` has built it
    python scripts/research/extract_features.py --mars-index data/mars/tracklets.jsonl \\
        --split train --gender-repo abhshkp/footfall-analysis-vit-stage1 \\
        --out data/mars/features_gender_train.npz

The features are the ones the product's classifiers already compute: the pooled
ViT feature the gender head reads (`GenderModel.embed`) and the colour and
texture histograms the age SVMs score (`AgeModel.embed`). Nothing is trained
here. A retail run gets `features_gender.npz` and `features_age.npz`, keyed by
the same "run/track/frame" ids as eval_frames.py. A MARS file also stores each
frame's tracklet and gender label, which `pooling_eval.py --attention-train`
trains on. MARS age groups do not match PETA's, so MARS trains gender only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.attention import save_features  # noqa: E402
from src.research.data import read_jsonl, read_mars_index  # noqa: E402


def _embed(model, paths, batch=32, say=None):
    from PIL import Image
    out = []
    for i in range(0, len(paths), batch):
        crops = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        out.append(model.embed(crops))
        if say:
            say(min(i + batch, len(paths)), len(paths))
    return np.vstack(out) if out else np.zeros((0, 0), np.float32)


def run_features(run_dir, gender=None, age=None, force=False, say=print):
    """Write features_<task>.npz for one run; returns {task: path}."""
    run_dir = Path(run_dir)
    crops = read_jsonl(run_dir / "crops.jsonl")
    items = [f"{run_dir.name}/{c['track']}/{c['frame']:06d}" for c in crops]
    paths = [run_dir / c["crop"] for c in crops]
    written = {}
    for task, model in (("gender", gender), ("age", age)):
        if model is None:
            continue
        out = run_dir / f"features_{task}.npz"
        if out.exists() and not force:
            say(f"{run_dir.name}: {out.name} exists, kept (--force to redo)")
            written[task] = out
            continue
        vecs = _embed(model, paths)
        save_features(out, items, vecs)
        say(f"{run_dir.name}: {len(items)} crops -> {out.name} ({vecs.shape[1]} values each)")
        written[task] = out
    return written


def mars_features(index, gender, out, split="train", per_tracklet=12, say=print):
    items, paths, groups, labels = [], [], [], []
    for r in read_mars_index(index):
        if split and r.get("split") not in (split, None):
            continue
        if r.get("gender") not in ("female", "male"):
            continue
        frames = r["frames"]
        if per_tracklet and len(frames) > per_tracklet:
            pick = np.linspace(0, len(frames) - 1, per_tracklet).round().astype(int)
            frames = [frames[i] for i in pick]
        for f in frames:
            items.append(f"mars/{Path(f).name}")
            paths.append(f)
            groups.append(str(r["tracklet"]))
            labels.append(r["gender"])
    vecs = _embed(gender, paths, say=lambda i, n: say(f"  {i}/{n}") if i % 1024 < 32 else None)
    save_features(out, items, vecs, groups=groups, labels=labels)
    say(f"MARS {split}: {len(set(groups))} tracklets, {len(items)} frames -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="run folders with crops.jsonl")
    p.add_argument("--mars-index", help="data/mars/tracklets.jsonl from get_mars.py index")
    p.add_argument("--split", default="train")
    p.add_argument("--frames-per-tracklet", type=int, default=12)
    p.add_argument("--out", help="output .npz for --mars-index")
    p.add_argument("--gender-repo")
    p.add_argument("--gender-ckpt")
    p.add_argument("--age-bundle")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true", help="recompute existing feature files")
    a = p.parse_args()
    if not a.runs and not a.mars_index:
        raise SystemExit("Pass --runs, --mars-index, or both.")

    from src.inference.models import load_age_model, load_gender_model
    gender = load_gender_model(a.gender_repo, a.gender_ckpt, device=a.device)
    age = load_age_model(a.age_bundle) if a.runs else None
    if gender is None and age is None:
        raise SystemExit("Pass a gender model, an age model, or both.")
    for r in a.runs:
        run_features(r, gender, age, force=a.force)
    if a.mars_index:
        if gender is None:
            raise SystemExit("MARS features need the gender model.")
        mars_features(a.mars_index, gender, Path(a.out or "data/mars/features_gender_train.npz"),
                      a.split, a.frames_per_tracklet)


if __name__ == "__main__":
    main()
