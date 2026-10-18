"""Train and evaluate the gender model under each RQ2 condition (research plan B6, B4).

One call of `train_run` trains GenderClassifier on the PETA training split under
one condition and one seed, picks the best epoch on validation, then scores the
held-out test split clean and under every CCTV degradation of the B3 suite
(the PETA half of the RQ1 gap measurement). Optional evaluation packs add
other domains (MARS, retail) without retraining.

Each run writes two files to `out_dir`, and nothing else is kept unless asked:

    <condition>_seed<k>.json      config, training history, accuracy per test cell
    <condition>_seed<k>.preds.gz  per-image male probability for every cell,
                                  for paired tests between conditions

`aggregate` turns a folder of runs into paper/results/rq2_adaptation.json.
Runs are skipped when their summary already exists, so a Colab session that
dies can simply be restarted.

    python -m src.research.rq2 split     --peta-root PETA --out split.csv
    python -m src.research.rq2 train     --condition degrade --seed 0 --split split.csv \\
                                         --peta-root PETA --out runs/
    python -m src.research.rq2 aggregate --runs runs/ --out paper/results/rq2_adaptation.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

GENDER_TO_IDX = {"female": 0, "male": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# The released stage-1 recipe (train_stage1.ipynb), with a shorter fine-tune:
# the learning-rate study converged within 4 epochs, and 15 runs must fit Colab.
DEFAULTS = dict(epochs_a=4, head_lr_a=1e-3, epochs_b=5, head_lr_b=1e-5, backbone_lr_b=1e-5,
                weight_decay=0.3, warmup_ratio=0.2, batch_size=64, eval_batch_size=128,
                num_workers=2, fda_beta=0.01, degrade_p=0.5, split_column="split_identity")


def run_name(condition, seed):
    return f"{condition.replace('+', '_')}_seed{seed}"


def cells():
    """[("clean", None), (kind, level), ...]: every test condition."""
    from src.research.degrade import iter_suite
    return [("clean", None)] + list(iter_suite())


def cell_name(kind, level):
    return "clean" if kind == "clean" else f"{kind}:{level}"


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LabelledImages:
    """(image path, label) rows as a torch Dataset, optionally degraded.

    A degraded copy uses the image's index as its seed, so every condition is
    tested on exactly the same degraded images.
    """

    def __init__(self, items, transform, degrade=None):
        self.items, self.transform, self.degrade = items, transform, degrade

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.degrade is not None:
            from src.research.degrade import apply
            img = apply(img, *self.degrade, seed=i)
        return self.transform(img), label


def peta_items(split_csv, peta_root, split, column):
    from src.research.peta import read_split_csv
    root = Path(peta_root)
    rows = read_split_csv(split_csv, split, column=column)
    return [(str(root / r["image"]), GENDER_TO_IDX[r["gender"]]) for r in rows], [r["image"] for r in rows]


def pack_items(pack_dir):
    """An evaluation pack: a folder with labels.jsonl lines {"image", "gender", "group"}."""
    pack_dir = Path(pack_dir)
    rows = [json.loads(l) for l in open(pack_dir / "labels.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("gender") in GENDER_TO_IDX]
    return [(str(pack_dir / r["image"]), GENDER_TO_IDX[r["gender"]]) for r in rows], rows


def predict(model, dataset, device, batch_size, num_workers):
    """Male probability per item, in dataset order."""
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=(device == "cuda"))
    out = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs.to(device))
            out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def score(labels, p_male):
    from src.research.stats import balanced_accuracy
    pred = (np.asarray(p_male) >= 0.5).astype(int)
    labels = np.asarray(labels)
    return {"accuracy": float(np.mean(pred == labels)),
            "balanced_accuracy": balanced_accuracy(labels, pred, labels=[0, 1]), "n": int(len(labels))}


def fda_target_paths(fda_dir, limit=500, seed=0):
    paths = sorted(p for p in Path(fda_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.Random(seed).shuffle(paths)
    return [str(p) for p in paths[:limit]]


def train_run(condition, seed, split_csv, peta_root, out_dir, fda_dir=None, eval_packs=None,
              keep_checkpoint=False, device=None, **overrides):
    """Train, select on validation, evaluate; returns the summary dict."""
    import torch
    from torch.utils.data import DataLoader
    from src.training.model import GenderClassifier
    from src.training.trainer import train_phase
    from src.training.transforms import build_transforms

    cfg = {**DEFAULTS, **overrides}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(condition, seed)
    summary_path = out_dir / f"{name}.json"
    if summary_path.exists():
        print(f"[{name}] already done, skipping")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    t0 = time.time()
    targets = fda_target_paths(fda_dir, seed=seed) if fda_dir else None
    train_tf, val_tf = build_transforms(condition=condition, fda_targets=targets,
                                        fda_beta=cfg["fda_beta"], degrade_p=cfg["degrade_p"])
    col = cfg["split_column"]
    train_items, _ = peta_items(split_csv, peta_root, "train", col)
    val_items, _ = peta_items(split_csv, peta_root, "val", col)
    test_items, test_ids = peta_items(split_csv, peta_root, "test", col)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(LabelledImages(train_items, train_tf), batch_size=cfg["batch_size"],
                              shuffle=True, generator=g, num_workers=cfg["num_workers"],
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(LabelledImages(val_items, val_tf), batch_size=cfg["eval_batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device == "cuda"))

    ckpt = out_dir / f"_ckpt_{name}"
    model = GenderClassifier().to(device)
    common = dict(weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
                  fp16=(device == "cuda"), output_dir=str(ckpt))
    best, hist_a = train_phase(model, train_loader, val_loader, device, "probe", cfg["epochs_a"],
                               cfg["head_lr_a"], 0.0, freeze=True, **common)
    best, hist_b = train_phase(model, train_loader, val_loader, device, "finetune", cfg["epochs_b"],
                               cfg["head_lr_b"], cfg["backbone_lr_b"], freeze=False, best_acc=best, **common)
    train_sec = time.time() - t0

    model = GenderClassifier.load(ckpt / "best", device=device)
    labels = [lab for _, lab in test_items]
    preds, results = {}, {}
    for kind, level in cells():
        ds = LabelledImages(test_items, val_tf, None if kind == "clean" else (kind, level))
        p = predict(model, ds, device, cfg["eval_batch_size"], cfg["num_workers"])
        preds[cell_name(kind, level)] = np.round(p, 4).tolist()
        results[cell_name(kind, level)] = score(labels, p)
        print(f"[{name}] test {cell_name(kind, level):16} bal.acc {results[cell_name(kind, level)]['balanced_accuracy']:.4f}")
    packs = {}
    for pack_name, pack_dir in (eval_packs or {}).items():
        items, rows = pack_items(pack_dir)
        p = predict(model, LabelledImages(items, val_tf), device, cfg["eval_batch_size"], cfg["num_workers"])
        packs[pack_name] = {"result": score([lab for _, lab in items], p),
                            "p_male": np.round(p, 4).tolist(),
                            "groups": [r.get("group") for r in rows], "labels": [lab for _, lab in items]}
        print(f"[{name}] pack {pack_name:14} bal.acc {packs[pack_name]['result']['balanced_accuracy']:.4f}")

    summary = {"condition": condition, "seed": seed, "config": cfg, "device": device,
               "fda_targets": len(targets) if targets else 0, "val_best_accuracy": best,
               "history": hist_a + hist_b, "test": results,
               "packs": {k: v["result"] for k, v in packs.items()},
               "train_sec": round(train_sec), "total_sec": round(time.time() - t0)}
    with gzip.open(out_dir / f"{name}.preds.gz", "wt", encoding="utf-8") as f:
        json.dump({"test_images": test_ids, "test_labels": labels, "test": preds,
                   "packs": {k: {kk: v[kk] for kk in ("p_male", "groups", "labels")} for k, v in packs.items()}}, f)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")   # written last: marks the run done
    if not keep_checkpoint:
        shutil.rmtree(ckpt, ignore_errors=True)
    return summary


def aggregate(runs_dir, baseline="baseline"):
    """Mean and sd over seeds per condition and test cell, plus each condition's
    difference from the baseline with the same seed (McNemar per seed on the
    clean test set)."""
    from src.research.stats import mcnemar, mean_sd
    runs_dir = Path(runs_dir)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]
    by_cond = defaultdict(dict)
    for r in runs:
        by_cond[r["condition"]][r["seed"]] = r

    def preds(cond, seed):
        with gzip.open(runs_dir / f"{run_name(cond, seed)}.preds.gz", "rt", encoding="utf-8") as f:
            return json.load(f)

    rows = []
    for cond, seeds in sorted(by_cond.items()):
        cell_names = list(next(iter(seeds.values()))["test"].keys())
        row = {"condition": cond, "seeds": sorted(seeds), "test": {}, "packs": {}}
        for c in cell_names:
            m, sd = mean_sd([s["test"][c]["balanced_accuracy"] for s in seeds.values()])
            row["test"][c] = {"balanced_accuracy_mean": round(m, 4), "balanced_accuracy_sd": round(sd, 4)}
        for pk in sorted({k for s in seeds.values() for k in s.get("packs", {})}):
            vals = [s["packs"][pk]["balanced_accuracy"] for s in seeds.values() if pk in s.get("packs", {})]
            m, sd = mean_sd(vals)
            row["packs"][pk] = {"balanced_accuracy_mean": round(m, 4), "balanced_accuracy_sd": round(sd, 4)}
        if cond != baseline and baseline in by_cond:
            paired = []
            for seed in sorted(set(seeds) & set(by_cond[baseline])):
                a, b = preds(cond, seed), preds(baseline, seed)
                lab = np.asarray(a["test_labels"])
                ca = (np.asarray(a["test"]["clean"]) >= 0.5).astype(int) == lab
                cb = (np.asarray(b["test"]["clean"]) >= 0.5).astype(int) == lab
                diff = seeds[seed]["test"]["clean"]["balanced_accuracy"] - by_cond[baseline][seed]["test"]["clean"]["balanced_accuracy"]
                paired.append({"seed": seed, "clean_diff": round(diff, 4), "mcnemar": mcnemar(ca, cb)})
            row["vs_baseline"] = paired
        rows.append(row)
    return {"meta": {"date": time.strftime("%Y-%m-%d"), "runs": len(runs), "baseline": baseline,
                     "metric": "balanced accuracy on the PETA person-disjoint test split"},
            "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--peta-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train")
    t.add_argument("--condition", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--split", required=True)
    t.add_argument("--peta-root", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--fda-dir")
    t.add_argument("--pack", action="append", default=[], metavar="NAME=DIR")
    t.add_argument("--epochs-b", type=int, default=DEFAULTS["epochs_b"])
    t.add_argument("--split-column", default=DEFAULTS["split_column"])
    t.add_argument("--keep-checkpoint", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("--runs", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args()

    if args.cmd == "split":
        from src.research.peta import parse_peta, write_split_csv
        recs = parse_peta(args.peta_root)
        write_split_csv(recs, args.out, seed=args.seed)
        print(f"{len(recs)} images -> {args.out}")
    elif args.cmd == "train":
        packs = dict(x.split("=", 1) for x in args.pack)
        train_run(args.condition, args.seed, args.split, args.peta_root, args.out, fda_dir=args.fda_dir,
                  eval_packs=packs, keep_checkpoint=args.keep_checkpoint, epochs_b=args.epochs_b,
                  split_column=args.split_column)
    else:
        out = aggregate(args.runs)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{out['meta']['runs']} runs -> {args.out}")


if __name__ == "__main__":
    main()
