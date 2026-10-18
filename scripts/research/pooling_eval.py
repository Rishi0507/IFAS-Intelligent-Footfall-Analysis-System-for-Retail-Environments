"""RQ3: does pooling a track's frames help, and which frames should count? (task A5)

    python scripts/research/pooling_eval.py --runs outputs/runs/cam1 outputs/runs/cam2

Each labelled person (pipeline --export-crops + the labelling page) has up to 12
classified crops. For every strategy in src/research/pooling.py this reports
per-person accuracy with a 95% bootstrap interval over people, McNemar's test
against the majority vote the product uses, accuracy against the number of
frames pooled (k = 1 ... 12), and the gain on truncated and occluded people.

Per-crop probabilities come from the product's own predictions stored in
crops.jsonl, or from a frames_retail.jsonl written by eval_frames.py
(`--frames`, `--model`), so an adapted model can be pooled the same way.

The quality model is a logistic regression from crop cues to P(crop correct).
It is fitted with grouped cross-validation over people (`--folds`), so no
person's own crops ever weight their own vote, or on an external file of
{"cues", "correct"} rows (`--fit-file`, e.g. from MARS). A model fitted on all
people is saved too, for `process_video(pooling="quality")`.

The learned baseline is temporal attention over frozen per-frame features
(src/research/attention.py), the way video attribute papers pool a track. It
runs when the runs have features_<task>.npz (scripts/research/extract_features.py).
It trains on the same person folds as the quality model, or on a labelled
feature file (`--attention-train`, e.g. MARS train). The results also say how
far its frame weights agree with the quality weights (`attention_vs_quality`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research import pooling as P  # noqa: E402
from src.research import stats  # noqa: E402
from src.research.attention import AttentionPooler, load_features, tracks_from_file  # noqa: E402
from src.research.data import RESULTS, read_jsonl, read_retail, results_meta, save_results  # noqa: E402

SUBSET_SHARE = 0.5           # a person is "truncated" if at least half their crops are
OCCLUDED_IOU = 0.1           # a crop is occluded if it overlaps another person this much


def person_votes(people, task, frames=None, model=None, features=None):
    """[(person, [(probs, cues)], truth)] for people with a truth for `task`.

    With `features` ({item: vector}), each person also gets "feats": their crops'
    vectors in vote order, or None when a crop has no vector.
    """
    lookup = {}
    if frames:
        for r in read_jsonl(frames):
            if r.get("task", task) == task and (model is None or r.get("model") == model):
                lookup[(r["item"])] = r["probs"]
    out = []
    for p in people:
        truth = p["truth"][task]
        if truth is None:
            continue
        votes, feats = [], []
        for c in p["crops"]:
            item = f"{p['run']}/{p['track']}/{c['frame']:06d}"
            probs = lookup.get(item) if frames else (c.get("pred") or {}).get(task)
            if probs:
                votes.append((probs, c.get("cues") or {}))
                feats.append(features.get(item) if features else None)
        if votes:
            ok = features is not None and all(f is not None for f in feats)
            out.append((dict(p, feats=np.vstack(feats) if ok else None), votes, truth))
    return out


def crop_rows(entries):
    """(cues, correct, group) for every crop: the quality model's training data."""
    cues, correct, groups = [], [], []
    for p, votes, truth in entries:
        for probs, c in votes:
            cues.append(c)
            correct.append(max(probs, key=probs.get) == truth)
            groups.append(p["group"])
    return cues, np.array(correct), np.array(groups)


def fold_of(entries, folds, seed):
    """Person -> fold. The quality model and the attention baseline share it."""
    groups = sorted({p["group"] for p, _, _ in entries})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {g: i % folds for i, g in enumerate(groups)}


def fit_cv(entries, folds, l2, seed):
    """One QualityModel per person, fitted without that person's fold."""
    fold_of_ = fold_of(entries, folds, seed)
    models = {}
    for k in range(folds):
        train = [e for e in entries if fold_of_[e[0]["group"]] != k]
        cues, correct, _ = crop_rows(train)
        m = P.QualityModel(l2=l2).fit(cues, correct) if len(set(correct.tolist())) > 1 else None
        for e in entries:
            if fold_of_[e[0]["group"]] == k:
                models[e[0]["group"]] = m
    return models


def fit_attention_cv(entries, folds, seed, epochs=300, **kw):
    """One AttentionPooler per person, trained without that person's fold.
    None when a person has no features."""
    if any(p["feats"] is None for p, _, _ in entries):
        return None
    fold_of_ = fold_of(entries, folds, seed)
    models = {}
    for k in range(folds):
        train = [e for e in entries if fold_of_[e[0]["group"]] != k]
        labels = [t for _, _, t in train]
        m = (AttentionPooler(seed=seed, **kw).fit([p["feats"] for p, _, _ in train], labels,
                                                  epochs=epochs)
             if len(set(labels)) > 1 else None)
        for e in entries:
            if fold_of_[e[0]["group"]] == k:
                models[e[0]["group"]] = m
    return models if any(m is not None for m in models.values()) else None


def _predict(s, votes, feats, model, rng):
    """One person's label under strategy `s`. A fold with no usable model falls
    back to the mean probability."""
    if s == "attention":
        return P.mean_prob(votes)[0] if model is None else model.predict(feats)[0]
    if s == "quality" and model is None:
        return P.mean_prob(votes)[0]
    return P.pool(votes, s, model=model, rng=rng)[0]


def attention_vs_quality(entries, attn, quality):
    """Mean Spearman correlation, over people with 3+ crops, between the frames'
    attention weights and their quality-model weights. Near 1: the learned head
    trusts the frames the measured cues trust. Near 0: it looks at something else."""
    from scipy.stats import spearmanr
    rhos = []
    for p, votes, _ in entries:
        a, q = attn.get(p["group"]), quality.get(p["group"])
        if a is None or q is None or len(votes) < 3:
            continue
        wa = a.predict(p["feats"])[2]
        wq = q.predict([c for _, c in votes])
        if np.ptp(wa) > 1e-9 and np.ptp(wq) > 1e-9:
            rhos.append(float(spearmanr(wa, wq)[0]))
    return {"mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None, "n_people": len(rhos)}


def evaluate(entries, models, strategies, n_draws, seed, n_boot, attn=None):
    """`models`: person -> QualityModel; `attn`: person -> AttentionPooler."""
    rng = np.random.default_rng(seed)
    groups = np.array([p["group"] for p, _, _ in entries])
    model_for = {"quality": models, "attention": attn or {}}
    correct = {}
    for s in strategies:
        res = []
        for p, votes, truth in entries:
            m = model_for.get(s, {}).get(p["group"])
            res.append(_predict(s, votes, p.get("feats"), m, rng) == truth)
        correct[s] = np.array(res)
    rows = []
    for s in strategies:
        row = {"strategy": s, "accuracy": stats.ci(np.mean, correct[s].astype(float),
                                                   n_boot=n_boot, seed=seed)}
        if s != "majority" and "majority" in correct:
            row["mcnemar_vs_majority"] = stats.mcnemar(correct[s], correct["majority"])
        if s in model_for:                 # people whose fold had no usable model
            row["fallback_to_mean"] = int(sum(model_for[s].get(g) is None for g in groups))
        rows.append(row)

    # Accuracy against the number of frames pooled: k crops drawn at random.
    by_k = []
    kmax = max(len(v) for _, v, _ in entries)
    for k in range(1, kmax + 1):
        row = {"k": k}
        for s in strategies:
            if s == "single":
                continue
            hits, n = 0, 0
            for p, votes, truth in entries:
                if len(votes) < k:
                    continue
                m = model_for.get(s, {}).get(p["group"])
                for _ in range(n_draws):
                    pick = rng.choice(len(votes), size=k, replace=False)
                    sub = [votes[i] for i in pick]
                    feats = p["feats"][pick] if p.get("feats") is not None else None
                    hits += _predict(s, sub, feats, m, rng) == truth
                    n += 1
            row[s] = round(hits / n, 4) if n else None
            row["n_people"] = sum(1 for _, v, _ in entries if len(v) >= k)
        by_k.append(row)

    # Where should quality weighting help most? People cut off or hidden.
    subsets = {}
    for name, test in (
            ("truncated", lambda c: c.get("truncated") == 1.0),
            ("occluded", lambda c: (c.get("occlusion") or 0.0) >= OCCLUDED_IOU)):
        mask = np.array([np.mean([bool(test(c)) for _, c in v]) >= SUBSET_SHARE for _, v, _ in entries])
        subsets[name] = {"n_people": int(mask.sum())}
        for s in strategies:
            subsets[name][s] = round(float(correct[s][mask].mean()), 4) if mask.any() else None
        subsets[f"not_{name}"] = {"n_people": int((~mask).sum())}
        for s in strategies:
            subsets[f"not_{name}"][s] = round(float(correct[s][~mask].mean()), 4) if (~mask).any() else None
    return rows, by_k, subsets, groups


LABELS = {"gender": ["female", "male"], "age": ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]}


def run_features(run_dirs, task):
    """{item: vector} from every run's features_<task>.npz, or None if a run lacks one."""
    lookup = {}
    for d in run_dirs:
        f = Path(d) / f"features_{task}.npz"
        if not f.exists():
            return None
        lookup.update(load_features(f)[0])
    return lookup


def fit_attention(entries, task, a):
    """(person -> AttentionPooler, what it was trained on), or (None, reason)."""
    if any(p["feats"] is None for p, _, _ in entries):
        return None, (f"no features_{task}.npz for every crop "
                      "(scripts/research/extract_features.py)")
    kw = {"hidden": a.attention_hidden}
    if a.attention_train:
        tracks, labels = tracks_from_file(a.attention_train)
        if not set(labels) <= set(LABELS[task]):
            return None, f"{a.attention_train} holds {sorted(set(labels))[:4]}, not {task} labels"
        m = AttentionPooler(classes=LABELS[task], seed=a.seed, **kw).fit(
            tracks, labels, epochs=a.attention_epochs)
        return {e[0]["group"]: m for e in entries}, a.attention_train
    models = fit_attention_cv(entries, a.folds, a.seed, epochs=a.attention_epochs, **kw)
    if models is None:
        return None, "no training fold held more than one class"
    return models, f"{a.folds}-fold cross-validation over people"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="labelled run folders")
    p.add_argument("--annotator", default=None, help="one annotator's labels (default: consensus)")
    p.add_argument("--tasks", nargs="+", default=["gender", "age"], choices=["gender", "age"])
    p.add_argument("--frames", default=None, help="frames_retail.jsonl from eval_frames.py")
    p.add_argument("--model", default=None, help="model tag to read from --frames")
    p.add_argument("--fit-file", default=None, help="JSONL of {cues, correct} to fit the quality model on")
    p.add_argument("--attention-train", default=None,
                   help="labelled feature file to train the attention baseline on (e.g. MARS train, gender)")
    p.add_argument("--no-attention", action="store_true", help="skip the attention baseline")
    p.add_argument("--attention-hidden", type=int, default=32)
    p.add_argument("--attention-epochs", type=int, default=300)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--draws", type=int, default=20, help="random draws per person for each k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out-dir", default=str(RESULTS))
    a = p.parse_args()

    people, info = read_retail(a.runs, a.annotator)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in a.tasks:
        features = None if a.no_attention else run_features(a.runs, task)
        entries = person_votes(people, task, a.frames, a.model, features)
        if len(entries) < a.folds:
            print(f"{task}: only {len(entries)} labelled people, skipped")
            continue
        if a.fit_file:
            ext = read_jsonl(a.fit_file)
            m = P.QualityModel(l2=a.l2).fit([r["cues"] for r in ext], [r["correct"] for r in ext])
            models = {e[0]["group"]: m for e in entries}
            fitted_on = a.fit_file
        else:
            models = fit_cv(entries, a.folds, a.l2, a.seed)
            fitted_on = f"{a.folds}-fold cross-validation over people"
        attn, attn_on = (None, "switched off") if a.no_attention else fit_attention(entries, task, a)
        if not attn:
            print(f"{task}: attention baseline skipped: {attn_on}")
        strategies = P.STRATEGIES + (["attention"] if attn else [])
        res, by_k, subsets, _ = evaluate(entries, models, strategies, a.draws, a.seed, a.n_boot, attn)
        cues, correct, _ = crop_rows(entries)
        full = P.QualityModel(l2=a.l2).fit(cues, correct)
        full.meta.update(task=task, runs=[Path(r).name for r in a.runs])
        full.save(out_dir / f"quality_model_{task}.json")
        row = {"task": task, "n_people": len(entries), "n_crops": len(cues),
               "quality_model_fitted_on": fitted_on,
               "quality_model_coef": dict(zip(P.FEATURE_NAMES, full.to_dict()["coef"])),
               "strategies": res, "by_k": by_k, "subsets": subsets}
        row["attention"] = ({"fitted_on": attn_on, "hidden": a.attention_hidden,
                             "epochs": a.attention_epochs,
                             "vs_quality": attention_vs_quality(entries, attn, models)}
                            if attn else {"skipped": attn_on})
        rows.append(row)
        for r in res:
            acc = r["accuracy"]
            mc = r.get("mcnemar_vs_majority")
            print(f"{task:6s} {r['strategy']:10s} {acc['value']:.3f} [{acc['lo']:.3f}, {acc['hi']:.3f}]"
                  + (f"  McNemar p={mc['p']:.3g}" if mc else ""))
    if not rows:
        raise SystemExit("Nothing to evaluate: label more people first.")
    save_results(out_dir / "rq3_pooling.json", rows,
                 results_meta(models=[a.model or "product predictions in crops.jsonl"], seeds=[a.seed],
                              runs=[Path(r).name for r in a.runs], labels=info))


if __name__ == "__main__":
    main()
