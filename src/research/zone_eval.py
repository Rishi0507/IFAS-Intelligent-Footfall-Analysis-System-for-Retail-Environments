"""Evaluating zone occupancy (RQ4): against labels where they exist, and with
checks that need none.

Label-free checks ask whether the measurement is *stable*: how much flicker
smoothing removes, how much the answer moves with the anchor rule or the
smoothing settings, and how coarsely the video can be sampled before the
answer changes. Labelled checks score the smoothed state of each zone at each
labelled moment against what an annotator saw on the clean frame, with
confidence intervals that resample whole zones, because one zone's moments are
not independent of each other.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import stats


def taken_at(block, times, seated_only=True):
    """{zone_id: bool array over `times`}: whether each zone was taken."""
    out = {}
    t = np.asarray(times, dtype=float)
    for z in block["zones"]:
        if seated_only and z["kind"] not in ("seat", "table"):
            continue
        m = np.zeros(len(t), dtype=bool)
        for s in z["sessions"]:
            m |= (t >= s["start"]) & (t < s["end"])
        out[z["id"]] = m
    return out


def grid(block, step=1.0):
    return np.arange(0.0, block["duration_sec"], step)


def subsample(tb, factor):
    """The same run as if it had been sampled `factor` times less often."""
    if factor <= 1:
        return tb
    stride = int(tb["stride"]) * int(factor)
    return dict(tb, stride=stride, tracks={
        k: [o for o in obs if o[0] % stride == 0] for k, obs in tb["tracks"].items()})


def agreement(a, b, step=1.0):
    """How closely analysis `b` matches reference `a`, zone by zone."""
    times = grid(a, step)
    sa, sb = taken_at(a, times), taken_at(b, times)
    common = [z for z in sa if z in sb]
    if not common:
        return {"zones": 0}
    acc = float(np.mean(np.concatenate([sa[z] == sb[z] for z in common])))
    ua = {z["id"]: z["metrics"]["utilization"] for z in a["zones"]}
    ub = {z["id"]: z["metrics"]["utilization"] for z in b["zones"]}
    na = {z["id"]: z["metrics"]["sessions"] for z in a["zones"]}
    nb = {z["id"]: z["metrics"]["sessions"] for z in b["zones"]}
    return {
        "zones": len(common),
        "state_agreement": round(acc, 4),
        "utilization_abs_err": round(float(np.mean([abs(ua[z] - ub[z]) for z in common])), 4),
        "sessions_ref": int(sum(na[z] for z in common)),
        "sessions": int(sum(nb[z] for z in common)),
    }


def label_items(block, labels_doc):
    """(truth, pred, zone, moment) for every labelled seat or table, unsure left out.

    Returns the four arrays and how many items were unsure.
    """
    seated = {z["id"] for z in block["zones"] if z["kind"] in ("seat", "table")}
    truth, pred, zone, moment = [], [], [], []
    unsure = 0
    frames = labels_doc.get("frames", {})
    times = sorted(float(t) for t in frames)
    states = taken_at(block, times)
    for i, t in enumerate(times):
        row = frames[f"{t:.2f}"] if f"{t:.2f}" in frames else frames.get(str(t), {})
        for zid, v in row.items():
            if zid not in seated or zid not in states:
                continue
            if v is None:
                unsure += 1
                continue
            truth.append(bool(v))
            pred.append(bool(states[zid][i]))
            zone.append(zid)
            moment.append(t)
    return (np.array(truth, bool), np.array(pred, bool), np.array(zone), np.array(moment), unsure)


def _prf(truth, pred):
    tp = float(np.sum(truth & pred))
    fp = float(np.sum(~truth & pred))
    fn = float(np.sum(truth & ~pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(block, labels_doc, n_boot=2000):
    """Accuracy, precision, recall, F1 and kappa of the smoothed zone state,
    with 95% intervals resampling whole zones."""
    truth, pred, zone, moment, unsure = label_items(block, labels_doc)
    out = {"items": int(len(truth)), "unsure": int(unsure), "zones": int(len(set(zone.tolist()))),
           "moments": int(len(set(moment.tolist())))}
    if not len(truth):
        return out

    def ci(fn):
        try:
            v, lo, hi = stats.grouped_bootstrap_ci(fn, truth, pred, groups=zone, n_boot=n_boot)
        except ValueError:
            v, lo, hi = float(fn(truth, pred)), float("nan"), float("nan")
        return {"value": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    out["accuracy"] = ci(lambda a, b: float(np.mean(a == b)))
    out["precision"] = ci(lambda a, b: _prf(a, b)[0])
    out["recall"] = ci(lambda a, b: _prf(a, b)[1])
    out["f1"] = ci(lambda a, b: _prf(a, b)[2])
    out["kappa"] = round(float(stats.cohen_kappa(truth.astype(int), pred.astype(int))), 4) \
        if len(set(truth.tolist()) | set(pred.tolist())) > 1 else None
    out["base_rate_taken"] = round(float(truth.mean()), 4)
    return out


def inter_annotator(docs):
    """Cohen's kappa for every pair of annotators, on the zone-moments both
    labelled (unsure by either left out)."""
    out = []
    names = sorted(docs)
    for a, b in itertools.combinations(names, 2):
        fa, fb = docs[a].get("frames", {}), docs[b].get("frames", {})
        xa, xb = [], []
        for t in set(fa) & set(fb):
            for z in set(fa[t]) & set(fb[t]):
                va, vb = fa[t][z], fb[t][z]
                if va is None or vb is None:
                    continue
                xa.append(int(va))
                xb.append(int(vb))
        k = stats.cohen_kappa(xa, xb) if xa else None
        out.append({"a": a, "b": b, "items": len(xa),
                    "agreement": round(float(np.mean(np.array(xa) == np.array(xb))), 4) if xa else None,
                    "kappa": round(float(k), 4) if k is not None and not math.isnan(k) else None})
    return out


def majority_labels(docs):
    """One labels document from several annotators: the majority per zone-moment,
    unsure where they tie or all were unsure."""
    frames = {}
    keys = set().union(*(d.get("frames", {}).keys() for d in docs.values())) if docs else set()
    for t in keys:
        rows = [d["frames"][t] for d in docs.values() if t in d.get("frames", {})]
        zones = set().union(*(r.keys() for r in rows))
        out = {}
        for z in zones:
            votes = [r[z] for r in rows if r.get(z) is not None]
            yes = sum(1 for v in votes if v)
            no = len(votes) - yes
            out[z] = None if yes == no else yes > no
        frames[t] = out
    return {"annotator": "majority", "frames": frames}
