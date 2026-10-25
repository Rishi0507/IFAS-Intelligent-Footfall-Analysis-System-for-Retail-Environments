"""RQ4: how reliably can one fixed camera say which seats and tables are taken?

    python scripts/research/zone_study.py --run outputs/runs/<run> [--out paper/results/rq4_zones.json]

Everything is recomputed from the run's track_boxes.json and zones.json, so no
model is run. Five studies, each a list of rows in the results file:

* smoothing  - the on/off grid (stays shorter than on_sec dropped, gaps
               shorter than off_sec bridged): flicker, stays, utilisation.
               With labels, accuracy too.
* anchor     - where a person "is": feet, lower body, centre, or the overlap
               of the lower half of the box with the zone.
* stick      - spatial hysteresis: how far past a zone's edge a person may
               lean before they count as having left it (0 = off).
* proposal   - automatic seats (no one draws a zone) at several evidence
               thresholds: how many seats, and what they report.
* identity   - tracks counted against people seated: zone occupancy needs no
               identity, footfall does, so ID switches inflate only footfall.
* sampling   - the run re-sampled 2 to 16 times less often, against the full
               rate: how coarse can sampling get before the answer changes?
               (Detection is the pipeline's cost, so this is the cost curve.)
* next_free  - how often "taken longest" and the Kaplan-Meier rule name the
               next zone to free up, against chance.
* labels     - with zone_labels_<annotator>.json in the run (the zone labelling
               page): accuracy, precision, recall, F1 and kappa of the smoothed
               state, 95% intervals over zones, and agreement between annotators.

Labels from a non-human annotator are plumbing checks, not results; the
labelling page and this script do not know the difference, so keep such files
out of any run whose numbers are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.zones import ZoneParams, analyse, load_track_boxes, propose_zones  # noqa: E402
from src.research import zone_eval as E  # noqa: E402
from src.research.data import results_meta  # noqa: E402

ON = [0.0, 1.0, 3.0, 5.0, 10.0]
OFF = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0]
ANCHORS = ["foot", "lower", "center", "overlap"]
FACTORS = [1, 2, 4, 8, 16]
STICK = [0.0, 0.25, 0.5, 1.0]
EVIDENCE = [0.0, 20.0, 40.0, 60.0, 120.0]


def summary_row(block):
    s = block["summary"]
    return {k: s[k] for k in ("mean_utilization", "sessions", "median_session_sec", "raw_flips_per_min",
                              "smoothed_flips_per_min", "unassigned_share", "peak_occupied", "mean_occupied")}


def load_labels(run):
    docs = {}
    for p in sorted(Path(run).glob("zone_labels_*.json")):
        with open(p, encoding="utf-8") as f:
            docs[p.stem[len("zone_labels_"):]] = json.load(f)
    return docs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=str(ROOT / "paper" / "results" / "rq4_zones.json"))
    p.add_argument("--step", type=float, default=1.0, help="seconds between compared moments")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--source-label", default=None,
                   help="what to record as the footage's source; recorder exports often carry a network address")
    a = p.parse_args()

    run = Path(a.run)
    tb = load_track_boxes(run)
    with open(run / "zones.json", encoding="utf-8") as f:
        zones = json.load(f)
    with open(run / "analytics.json", encoding="utf-8") as f:
        doc = json.load(f)
    base_params = ZoneParams.from_dict((doc.get("zones") or {}).get("params"))
    base = analyse(tb, zones, base_params)
    labels = load_labels(run)
    gold = E.majority_labels(labels) if len(labels) > 1 else (next(iter(labels.values())) if labels else None)

    rows = []

    def add(study, block, **setting):
        r = {"study": study, **setting, **summary_row(block)}
        if gold:
            r["labels"] = E.score(block, gold, n_boot=a.boot)
        rows.append(r)

    print("smoothing grid ...")
    for on in ON:
        for off in OFF:
            prm = ZoneParams(**dict(asdict(base_params), on_sec=on, off_sec=off))
            add("smoothing", analyse(tb, zones, prm), on_sec=on, off_sec=off)

    print("anchors ...")
    for anchor in ANCHORS:
        prm = ZoneParams(**dict(asdict(base_params), anchor=anchor))
        add("anchor", analyse(tb, zones, prm), anchor=anchor)

    print("hysteresis ...")
    for stick in STICK:
        prm = ZoneParams(**dict(asdict(base_params), stick=stick))
        add("stick", analyse(tb, zones, prm), stick=stick)

    print("automatic seats ...")
    for ev in EVIDENCE:
        prop = propose_zones(tb, base_params, min_evidence_sec=ev)
        r = {"study": "proposal", "min_evidence_sec": ev, "zones": len(prop["zones"])}
        if prop["zones"]:
            block = analyse(tb, prop, base_params)
            r.update(summary_row(block), occupied_at_end=block["summary"]["occupied_at_end"])
        rows.append(r)

    k = doc.get("kpis", {})
    rows.append({"study": "identity", "tracks": len(doc.get("tracks", [])),
                 "tracks_rejoined": k.get("tracks_rejoined"), "peak_people_in_frame": k.get("peak_occupancy"),
                 "mean_people_in_frame": k.get("avg_occupancy"),
                 "peak_occupied_zones": base["summary"]["peak_occupied"],
                 "mean_occupied_zones": base["summary"]["mean_occupied"],
                 "occupied_at_end": base["summary"]["occupied_at_end"],
                 "tracks_per_person": round(len(doc.get("tracks", [])) / k["peak_occupancy"], 2)
                 if k.get("peak_occupancy") else None})

    print("sampling ...")
    for k in FACTORS:
        block = analyse(E.subsample(tb, k), zones, base_params)
        r = {"study": "sampling", "factor": k, "sample_sec": block["sample_sec"],
             **summary_row(block), **E.agreement(base, block, a.step)}
        if gold:
            r["labels"] = E.score(block, gold, n_boot=a.boot)
        rows.append(r)

    rows.append({"study": "next_free", **base["next_free"]["evaluation"],
                 "km_sessions": (base.get("survival") or {}).get("n"),
                 "km_events": (base.get("survival") or {}).get("events")})

    if labels:
        rows.append({"study": "labels", "annotators": sorted(labels),
                     "agreement": E.inter_annotator(labels) if len(labels) > 1 else [],
                     "base": E.score(base, gold, n_boot=a.boot)})

    meta = results_meta(
        models=[doc["config"].get("detector"), doc["config"].get("tiles") and f"tiles {doc['config']['tiles']}"],
        run=run.name, source=a.source_label or doc["meta"].get("source"), duration_sec=doc["meta"].get("duration_sec"),
        zones=len(base["zones"]), zone_source=zones.get("source"), base_params=asdict(base_params),
        frames_sampled=doc["meta"].get("frames_sampled"), sample_sec=base["sample_sec"],
        people=len(doc.get("tracks", [])), labelled=bool(labels))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)

    b = summary_row(base)
    print(f"\n{len(base['zones'])} zones, {b['sessions']} stays, utilisation {100 * b['mean_utilization']:.1f}%")
    print(f"flicker {b['raw_flips_per_min']:.2f} -> {b['smoothed_flips_per_min']:.2f} changes per zone-minute")
    for r in rows:
        if r["study"] == "sampling":
            print(f"  sample every {r['sample_sec']:.1f}s: state agreement {r.get('state_agreement')}, "
                  f"utilisation error {r.get('utilization_abs_err')}")
    nf = next(r for r in rows if r["study"] == "next_free")
    if nf.get("moments"):
        print(f"next free: longest-seated {nf['longest_seated']}, KM {nf['km_residual']}, chance {nf['random']} "
              f"({nf['moments']} moments)")
    print(f"written {out}")


if __name__ == "__main__":
    main()
