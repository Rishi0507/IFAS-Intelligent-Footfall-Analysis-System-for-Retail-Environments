"""Generate a synthetic run so the dashboard can be opened without model weights.

This does NOT run the models. It simulates plausible tracks and feeds them
through the real `build_analytics`, so the JSON is byte-for-byte the same shape
a real run produces. The output is flagged `"demo": true` and the dashboard
shows a "sample data" badge for it.

    python scripts/generate_demo_run.py --out outputs/runs/demo
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.analytics import AGE_LABELS, build_analytics  # noqa: E402
from src.inference.tracker import Track  # noqa: E402
from src.inference.trends import insights  # noqa: E402

FPS = 25.0
W, H = 1280, 720
DURATION = 420.0  # 7 minutes

# Store-entrance priors: a slight female skew, most shoppers 16-45.
GENDER_P = {"female": 0.54, "male": 0.46}
AGE_P = {"Age16-30": 0.44, "Age31-45": 0.33, "Age46-60": 0.17, "AgeAbove61": 0.06}
# Older visitors linger; the youngest bracket passes through fastest.
DWELL_MEAN = {"Age16-30": 11.0, "Age31-45": 18.0, "Age46-60": 26.0, "AgeAbove61": 31.0}


def _pick(weights, rng):
    r, acc = rng.random(), 0.0
    for k, p in weights.items():
        acc += p
        if r <= acc:
            return k
    return list(weights)[-1]


def _probs(true_label, labels, conf, rng):
    """A classifier output that is usually right, with confidence `conf`."""
    rest = (1.0 - conf) / (len(labels) - 1)
    out = {l: rest * rng.uniform(0.6, 1.4) for l in labels if l != true_label}
    out[true_label] = conf
    total = sum(out.values())
    return {l: v / total for l, v in out.items()}


def make_tracks(n_people, rng):
    tracks = []
    for tid in range(1, n_people + 1):
        # Arrivals cluster into two rushes (a lunch peak and a late peak).
        peak = rng.choice([0.28, 0.72])
        start_sec = min(max(rng.gauss(peak * DURATION, DURATION * 0.12), 0.0), DURATION - 5)
        age = _pick(AGE_P, rng)
        gender = _pick(GENDER_P, rng)
        dwell = max(1.5, rng.lognormvariate(0, 0.55) * DWELL_MEAN[age] * 0.75)
        dwell = min(dwell, DURATION - start_sec)

        f0 = int(start_sec * FPS)
        f1 = int((start_sec + dwell) * FPS)
        # Walk from one edge across the frame; some stop mid-aisle.
        x0 = rng.choice([40, W - 40, rng.uniform(0.3, 0.7) * W])
        x1 = rng.uniform(0.05, 0.95) * W
        y0, y1 = rng.uniform(0.35, 0.85) * H, rng.uniform(0.35, 0.9) * H
        n_steps = max(2, (f1 - f0) // 2)

        t = Track(id=tid, box=(0, 0, 1, 1), first_frame=f0, last_frame=f1)
        t.frames = n_steps
        t.det_conf_sum = n_steps * rng.uniform(0.62, 0.94)
        for s in range(n_steps):
            u = s / max(n_steps - 1, 1)
            cx = x0 + (x1 - x0) * u + rng.gauss(0, 6)
            cy = y0 + (y1 - y0) * u + rng.gauss(0, 4)
            scale = 0.05 + 0.05 * (cy / H)
            t.path.append((f0 + s * 2, round(cx, 1), round(cy, 1), round(scale * scale * 1.8, 5)))

        # Small crops and edge-of-frame people get classified less confidently.
        g_conf_base = rng.uniform(0.70, 0.985)
        a_conf_base = rng.uniform(0.42, 0.86)
        for _ in range(max(1, n_steps // 3)):
            t.vote(
                gender_probs=_probs(gender, ["female", "male"],
                                    min(0.99, max(0.34, rng.gauss(g_conf_base, 0.06))), rng),
                age_probs=_probs(age, AGE_LABELS,
                                 min(0.97, max(0.28, rng.gauss(a_conf_base, 0.08))), rng),
            )
        tracks.append(t)
    return tracks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/runs/demo")
    p.add_argument("--people", type=int, default=118)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--captured-at", default="2026-09-05T14:10",
                   help="when this sample was 'recorded' - drives the trend charts")
    p.add_argument("--label", default="sample_store_entrance.mp4")
    a = p.parse_args()

    rng = random.Random(a.seed)
    tracks = make_tracks(a.people, rng)

    occupancy = {}
    for f in range(0, int(DURATION * FPS), 2):
        occupancy[f] = sum(1 for t in tracks if t.first_frame <= f <= t.last_frame)

    meta = {
        "source": a.label,
        "fps": FPS, "width": W, "height": H,
        "frames_total": int(DURATION * FPS),
        "frames_sampled": len(occupancy),
        "duration_sec": DURATION,
        "processing_sec": 386.4,
        "processing_fps": 27.2,
        "crops_classified": sum(t.n_classified for t in tracks),
        "device": "cuda",
        "has_video": False,
        "demo": True,
        "captured_at": a.captured_at,
    }
    config = {"detector": "yolov8n.pt", "det_conf": 0.4, "frame_stride": 2, "iou_thresh": 0.3,
              "gender_model": "abhshkp/footfall-analysis-vit-stage1", "age_model": "peta-svm-v3"}
    model_card = {
        "gender": {"family": "ViT-B/16 fine-tuned (2-stage)", "labels": ["female", "male"],
                   "source": config["gender_model"]},
        "age": {"family": "region colour + LBP histograms, one-vs-rest SVM per bucket",
                "labels": AGE_LABELS, "reported_avg_mA": 0.835, "source": "classifiers.pkl"},
    }

    doc = build_analytics(tracks, meta, config, occupancy, model_card=model_card, bin_sec=15.0)
    doc["insights"] = insights(doc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "analytics.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    k = doc["kpis"]
    print(f"{out/'analytics.json'}: {k['visitors']} visitors, peak {k['peak_occupancy']}, "
          f"avg dwell {k['avg_dwell_sec']}s, {k['male_pct']}%M/{k['female_pct']}%F")


if __name__ == "__main__":
    main()
