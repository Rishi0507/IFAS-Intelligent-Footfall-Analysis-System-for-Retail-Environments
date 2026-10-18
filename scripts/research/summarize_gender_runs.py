"""Summarise the gender output of several pipeline runs of the same video.

Used to record what the gender loader fix changed (research plan, task B1): the
same clip processed before and after the fix, with stage 1 and stage 2. Runs of
one video share their tracking, so person ids line up across them.

    python scripts/research/summarize_gender_runs.py \\
        --run "stage1, encoder skipped=outputs/runs/<before-stage1>/analytics.json" \\
        --run "stage1, fixed=outputs/runs/<after-stage1>/analytics.json" \\
        --out paper/results/b1_gender_rebaseline.json

Each --run is NAME=PATH. The name says which model and loader produced the run,
since analytics.json records the model but not whether its weights loaded.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

LOW_CONF = 0.7


def summarize(doc):
    """One analytics document -> counts and confidence of its gender labels."""
    tracks = doc["tracks"]
    conf = [t["gender_conf"] for t in tracks if t.get("gender_conf") is not None]
    return {
        "gender_model": doc.get("config", {}).get("gender_model"),
        "people": len(tracks),
        "female": sum(t.get("gender") == "female" for t in tracks),
        "male": sum(t.get("gender") == "male" for t in tracks),
        "low_conf": sum(c < LOW_CONF for c in conf),
        "mean_conf": round(sum(conf) / len(conf), 3) if conf else None,
    }


def per_person(named_docs):
    """{person id: {run name: [gender, conf, votes]}} over every run."""
    out = {}
    for name, doc in named_docs:
        for t in doc["tracks"]:
            conf = t.get("gender_conf")
            out.setdefault(str(t["id"]), {})[name] = [
                t.get("gender"), None if conf is None else round(conf, 3), t.get("gender_votes")]
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def changed(people, a, b):
    """How many people carry a different gender label in run a and run b."""
    return sum(1 for runs in people.values()
               if a in runs and b in runs and runs[a][0] != runs[b][0])


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    named = []
    for spec in a.run:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--run needs NAME=PATH, got {spec!r}")
        with open(path, encoding="utf-8") as f:
            named.append((name.strip(), json.load(f)))

    sources = {doc["meta"].get("source") for _, doc in named}
    if len(sources) > 1:
        p.error(f"runs come from different videos: {sorted(map(str, sources))}")

    people = per_person(named)
    names = [n for n, _ in named]
    result = {
        "meta": {
            "commit": _commit(),
            "date": _dt.date.today().isoformat(),
            "video": sources.pop(),
            "models": sorted({d.get("config", {}).get("gender_model") or "" for _, d in named}),
            "seeds": None,
            "low_conf_below": LOW_CONF,
        },
        "rows": [{"run": n, **summarize(d)} for n, d in named],
        "changed_labels": {f"{a} -> {b}": changed(people, a, b)
                           for i, a in enumerate(names) for b in names[i + 1:]},
        "people": people,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for r in result["rows"]:
        print(f"{r['run']:28} female {r['female']:3}  male {r['male']:3}  "
              f"conf<{LOW_CONF}: {r['low_conf']:3}  mean conf {r['mean_conf']}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
