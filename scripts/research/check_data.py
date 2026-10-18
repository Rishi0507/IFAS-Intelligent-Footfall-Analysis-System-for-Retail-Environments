"""Check the retail data before any number is reported from it (task A3).

    python scripts/research/check_data.py --test footage/test --fda footage/fda \\
        --stage2 footage/stage2_training --runs outputs/runs/cam1-* --report data/check.json

Three things can quietly spoil RQ1-RQ3, and this catches each of them:

1. **A test clip that was also training data.** A stage-2 training clip used as
   test data inflates every result (docs/research-plan.md, Section 6). The same
   footage often comes back cut or re-encoded under a new name, so clips are
   compared by content as well as by file hash: one small fingerprint per second
   of video (a difference hash of the frame), and two clips that share more than
   `--shared` of the shorter clip's seconds count as the same footage.
2. **A test clip whose frames also went to FDA.** FDA frames must come from
   clips nobody labels or scores.
3. **A clip without written permission.** Every test and FDA clip needs a row in
   the retail clip table in docs/research-data.md, with the Permission cell
   filled and the Use cell saying test or FDA frames. Every run passed with
   `--runs` must come from a registered test clip.

Two recordings of the same empty camera view on different days can look alike
to the fingerprint. If a pair is flagged and you have checked by eye that the
footage differs, pass it with `--allow "clipA.mp4:clipB.mp4"`.

Exits with status 1 when anything fails, so it can gate a results script.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fda_frames import clips_in, sha1  # noqa: E402

REGISTRY = ROOT / "docs" / "research-data.md"


# --------------------------------------------------------------- fingerprints
def fingerprint(clip, every_sec=1.0):
    """One 64-bit difference hash per `every_sec` of video, as uint64."""
    import cv2
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * every_sec))
    hashes = []
    for idx in range(0, max(n, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        hashes.append(int(np.packbits(bits).view(">u8")[0]))
    cap.release()
    return np.array(hashes, dtype=np.uint64)


def shared_share(a, b, max_bits=5):
    """Share of the shorter clip's fingerprints that have a near twin in the other."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    x = a[:, None] ^ b[None, :]
    bits = np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(axis=2)
    return float((bits.min(axis=1) <= max_bits).mean())


# ------------------------------------------------------------------ registry
def read_registry(path=REGISTRY):
    """Rows of the table under the "Retail clips" heading of docs/research-data.md."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"^## [^\n]*Retail clips[^\n]*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    rows, header = [], None
    for line in (m.group(1) if m else "").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if set("".join(cells)) <= set(":- "):
            continue
        row = dict(zip(header, cells))
        if row.get("clip"):
            rows.append(row)
    return rows


def _use(row):
    use = row.get("use", "").lower()
    return "test" if "test" in use and "fda" not in use else "fda" if "fda" in use else None


# --------------------------------------------------------------------- check
def check(test=(), fda=(), stage2=(), runs=(), registry=REGISTRY, shared=0.3, allow=(), say=print):
    clips = {role: clips_in(paths) for role, paths in (("test", test), ("fda", fda), ("stage2", stage2))}
    allowed = {frozenset(p.split(":")) for p in allow}
    errors, warnings = [], []
    info = {}
    for role, cs in clips.items():
        for c in cs:
            info[c] = {"role": role, "sha1": sha1(c), "fingerprint": fingerprint(c)}
        say(f"{role}: {len(cs)} clips")

    # 1 and 2: the same footage in two roles that must stay apart.
    for (ra, rb) in (("test", "stage2"), ("test", "fda")):
        for a, b in itertools.product(clips[ra], clips[rb]):
            if frozenset((a.name, b.name)) in allowed:
                continue
            if info[a]["sha1"] == info[b]["sha1"]:
                errors.append(f"{a.name} ({ra}) and {b.name} ({rb}) are the same file")
                continue
            s = shared_share(info[a]["fingerprint"], info[b]["fingerprint"])
            if s >= shared:
                errors.append(f"{a.name} ({ra}) and {b.name} ({rb}) share {s:.0%} of their footage")
    # Duplicates inside one role only count people twice.
    for role, cs in clips.items():
        for a, b in itertools.combinations(cs, 2):
            if info[a]["sha1"] == info[b]["sha1"]:
                warnings.append(f"{a.name} and {b.name} are the same file ({role})")

    # 3: permission and use, from the registry.
    reg = {r["clip"]: r for r in read_registry(registry)}
    for role in ("test", "fda"):
        for c in clips[role]:
            r = reg.get(c.name)
            if r is None:
                errors.append(f"{c.name} ({role}) has no row in the clip table of {Path(registry).name}")
                continue
            if not r.get("permission (who, date, where the signed copy is)", r.get("permission", "")).strip():
                errors.append(f"{c.name} has no written permission recorded")
            if _use(r) != role:
                errors.append(f"{c.name} is used as {role} but registered as '{r.get('use', '')}'")
    test_names = {c.name for c in clips["test"]} | {n for n, r in reg.items() if _use(r) == "test"}
    people = 0
    for d in map(Path, runs):
        doc_path = d / "analytics.json"
        if not doc_path.exists():
            errors.append(f"{d} has no analytics.json")
            continue
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        src = Path(doc["meta"].get("source") or "").name
        if src not in test_names:
            errors.append(f"run {d.name} comes from '{src}', which is not a registered test clip")
        if not (d / "crops.jsonl").exists():
            errors.append(f"run {d.name} has no crops.jsonl (process it with --export-crops)")
        people += int(doc["kpis"].get("visitors", 0))
    if runs and not 150 <= people <= 300:
        warnings.append(f"{people} people in the test runs; the plan asks for about 150 to 300")

    report = {"clips": {role: [c.name for c in cs] for role, cs in clips.items()},
              "test_people": people if runs else None, "errors": errors, "warnings": warnings,
              "ok": not errors}
    for w in warnings:
        say(f"warning: {w}")
    for e in errors:
        say(f"FAIL: {e}")
    say("data check passed" if not errors else f"data check failed: {len(errors)} problem(s)")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", nargs="*", default=[], help="test clips or folders")
    p.add_argument("--fda", nargs="*", default=[], help="FDA clips or folders")
    p.add_argument("--stage2", nargs="*", default=[], help="stage-2 training clips or folders")
    p.add_argument("--runs", nargs="*", default=[], help="processed test runs")
    p.add_argument("--registry", default=str(REGISTRY))
    p.add_argument("--shared", type=float, default=0.3,
                   help="share of seconds two clips must share to count as the same footage")
    p.add_argument("--allow", nargs="*", default=[], help='pairs checked by eye, "a.mp4:b.mp4"')
    p.add_argument("--report", default=None, help="write the report as JSON here")
    a = p.parse_args()
    report = check(a.test, a.fda, a.stage2, a.runs, a.registry, a.shared, a.allow)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
