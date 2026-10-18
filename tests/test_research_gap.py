"""Tests for the RQ1/RQ2 research code: spectra at the model's input size, FDA
frame picking, statistics, and the scripts that build splits and summarise
results (tasks B4-B8)."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("cv2")

from src.research import degrade, spectrum, stats  # noqa: E402


def _textured(w=48, h=96, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    arr[: h // 2, :, 0] = 200                          # some structure, not just noise
    return Image.fromarray(arr)


# ----------------------------------------------------------- spectrum and FDA
# tests/test_research_degrade.py and test_research_spectrum.py cover the
# degradations, the native-size spectrum and FDA. These cover what the RQ1 and
# RQ3 code adds: measuring a crop at the model's input size, and picking FDA
# frames from a folder.
def test_hf_ratio_falls_as_blur_grows():
    img = _textured(96, 192)
    ratios = [spectrum.hf_ratio(degrade.gaussian(img, s)) for s in (0.01, 1, 2, 3)]
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios


def test_hf_ratio_of_flat_crop_is_zero():
    assert spectrum.hf_ratio(Image.new("RGB", (40, 80), (120, 120, 120))) == 0.0


def test_model_size_spectrum_loses_detail_when_downscaled():
    img = _textured(96, 192)
    g = spectrum.to_gray(img)
    assert g.shape == (spectrum.SIZE, spectrum.SIZE) and g.dtype == np.float32
    _, sharp = spectrum.mean_log_spectrum([img])
    _, small = spectrum.mean_log_spectrum([degrade.resolution(img, 16)])
    assert small[-4:].mean() < sharp[-4:].mean()       # high frequencies drop most


def test_target_frames_spread_over_every_clip(tmp_path):
    from src.training.transforms import target_frames
    for clip in ("a", "b", "c"):
        for i in range(10):
            Image.new("RGB", (8, 8)).save(tmp_path / f"{clip}_{i:03d}.jpg")
    (tmp_path / "manifest.json").write_text("{}")
    picked = target_frames(tmp_path, max_frames=6)
    assert len(picked) == 6 and {p.name[0] for p in picked} == {"a", "b", "c"}
    assert len(target_frames(tmp_path)) == 30


def test_eval_frames_conditions_follow_the_degrade_suite():
    import eval_frames
    from types import SimpleNamespace
    conds = eval_frames.conditions_for(SimpleNamespace(domain="peta-degraded"))
    assert conds[0] == ("clean", None)
    assert conds[1:] == list(degrade.iter_suite())
    assert eval_frames.conditions_for(SimpleNamespace(domain="retail")) == [("clean", None)]
    assert eval_frames.condition_id("clean", None) == "clean"
    assert eval_frames.condition_id("resolution", 16) == "resolution@16"


# ------------------------------------------------------------------- stats (B7)
# tests/test_research_stats.py covers the core metrics; these cover the helpers
# the RQ1 and RQ3 scripts add on top.
def test_accuracy_by_hand():
    assert stats.accuracy(["m", "m", "m", "f"], ["m", "m", "f", "f"]) == 0.75
    assert math.isnan(stats.accuracy([], []))


def test_one_vs_rest_mA():
    per, mean = stats.one_vs_rest_mA(["a", "b", "c"], ["a", "b", "b"], ["a", "b", "c"])
    assert per["a"] == 1.0
    assert per["b"] == pytest.approx(0.5 * (1 + 0.5))
    assert per["c"] == pytest.approx(0.5 * (0 + 1))
    assert mean == pytest.approx((1 + 0.75 + 0.5) / 3)
    per, _ = stats.one_vs_rest_mA(["a", "a"], ["a", "a"], ["a", "b"])
    assert math.isnan(per["b"])                        # no positives for b


def test_ordinal_metrics():
    t = ["Age16-30", "Age31-45", "Age46-60", "AgeAbove61"]
    p = ["Age16-30", "Age16-30", "Age16-30", "AgeAbove61"]
    m = stats.ordinal_metrics(t, p)
    assert m["exact"] == 0.5 and m["off_by_one"] == 0.75 and m["mae_groups"] == pytest.approx(3 / 4)


def test_cohen_kappa_by_hand():
    a = ["m"] * 6 + ["f"] * 4
    b = ["m"] * 5 + ["f"] + ["f"] * 3 + ["m"]
    # po = 0.8, pe = 0.6*0.6 + 0.4*0.4 = 0.52, kappa = 0.28 / 0.48
    assert stats.cohen_kappa(a, b) == pytest.approx(0.28 / 0.48)
    assert stats.cohen_kappa(a, a) == 1.0


def test_mcnemar_exact_p_by_hand():
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [True] * 2
    r = stats.mcnemar(a, b)
    assert (r["b"], r["c"]) == (8, 2) and r["statistic"] is None
    assert r["p"] == pytest.approx(2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10)


def test_auc_by_hand():
    assert stats.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert stats.auc([1, 1, 1], [0, 1, 1]) == 0.5
    assert math.isnan(stats.auc([0.2, 0.3], [1, 1]))


def test_bootstrap_groups_move_together():
    # Two people, ten crops each: person A always right, B always wrong.
    t = np.array(["m"] * 20)
    p = np.array(["m"] * 10 + ["f"] * 10)
    g = np.array(["A"] * 10 + ["B"] * 10)
    point, lo, hi = stats.grouped_bootstrap_ci(stats.accuracy, t, p, groups=g, n_boot=500)
    assert point == 0.5 and lo == 0.0 and hi == 1.0    # only 0, 0.5 and 1 are possible
    r = stats.ci(stats.accuracy, t, p, groups=g, n_boot=500)
    assert r == {"value": 0.5, "lo": 0.0, "hi": 1.0, "n": 2}
    constant = stats.ci(stats.accuracy, t[:10], p[:10], n_boot=50)
    assert constant["lo"] == constant["hi"] == 1.0 and constant["n"] == 10
    empty = stats.ci(stats.accuracy, np.array([]), np.array([]))
    assert empty["n"] == 0 and math.isnan(empty["value"])


# --------------------------------------------------------- scripts: PETA split
def _fake_peta(root):
    rows = {"VIPeR": [("1", "personalMale personalLess30"), ("2", "personalFemale personalLess45"),
                      ("3", "personalMale personalLarger60"), ("4", "personalFemale personalLess15")],
            "CUHK": [(str(i), ("personalMale" if i % 2 else "personalFemale") + " personalLess60")
                     for i in range(1, 21)]}
    for subset, people in rows.items():
        d = root / subset / "archive"
        d.mkdir(parents=True)
        (d / "Label.txt").write_text("\n".join(f"{pid} {tags}" for pid, tags in people), encoding="utf-8")
        for pid, _ in people:
            for n in range(3):                         # three images of every person
                Image.new("RGB", (8, 16)).save(d / f"{pid}_{n}.png")


def test_identity_split_keeps_people_on_one_side(tmp_path):
    import make_peta_split as mps
    _fake_peta(tmp_path)
    recs = [r for r in mps.parse_peta(tmp_path) if r["gender"]]
    assert len(recs) == 24 * 3
    assert {r["age"] for r in recs} >= {"Age16-30", "AgeAbove61", ""}
    mps.assign(recs, "identity", seed=1)
    sides = {}
    for r in recs:
        sides.setdefault((r["subset"], r["pid"]), set()).add(r["split"])
    assert all(len(s) == 1 for s in sides.values())
    info = mps.summary(recs)
    assert info["people_in_train_and_test"] == 0
    assert info["splits"]["train"]["images"] > info["splits"]["val"]["images"]


def test_file_labelled_subset_is_found_and_kept_out_of_identity_splits(tmp_path):
    """CUHK labels files, not people: every image is found, none has a person id."""
    import make_peta_split as mps
    _fake_peta(tmp_path)
    d = tmp_path / "FileLabelled" / "archive"
    d.mkdir(parents=True)
    (d / "Label.txt").write_text("0001.png personalMale personalLess30\n"
                                 "0002.png personalFemale personalLess45\n")
    for n in ("0001", "0002"):
        Image.new("RGB", (8, 16)).save(d / f"{n}.png")
    recs = [r for r in mps.parse_peta(tmp_path) if r["gender"]]
    cuhk = [r for r in recs if r["subset"] == "FileLabelled"]
    assert len(cuhk) == 2 and not any(r["identified"] for r in cuhk)
    assert {r["gender"] for r in cuhk} == {"male", "female"}
    assert all(r["identified"] for r in recs if r["subset"] != "FileLabelled")


def test_image_split_can_put_one_person_on_both_sides(tmp_path):
    import make_peta_split as mps
    _fake_peta(tmp_path)
    recs = mps.assign([r for r in mps.parse_peta(tmp_path) if r["gender"]], "image", seed=1)
    assert mps.summary(recs)["people_in_train_and_test"] > 0


def test_read_peta_split_round_trip(tmp_path):
    from src.research.data import read_peta_split
    path = tmp_path / "split.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "gender", "age", "subset", "pid", "split"])
        w.writeheader()
        w.writerow({"image": "VIPeR/archive/1_0.png", "gender": "male", "age": "", "subset": "VIPeR",
                    "pid": "1", "split": "test"})
        w.writerow({"image": "VIPeR/archive/2_0.png", "gender": "female", "age": "Age31-45",
                    "subset": "VIPeR", "pid": "2", "split": "train"})
    rows = read_peta_split(path, "test")
    assert len(rows) == 1 and rows[0]["age"] is None and rows[0]["group"] == "VIPeR/1"
    assert Path(rows[0]["image"]).is_absolute()


# --------------------------------------------------------------- scripts: MARS
def test_read_mars_index_gives_absolute_frames_and_coarse_age(tmp_path):
    from src.research.data import read_mars_index
    from src.research.mars import build_index
    for split, pid, code in (("bbox_train", "0001", "0001C1T0001"), ("bbox_test", "0002", "0002C2T0003")):
        d = tmp_path / split / pid
        d.mkdir(parents=True)
        for f in range(3):
            Image.new("RGB", (8, 16)).save(d / f"{code}F{f:03d}.jpg")
    labels = tmp_path / "mars_attributes.csv"
    labels.write_text("person-id,camera,tracklet,Age,gender\n1,1,1,3,1\n2,2,3,1,0\n")
    index = tmp_path / "tracklets.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in build_index(tmp_path, labels)))
    rows = {r["tracklet"]: r for r in read_mars_index(index)}
    assert set(rows) == {"0001C1T0001", "0002C2T0003"}
    old = rows["0001C1T0001"]
    assert old["age_coarse"] == "old" and old["split"] == "train" and len(old["frames"]) == 3
    assert all(Path(f).is_absolute() and Path(f).exists() for f in old["frames"])
    assert rows["0002C2T0003"]["age_coarse"] is None                  # teenager: no PETA match


# ------------------------------------------------------------ scripts: summaries
def test_eval_frames_summarise_and_results_merge(tmp_path):
    import eval_frames
    from src.research.data import save_results
    rows = []
    for g in range(6):
        for f in range(4):
            right = g < 4
            rows.append({"domain": "retail", "condition": "clean", "model": "m", "task": "gender",
                         "item": f"r/{g}/{f}", "track": g, "frame": f, "group": f"r/{g}",
                         "probs": {"male": 0.8, "female": 0.2}, "label": "male",
                         "truth": "male" if right else "female", "hf_ratio": 0.1 * f + (0.2 if right else 0)})
    summary = eval_frames.summarise(rows, n_boot=200)
    assert len(summary) == 1
    s = summary[0]
    assert s["n_people"] == 6 and s["accuracy"]["value"] == pytest.approx(4 / 6)
    assert 0 <= s["hf_auc_correct"] <= 1 and len(s["error_by_hf_quintile"]) == 5
    path = tmp_path / "rq1_gap.json"
    save_results(path, [{"domain": "peta", "model": "m", "x": 1}], {})
    save_results(path, summary, {}, replace=lambda r: r.get("domain") == "retail")
    save_results(path, summary, {}, replace=lambda r: r.get("domain") == "retail")
    doc = json.loads(path.read_text())
    assert [r["domain"] for r in doc["rows"]] == ["peta", "retail"]


# ----------------------------------------------------------- experiment figures
def test_experiment_figures_draw_from_results_files(tmp_path):
    """make_figures.py draws RQ1-RQ3 from results files, and skips missing ones.
    The numbers here are made up and only ever live in tmp_path."""
    pytest.importorskip("matplotlib")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_figures", Path(__file__).resolve().parents[1] / "paper" / "figures" / "make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    ci = {"value": 0.8, "lo": 0.75, "hi": 0.85, "n": 10}
    rq1_rows = [{"domain": d, "condition": c, "task": t, "model": "m", "accuracy": ci, "balanced_accuracy": ci}
                for d, c in (("peta", "clean"), ("retail", "clean"), ("peta-degraded", "clean"),
                             ("peta-degraded", "downscale@16"))
                for t in ("gender", "age")]
    rq1_rows += [{"domain": d, "condition": "clean",
                  "spectrum": {"freqs": [0.1, 0.3], "log_power": [2.0, 0.5], "n": 5}} for d in ("peta", "retail")]
    stat = lambda v: {"mean": v, "sd": 0.01, "values": [v - 0.01, v, v + 0.01]}
    rq2_rows = [{"condition": c, "seeds": [1, 2, 3], "peta_test_accuracy": stat(0.9),
                 "retail_person_accuracy": stat(0.7)} for c in ("baseline", "fda")]
    strategies = [{"strategy": s, "accuracy": ci} for s in ("single", "majority", "mean", "confidence", "quality")]
    rq3_rows = [{"task": t, "strategies": strategies,
                 "by_k": [{"k": k, "majority": 0.7, "mean": 0.71, "confidence": 0.72, "quality": 0.75}
                          for k in range(1, 4)]} for t in ("gender", "age")]
    for name, rows in (("rq1_gap.json", rq1_rows), ("rq2_adaptation.json", rq2_rows),
                       ("rq3_pooling.json", rq3_rows)):
        (tmp_path / name).write_text(json.dumps({"meta": {}, "rows": rows}))
    out = tmp_path / "figs"
    out.mkdir()
    mf.rq1(tmp_path, out)
    mf.rq2(tmp_path, out)
    mf.rq3(tmp_path, out)
    assert {p.name for p in out.iterdir()} == {"rq1_gap.pdf", "rq2_adaptation.pdf", "rq3_pooling.pdf"}
    empty = tmp_path / "none"
    empty.mkdir()
    mf.rq1(empty, empty)                                   # no results yet: skipped, no error
    assert not list(empty.iterdir())
