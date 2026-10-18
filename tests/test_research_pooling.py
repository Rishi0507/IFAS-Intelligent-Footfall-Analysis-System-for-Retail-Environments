"""Tests for RQ3: quality cues, pooling strategies, the quality model, the
tracker's pooling option, retail label reading and the pooling study script
(tasks A4-A7)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

from src.research import pooling as P  # noqa: E402
from src.research import quality as Q  # noqa: E402


def _person_crop(seed=0, w=40, h=100):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8))


# --------------------------------------------------------------- quality (A4)
@pytest.mark.parametrize("size", [(40, 100), (64, 128), (120, 260)])
def test_sharpness_falls_monotonically_with_blur(size):
    """Over the range where a crop still has detail; past that both cues level
    off at the 8-bit floor, far below a sharp crop."""
    from PIL import ImageFilter
    for seed in range(3):
        crop = _person_crop(seed, *size)
        blurred = [crop.filter(ImageFilter.GaussianBlur(s)) for s in (0.01, 0.3, 0.6, 1.0)]
        hf = [Q.fft_sharpness(b) for b in blurred]
        lap = [Q.laplacian_variance(b) for b in blurred]
        assert all(a > b for a, b in zip(hf, hf[1:])), hf
        assert all(a > b for a, b in zip(lap, lap[1:])), lap
        gone = crop.filter(ImageFilter.GaussianBlur(8))
        assert Q.fft_sharpness(gone) < 0.05 * hf[0]
        assert Q.laplacian_variance(gone) < 0.05 * lap[0]


def _frame_pair(shift):
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (5, 5), 0)
    moved = base.copy()
    if shift:
        box = base[60:200, 120:180]
        moved[60:200, 120 + shift:180 + shift] = box     # only the "person" moves
    return base, moved


def test_motion_is_zero_on_a_static_pair_and_positive_when_the_person_moves():
    box = (120, 60, 180, 200)
    still = Q.flow_stats(*Q.frame_flow(*_frame_pair(0)), box)
    moved = Q.flow_stats(*Q.frame_flow(*_frame_pair(6)), box)
    assert still["box_mag"] < 0.05 and still["bg_mag"] < 0.05
    assert moved["box_mag"] - moved["bg_mag"] > 1.0


def test_flow_magnitudes_are_in_original_pixels():
    a, b = _frame_pair(6)
    big_a, big_b = cv2.resize(a, (1280, 960)), cv2.resize(b, (1280, 960))
    small = Q.flow_stats(*Q.frame_flow(a, b, max_width=320), (120, 60, 180, 200))
    large = Q.flow_stats(*Q.frame_flow(big_a, big_b, max_width=320), (480, 240, 720, 800))
    assert large["box_mag"] == pytest.approx(4 * small["box_mag"], rel=0.35)   # 4x the pixels moved


def test_truncation_occlusion_and_size():
    assert Q.truncation((0, 10, 30, 90), 320, 240) == 1.0
    assert Q.truncation((10, 10, 30, 239), 320, 240) == 1.0
    assert Q.truncation((10, 10, 30, 90), 320, 240) == 0.0
    assert Q.occlusion((0, 0, 10, 10), []) == 0.0
    assert Q.occlusion((0, 0, 10, 10), [(5, 0, 15, 10), (50, 50, 60, 60)]) == pytest.approx(50 / 150)
    assert Q.size_frac((0, 0, 32, 24), 320, 240) == pytest.approx(0.01)


def test_crop_cues_has_every_cue():
    cues = Q.crop_cues(_person_crop(), (10, 10, 50, 110), 320, 240, det_conf=0.8,
                       flow={"box_mag": 2.0, "bg_mag": 0.5}, overlap=0.2)
    assert set(cues) == set(Q.CUE_NAMES)
    assert cues["motion"] == 1.5 and cues["occlusion"] == 0.2 and cues["det_conf"] == 0.8
    assert Q.crop_cues(_person_crop(), (10, 10, 50, 110), 320, 240)["motion"] is None


# --------------------------------------------------------------- pooling (A5)
M, F = {"male": 0.9, "female": 0.1}, {"male": 0.4, "female": 0.6}


def test_majority_matches_the_tracker_rule():
    from src.inference.tracker import Track
    votes = [(M, {}), (F, {}), (F, {})]
    t = Track(id=1, box=(0, 0, 1, 1), first_frame=0, last_frame=0)
    for probs, _ in votes:
        t.vote(gender_probs=probs)
    label, conf = P.majority(votes)
    assert label == t.gender()[0] == "female"
    assert conf == pytest.approx(t.gender()[1])


def test_majority_tie_goes_to_higher_summed_probability():
    assert P.majority([(M, {}), (F, {})])[0] == "male"            # 1.3 vs 0.7


def test_mean_and_confidence_can_overrule_the_vote():
    votes = [({"male": 0.99, "female": 0.01}, {}), (F, {}), (F, {})]
    assert P.majority(votes)[0] == "female"
    assert P.mean_prob(votes)[0] == "male"                         # 0.597 vs 0.403
    label, conf = P.confidence_weighted(votes)
    assert label == "male" and 0.5 < conf < 1


def test_quality_weighting_trusts_the_sharp_crop():
    class Fixed:
        def predict(self, cues):
            return np.array([c["w"] for c in cues])
    votes = [(M, {"w": 0.95}), (F, {"w": 0.05}), (F, {"w": 0.05})]
    assert P.quality_weighted(votes, Fixed())[0] == "male"
    with pytest.raises(ValueError):
        P.quality_weighted(votes, None)


def test_single_frame_is_one_of_the_votes_and_empty_is_safe():
    rng = np.random.default_rng(0)
    assert P.single_frame([(M, {}), (F, {})], rng)[0] in ("male", "female")
    for s in P.STRATEGIES:
        if s != "quality":
            assert P.pool([], s) == (None, 0.0)
    with pytest.raises(ValueError):
        P.pool([(M, {})], "nonsense")


def _synthetic_crops(n=400, seed=0):
    """Blurred or truncated crops are right less often."""
    rng = np.random.default_rng(seed)
    cues, correct = [], []
    for _ in range(n):
        hf = rng.uniform(0.0, 0.3)
        trunc = float(rng.random() < 0.3)
        p = 1 / (1 + np.exp(-(-1 + 12 * hf - 1.5 * trunc)))
        cues.append({"hf_ratio": hf, "lap_var": 0.01, "motion": None, "truncated": trunc,
                     "occlusion": 0.0, "size": 0.05, "det_conf": 0.7})
        correct.append(rng.random() < p)
    return cues, np.array(correct)


def test_quality_model_learns_the_right_direction_and_round_trips(tmp_path):
    cues, correct = _synthetic_crops()
    m = P.QualityModel(l2=0.1).fit(cues, correct)
    coef = dict(zip(P.FEATURE_NAMES, m.coef))
    assert coef["hf_ratio"] > 0 and coef["truncated"] < 0
    sharp = {"hf_ratio": 0.28, "truncated": 0.0}
    blurry = {"hf_ratio": 0.02, "truncated": 1.0}
    assert m.predict([sharp])[0] > m.predict([blurry])[0]
    m.save(tmp_path / "q.json")
    again = P.QualityModel.load(tmp_path / "q.json")
    assert np.allclose(again.predict(cues[:10]), m.predict(cues[:10]), atol=1e-5)
    bad = json.loads((tmp_path / "q.json").read_text())
    bad["features"] = ["x"]
    with pytest.raises(ValueError):
        P.QualityModel.from_dict(bad)


# ------------------------------------------------------- tracker option (A6)
def test_tracker_uses_the_requested_pooling():
    from src.inference.tracker import IoUTracker
    tr = IoUTracker(pooling="mean")
    (t, _, _), = tr.update([((0, 0, 10, 20), 0.9)], 0, 100.0)
    for probs in ({"male": 0.99, "female": 0.01}, F, F):
        t.vote(gender_probs=probs, cues={})
    assert t.gender()[0] == "male"                                 # majority would say female
    assert t.n_classified == 3 and len(t.vote_log) == 3
    with pytest.raises(ValueError):
        IoUTracker(pooling="quality")


# ------------------------------------------------- labels and pooling study
def _labelled_run(root, name, people, annotators):
    """people: {track: (truth gender, [crop correct?...], truncated?)}"""
    d = root / name
    (d / "crops").mkdir(parents=True)
    lines = []
    for track, (truth, rights, trunc) in people.items():
        other = "female" if truth == "male" else "male"
        for f, right in enumerate(rights):
            rel = f"crops/{track}/{f:06d}.jpg"
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            _person_crop(track * 100 + f).save(d / rel)
            lab = truth if right else other
            probs = {lab: 0.8, ("female" if lab == "male" else "male"): 0.2}
            hf = 0.25 if right else 0.03
            lines.append({"track": track, "frame": f, "t": f / 10, "box": [0, 0, 10, 20], "det_conf": 0.8,
                          "frame_w": 320, "frame_h": 240, "crop": rel, "flow": None, "overlap": 0.0,
                          "cues": {"hf_ratio": hf, "lap_var": 0.01, "motion": None,
                                   "truncated": float(trunc), "occlusion": 0.0, "size": 0.01,
                                   "det_conf": 0.8},
                          "pred": {"gender": probs, "age": None}})
    (d / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    for who, labels in annotators.items():
        (d / f"labels_{who}.json").write_text(json.dumps({"run": name, "annotator": who, "labels": labels}))
    return d


def test_read_labels_consensus_final_and_unsure(tmp_path):
    from src.research.data import read_labels, read_retail
    d = _labelled_run(tmp_path, "r1", {1: ("male", [True], False), 2: ("female", [True], False),
                                        3: ("male", [True], False)},
                      {"a": {"1": {"gender": "male", "age": "Age16-30"},
                             "2": {"gender": "female", "age": "unsure"},
                             "3": {"gender": "male", "age": "Age31-45"}},
                       "b": {"1": {"gender": "male", "age": "Age16-30"},
                             "2": {"gender": "female", "age": "unsure"},
                             "3": {"gender": "female", "age": "Age31-45"}}})
    truth, info = read_labels(d)
    assert truth[1] == {"gender": "male", "age": "Age16-30"}
    assert truth[3]["gender"] is None and info["disagreements"] == 1
    assert read_labels(d, "b")[0][3]["gender"] == "female"
    people, info = read_retail([d])
    by = {p["track"]: p for p in people}
    assert by[2]["truth"]["age"] is None and info["unsure"]["age"] == 1
    (d / "labels_final.json").write_text(json.dumps({"labels": {"3": {"gender": "male", "age": None}}}))
    assert read_labels(d)[0] == {3: {"gender": "male", "age": None}}


def test_pooling_study_on_synthetic_tracks_with_known_answers(tmp_path):
    import pooling_eval
    from src.research.data import read_retail
    rng = np.random.default_rng(1)
    people = {}
    for track in range(1, 31):
        truth = "male" if track % 2 else "female"
        # Every person: 5 sharp crops all right, 7 blurred crops all wrong. The
        # majority vote is wrong for everyone; weighting by sharpness is right.
        rights = [True] * 5 + [False] * 7
        rng.shuffle(rights)
        people[track] = (truth, rights, track % 3 == 0)
    labels = {str(t): {"gender": g, "age": None} for t, (g, _, _) in people.items()}
    d = _labelled_run(tmp_path, "synthetic", people, {"a": labels})
    ppl, _ = read_retail([d])
    entries = pooling_eval.person_votes(ppl, "gender")
    assert len(entries) == 30 and all(len(v) == 12 for _, v, _ in entries)
    models = pooling_eval.fit_cv(entries, folds=5, l2=0.1, seed=0)
    res, by_k, subsets, _ = pooling_eval.evaluate(entries, models, P.STRATEGIES, n_draws=3, seed=0, n_boot=200)
    acc = {r["strategy"]: r["accuracy"]["value"] for r in res}
    assert acc["majority"] == 0.0 and acc["mean"] == 0.0
    assert acc["quality"] == 1.0
    q = next(r for r in res if r["strategy"] == "quality")
    assert q["mcnemar_vs_majority"]["p"] < 1e-6
    assert by_k[0]["k"] == 1 and by_k[-1]["k"] == 12 and by_k[-1]["quality"] == 1.0
    assert subsets["truncated"]["n_people"] == 10


def test_pooling_eval_cli_writes_results(tmp_path, monkeypatch):
    import pooling_eval
    people = {t: ("male" if t % 2 else "female", [True] * 4 + [False] * 2, False) for t in range(1, 13)}
    labels = {str(t): {"gender": g, "age": None} for t, (g, _, _) in people.items()}
    d = _labelled_run(tmp_path, "run", people, {"a": labels})
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["pooling_eval.py", "--runs", str(d), "--tasks", "gender",
                                      "--folds", "3", "--draws", "2", "--n-boot", "100",
                                      "--out-dir", str(out)])
    pooling_eval.main()
    doc = json.loads((out / "rq3_pooling.json").read_text())
    assert doc["rows"][0]["task"] == "gender" and doc["rows"][0]["n_people"] == 12
    assert {"commit", "date", "models", "seeds"} <= set(doc["meta"])
    assert P.QualityModel.load(out / "quality_model_gender.json").coef is not None


def test_agreement_kappa(tmp_path):
    import agreement
    d = _labelled_run(tmp_path, "r", {1: ("male", [True], False), 2: ("female", [True], False)},
                      {"x": {"1": {"gender": "male", "age": "Age16-30"}, "2": {"gender": "female", "age": "unsure"}},
                       "y": {"1": {"gender": "male", "age": "Age31-45"}, "2": {"gender": "male", "age": "Age16-30"}}})
    rows, dis = agreement.compare([d], "x", "y")
    by = {r["field"]: r for r in rows}
    assert by["gender"]["n_people"] == 2 and by["gender"]["raw_agreement"] == 0.5
    assert by["age"]["n_people"] == 1                                # unsure is left out
    assert len(dis) == 2


# ----------------------------------------------------- learned attention (A5)
def _attention_tracks(n_people, seed, d=8):
    """Each person: 4 clean frames that show the truth along feature 1, marked by
    feature 0 = +1, and 8 bad frames marked -1 whose feature 1 is loud noise of
    random sign, so they say nothing about the person. Averaging every frame is
    often wrong; a head that learns to trust the marker is always right."""
    rng = np.random.default_rng(seed)
    tracks, labels, good = [], [], []
    for i in range(n_people):
        y = "male" if i % 2 else "female"
        sign = 1.0 if y == "male" else -1.0
        frames, mask = [], []
        for k in range(12):
            clean = k < 4
            f = rng.normal(0, 0.3, d)
            f[0] = 1.0 if clean else -1.0
            f[1] = sign if clean else rng.choice([-1.6, 1.6])
            frames.append(f)
            mask.append(clean)
        order = rng.permutation(12)
        tracks.append(np.array(frames, np.float32)[order])
        good.append(np.array(mask)[order])
        labels.append(y)
    return tracks, labels, good


def test_attention_learns_to_trust_the_informative_frames(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.research.attention import AttentionPooler
    train, ytrain, _ = _attention_tracks(40, seed=0)
    test, ytest, good = _attention_tracks(20, seed=1)
    m = AttentionPooler(hidden=8, seed=0).fit(train, ytrain, epochs=300)
    preds = [m.predict(t) for t in test]
    assert np.mean([p[0] == y for p, y in zip(preds, ytest)]) == 1.0
    # Plain averaging of the same features is often wrong.
    assert np.mean([(t[:, 1].mean() > 0) == (y == "male") for t, y in zip(test, ytest)]) < 0.85
    w_good = np.mean([p[2][g].sum() for p, g in zip(preds, good)])
    assert w_good > 0.5               # uniform weights would give the clean frames 4/12
    assert all(abs(p[2].sum() - 1) < 1e-5 for p in preds)
    m.save(tmp_path / "attn.json")
    back = AttentionPooler.load(tmp_path / "attn.json")
    assert [back.predict(t)[0] for t in test] == [p[0] for p in preds]
    assert m.predict(np.zeros((0, 8)))[0] is None


def test_feature_files_round_trip_and_feed_training(tmp_path):
    from src.research.attention import load_features, save_features, tracks_from_file
    vecs = np.arange(12, dtype=np.float32).reshape(6, 2)
    save_features(tmp_path / "f.npz", [f"m/{i}" for i in range(6)], vecs,
                  groups=["a", "a", "b", "b", "b", "c"], labels=["male", "male", "female", "female",
                                                                 "female", "None"])
    lookup, extra = load_features(tmp_path / "f.npz")
    assert lookup["m/3"].tolist() == [6.0, 7.0] and set(extra) == {"groups", "labels"}
    tracks, labels = tracks_from_file(tmp_path / "f.npz")
    assert labels == ["male", "female"] and [len(t) for t in tracks] == [2, 3]   # c has no label
    save_features(tmp_path / "plain.npz", ["x"], np.zeros((1, 2)))
    with pytest.raises(ValueError):
        tracks_from_file(tmp_path / "plain.npz")


def _with_features(run_dir, seed=0):
    """features_gender.npz for a _labelled_run: the clean marker of
    _attention_tracks on crops that are right, the noisy one on the rest."""
    from src.research.attention import save_features
    rng = np.random.default_rng(seed)
    items, vecs = [], []
    for r in (json.loads(x) for x in (run_dir / "crops.jsonl").read_text().splitlines()):
        probs = r["pred"]["gender"]
        lab = max(probs, key=probs.get)
        truth = json.loads((run_dir / "labels_a.json").read_text())["labels"][str(r["track"])]["gender"]
        right = lab == truth
        sign = 1.0 if truth == "male" else -1.0
        f = rng.normal(0, 0.3, 8)
        f[0] = 1.0 if right else -1.0
        f[1] = sign if right else rng.choice([-1.6, 1.6])
        items.append(f"{run_dir.name}/{r['track']}/{r['frame']:06d}")
        vecs.append(f)
    save_features(run_dir / "features_gender.npz", items, vecs)


def test_pooling_study_includes_the_attention_baseline(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import pooling_eval
    rng = np.random.default_rng(2)
    people = {}
    for track in range(1, 31):
        rights = [True] * 5 + [False] * 7               # majority wrong for everyone
        rng.shuffle(rights)
        people[track] = ("male" if track % 2 else "female", rights, False)
    labels = {str(t): {"gender": g, "age": None} for t, (g, _, _) in people.items()}
    d = _labelled_run(tmp_path, "attn", people, {"a": labels})
    _with_features(d)
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["pooling_eval.py", "--runs", str(d), "--tasks", "gender",
                                      "--folds", "5", "--draws", "2", "--n-boot", "100",
                                      "--attention-hidden", "8", "--out-dir", str(out)])
    pooling_eval.main()
    row = json.loads((out / "rq3_pooling.json").read_text())["rows"][0]
    acc = {s["strategy"]: s["accuracy"]["value"] for s in row["strategies"]}
    assert acc["majority"] == 0.0 and acc["attention"] >= 0.9 and acc["quality"] == 1.0
    assert row["attention"]["fitted_on"].startswith("5-fold")
    assert row["attention"]["vs_quality"]["n_people"] == 30
    assert row["attention"]["vs_quality"]["mean_spearman"] > 0.5   # both trust the sharp crops
    assert any("attention" in b for b in row["by_k"])


def test_pooling_study_without_features_skips_attention(tmp_path, monkeypatch):
    import pooling_eval
    people = {t: ("male" if t % 2 else "female", [True] * 4 + [False] * 2, False) for t in range(1, 13)}
    labels = {str(t): {"gender": g, "age": None} for t, (g, _, _) in people.items()}
    d = _labelled_run(tmp_path, "nofeat", people, {"a": labels})
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["pooling_eval.py", "--runs", str(d), "--tasks", "gender",
                                      "--folds", "3", "--draws", "1", "--n-boot", "50",
                                      "--out-dir", str(out)])
    pooling_eval.main()
    row = json.loads((out / "rq3_pooling.json").read_text())["rows"][0]
    assert "attention" not in {s["strategy"] for s in row["strategies"]}
    assert "skipped" in row["attention"]


# ------------------------------------------------------ A3 data tools
def _video(path, seed, n=60, fps=10, size=(160, 120), start=0):
    """A clip of drifting random blobs; `start` skips into the same stream, so
    two calls with one seed and different starts give overlapping footage."""
    rng = np.random.default_rng(seed)
    base = (rng.random((size[1] * 2, size[0] * 2, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 6)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(start, start + n):
        x, y = (i * 3) % size[0], (i * 2) % size[1]
        w.write(np.ascontiguousarray(base[y:y + size[1], x:x + size[0]]))
    w.release()
    return path


def test_fda_frames_exports_evenly_and_skips_done_clips(tmp_path, monkeypatch):
    import fda_frames
    clips = tmp_path / "fda"
    clips.mkdir()
    _video(clips / "a.avi", 1, n=50)
    _video(clips / "b.avi", 2, n=30)
    out = tmp_path / "frames"
    monkeypatch.setattr(sys, "argv", ["fda_frames.py", str(clips), "--per-clip", "10", "--out", str(out)])
    fda_frames.main()
    man = json.loads((out / "manifest.json").read_text())
    assert [c["clip"] for c in man["clips"]] == ["a.avi", "b.avi"]
    assert len(man["clips"][0]["frames"]) == 10 and man["clips"][0]["frames"][-1] == 49
    assert len(list(out.glob("*.jpg"))) == 20
    fda_frames.main()                                    # a second run adds nothing
    assert len(json.loads((out / "manifest.json").read_text())["clips"]) == 2


def _registry(path, rows):
    lines = ["# Data", "", "## 3. Retail clips (task A3)", "",
             "| Clip | Store and camera | Recorded | Length | Permission (who, date, where the signed copy is) | Use |",
             "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    lines += [f"| {c} | Shop, cam 1 | 2026-10-20 | 1 min | {perm} | {use} |" for c, perm, use in rows]
    lines += ["", "## 4. Labelling", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_check_data_finds_reused_footage_and_missing_permission(tmp_path):
    import check_data
    test_dir, s2_dir, fda_dir = (tmp_path / d for d in ("test", "stage2", "fda"))
    for d in (test_dir, s2_dir, fda_dir):
        d.mkdir()
    _video(test_dir / "t1.avi", 1, n=60)
    _video(test_dir / "t2.avi", 2, n=60)
    _video(s2_dir / "old_training.avi", 1, n=40, start=20, size=(160, 120))   # a cut of t1
    _video(fda_dir / "f1.avi", 3, n=40)
    reg = _registry(tmp_path / "data.md", [("t1.avi", "Owner, 2026-10-01, Drive/consent/t1.pdf", "test"),
                                           ("t2.avi", "", "test"),
                                           ("f1.avi", "Owner, 2026-10-01, Drive/consent/f1.pdf", "FDA frames")])
    r = check_data.check([test_dir], [fda_dir], [s2_dir], registry=reg, say=lambda *_: None)
    assert not r["ok"]
    assert any("t1.avi (test) and old_training.avi (stage2) share" in e for e in r["errors"])
    assert any("t2.avi has no written permission" in e for e in r["errors"])
    assert not any("f1.avi" in e for e in r["errors"])                 # unrelated footage, registered
    assert not any("t2.avi (test) and" in e for e in r["errors"])
    allowed = check_data.check([test_dir], [fda_dir], [s2_dir], registry=reg, say=lambda *_: None,
                               allow=["t1.avi:old_training.avi"])
    assert not any("share" in e for e in allowed["errors"])


def test_check_data_passes_clean_data_and_checks_runs(tmp_path):
    import check_data
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    _video(test_dir / "t1.avi", 1)
    reg = _registry(tmp_path / "data.md", [("t1.avi", "Owner, 2026-10-01, Drive/consent/t1.pdf", "test")])
    good = tmp_path / "runs" / "good"
    bad = tmp_path / "runs" / "bad"
    for d, src in ((good, "t1.avi"), (bad, "random_web_clip.mp4")):
        d.mkdir(parents=True)
        (d / "analytics.json").write_text(json.dumps({"meta": {"source": src}, "kpis": {"visitors": 3}}))
        (d / "crops.jsonl").write_text("")
    r = check_data.check([test_dir], runs=[good], registry=reg, say=lambda *_: None)
    assert r["ok"] and r["test_people"] == 3 and r["warnings"]          # 3 people is far below 150
    r = check_data.check([test_dir], runs=[good, bad], registry=reg, say=lambda *_: None)
    assert any("random_web_clip.mp4" in e for e in r["errors"])
