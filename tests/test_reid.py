import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.tracking import ReIDBank


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(16).astype(np.float32)
    return v / np.linalg.norm(v)


def test_no_match():
    bank = ReIDBank(cosine_threshold=0.85)
    bank.add(_emb(1), 10, "male", fps=25.0, last_seen_frame=100)
    assert bank.query(_emb(999)) is None


def test_exact_match():
    bank = ReIDBank(cosine_threshold=0.85)
    e = _emb(42)
    bank.add(e, 7, "female", fps=25.0, last_seen_frame=100)
    assert bank.query(e) == 7


def test_expired_ignored():
    bank = ReIDBank(cosine_threshold=0.85, expiry_seconds=0.05)
    e = _emb(5)
    bank.add(e, 3, "male", fps=25.0, last_seen_frame=1)
    time.sleep(0.1)
    assert bank.query(e) is None


def test_match_consumed():
    bank = ReIDBank(cosine_threshold=0.85)
    e = _emb(8)
    bank.add(e, 21, "female", fps=25.0, last_seen_frame=1)
    assert bank.query(e) == 21
    assert bank.query(e) is None
