"""CLIP attribute heads and the ensemble (src/inference/clip_attrs.py), without
downloading CLIP: a fake encoder stands in for it."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import clip_attrs as CA  # noqa: E402
from tests.test_pipeline_e2e import StubDetector, StubGender, video  # noqa: E402,F401  (fixture)


class FakeEncoder:
    """Text 'woman' -> e0, 'man' -> e1, anything else -> e2; an image's
    embedding is e0 if it is mostly red, e1 if mostly blue."""
    name = "fake-clip"

    def __init__(self):
        self.image_calls = 0

    @staticmethod
    def _unit(i):
        v = np.zeros(8, np.float32)
        v[i] = 1.0
        return v

    def embed_text(self, prompts):
        out = []
        for p in prompts:
            if "woman" in p:
                out.append(self._unit(0))
            elif "man" in p:
                out.append(self._unit(1))
            else:
                out.append(self._unit(2 + len(out) % 4))
        return np.stack(out)

    def embed(self, crops):
        self.image_calls += 1
        out = []
        for c in crops:
            r, _g, b = np.asarray(c.convert("RGB"), dtype=float).reshape(-1, 3).mean(0)
            out.append(self._unit(0) if r > b else self._unit(1))
        return np.stack(out)


def crop(color):
    return Image.new("RGB", (20, 40), color)


def test_zero_shot_head_points_at_the_prompt_embeddings():
    head = CA.LinearHead.zero_shot(FakeEncoder(), {"female": ["a photo of a woman"], "male": ["a photo of a man"]})
    p = head.predict(np.stack([FakeEncoder._unit(0), FakeEncoder._unit(1)]))
    assert p[0]["female"] > 0.99 and p[1]["male"] > 0.99


def test_fitted_binary_head_reads_one_logit_for_the_second_label():
    head = CA.LinearHead(["female", "male"], [[2.0, 0.0]], [0.0])
    p = head.predict(np.array([[3.0, 0.0], [-3.0, 0.0]]))
    assert p[0]["male"] > 0.99 and p[1]["female"] > 0.99
    back = CA.LinearHead.from_dict(head.to_dict())
    assert np.allclose(back.proba([[3.0, 0.0]]), head.proba([[3.0, 0.0]]))


def test_both_questions_share_one_pass_and_the_average_merges_models():
    enc = FakeEncoder()
    both = CA.ClipAttributes("zero-shot", encoder=enc)
    crops = [crop("red"), crop("blue")]
    g = both.gender.predict(crops)
    a = both.age.predict(crops)
    assert enc.image_calls == 1                                  # gender then age: one CLIP pass
    assert g[0]["female"] > 0.9 and g[1]["male"] > 0.9
    assert set(a[0]) == {"Age16-30", "Age31-45", "Age46-60", "AgeAbove61"}

    class Sure:
        source = "vit"

        def predict(self, crops):
            return [{"female": 0.0, "male": 1.0} for _ in crops]

    avg = CA.Average(Sure(), both.gender).predict(crops)
    assert 0.4 < avg[0]["male"] < 0.6                           # the two disagree on the red crop
    assert avg[1]["male"] > 0.95


def test_pipeline_ensemble_mode_averages_the_gender_models(monkeypatch, tmp_path, video):
    monkeypatch.setattr(CA, "load_clip_attributes",
                        lambda heads="zero-shot", device="cpu", say=None: CA.ClipAttributes("zero-shot", encoder=FakeEncoder()))
    from src.inference.pipeline import process_video
    doc = process_video(video, tmp_path / "run", stride=1, write_video=False, device="cpu",
                        detector=StubDetector(), gender_model=StubGender(), attributes="ensemble",
                        min_gender_conf=0.0, min_age_conf=0.0)
    assert doc["config"]["attributes"] == "ensemble"
    assert doc["model"]["gender"]["family"].startswith("fine-tuned ViT-B/16 averaged")
    assert all(t["age"] for t in doc["tracks"])                  # age now comes from CLIP
