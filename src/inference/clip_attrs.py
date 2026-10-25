"""Gender and age from CLIP, with no training on any shop's footage.

CLIP (`openai/clip-vit-base-patch32`) learned from hundreds of millions of web
photos, which show people from every angle: seated, from behind, from above.
Asked "is this a CCTV image of a man or of a woman?", it answers from what it
learned there (zero-shot). Each question is a linear head on the image
embedding whose weights are the embeddings of a few text prompts, so the same
machinery also runs heads fitted on data (`scripts/research/clip_probe.py`).

What the studies found (docs/research-plan.md, Section 10):

* Gender: the fine-tuned ViT is best on PETA, CLIP zero-shot is best on the
  classroom recording, and their average beats both on both. `Average` is
  that average, and the pipeline's default.
* Age: CLIP zero-shot with coarse young / adult / middle-aged / old prompts
  beats the colour/texture SVMs on PETA and on the classroom. It is still a
  rough guess from CCTV, and the pipeline says "not sure" when it is weak.
* Heads fitted on PETA's training split do worse than zero-shot on the
  classroom: PETA's small, standing pedestrians do not look like CCTV of
  seated people.

`ClipAttributes` scores a list of crops once and serves both questions, with
the same `predict(crops) -> [{label: prob}]` interface as `GenderModel` and
`AgeModel`, through its `.gender` and `.age` views.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLIP_NAME = "openai/clip-vit-base-patch32"
HEADS = Path(__file__).with_name("assets") / "clip_heads.json"
LOGIT_SCALE = 100.0                      # CLIP's own temperature

# Several phrasings per answer, averaged: one wording is a coin toss, a few
# are a consensus. Chosen on PETA test identities and the classroom labels
# together (scripts/research/attribute_study.py), not on either alone.
GENDER_PROMPTS = {
    "female": ["a CCTV image of a woman", "a surveillance camera image of a woman",
               "a low resolution photo of a woman"],
    "male": ["a CCTV image of a man", "a surveillance camera image of a man",
             "a low resolution photo of a man"],
}
AGE_PROMPTS = {
    "Age16-30": ["a photo of a young person", "a photo of a student"],
    "Age31-45": ["a photo of an adult"],
    "Age46-60": ["a photo of a middle-aged person"],
    "AgeAbove61": ["a photo of an old person"],
}


def _features(out):
    """transformers 5 returns a model output; 4.x returns the tensor itself."""
    return getattr(out, "pooler_output", out)


class ClipEncoder:
    """Unit-length CLIP embeddings for PIL crops and for text prompts."""

    def __init__(self, name=CLIP_NAME, device="cpu", batch_size=32):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.device = device
        self.batch_size = batch_size
        self.name = name

    def embed(self, crops):
        out = []
        for i in range(0, len(crops), self.batch_size):
            chunk = [c.convert("RGB") for c in crops[i:i + self.batch_size]]
            px = self.processor(images=chunk, return_tensors="pt")["pixel_values"].to(self.device)
            with self._torch.no_grad():
                f = _features(self.model.get_image_features(pixel_values=px))
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 512), np.float32)

    def embed_text(self, prompts):
        batch = self.processor(text=list(prompts), return_tensors="pt", padding=True)
        with self._torch.no_grad():
            f = _features(self.model.get_text_features(**{k: v.to(self.device) for k, v in batch.items()}))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)


class LinearHead:
    """softmax(x W^T + b) over labels; optional standardisation of x first."""

    def __init__(self, labels, weights, bias, mean=None, scale=None):
        self.labels = list(labels)
        self.W = np.asarray(weights, dtype=np.float32)
        self.b = np.asarray(bias, dtype=np.float32)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)

    @classmethod
    def zero_shot(cls, encoder, prompts):
        """A head whose rows are the mean prompt embedding per label."""
        rows = []
        for label in prompts:
            t = encoder.embed_text(prompts[label]).mean(axis=0)
            rows.append(t / np.linalg.norm(t))
        return cls(list(prompts), LOGIT_SCALE * np.stack(rows), np.zeros(len(rows)))

    @classmethod
    def from_dict(cls, d):
        return cls(d["labels"], d["weights"], d["bias"], d.get("mean"), d.get("scale"))

    def to_dict(self, **extra):
        d = {"labels": self.labels, "weights": np.round(self.W, 6).tolist(), "bias": np.round(self.b, 6).tolist()}
        if self.mean is not None:
            d["mean"] = np.round(self.mean, 6).tolist()
            d["scale"] = np.round(self.scale, 6).tolist()
        d.update(extra)
        return d

    def proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mean is not None:
            X = (X - self.mean) / self.scale
        z = X @ self.W.T + self.b
        if z.shape[1] == 1:                                  # fitted binary head: one logit
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            return np.column_stack([1.0 - p1, p1])
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return [{k: float(v) for k, v in zip(self.labels, row)} for row in self.proba(X)]


class _View:
    """One question (gender or age) answered from the shared embeddings."""

    def __init__(self, parent, task):
        self.parent = parent
        self.task = task
        self.source = f"{parent.encoder.name}, {parent.kind} {task}"

    def predict(self, crops):
        return self.parent.predict(crops)[self.task]

    def embed(self, crops):
        return self.parent.embeddings(crops)


class ClipAttributes:
    """Both questions from one CLIP pass. `heads` is "zero-shot" (prompts in
    this file) or the path of a heads file fitted by clip_probe.py."""

    def __init__(self, heads="zero-shot", device="cpu", encoder=None):
        self.encoder = encoder or ClipEncoder(CLIP_NAME, device=device)
        if heads == "zero-shot":
            self.kind = "zero-shot"
            self.meta = {"trained_on": "nothing (zero-shot prompts)",
                         "prompts": {"gender": GENDER_PROMPTS, "age": AGE_PROMPTS}}
            self.heads = {"gender": LinearHead.zero_shot(self.encoder, GENDER_PROMPTS),
                          "age": LinearHead.zero_shot(self.encoder, AGE_PROMPTS)}
        else:
            doc = json.loads(Path(heads).read_text(encoding="utf-8"))
            self.kind = "fitted"
            self.meta = doc.get("meta", {})
            self.heads = {k: LinearHead.from_dict(v) for k, v in doc["heads"].items()}
        self._last = (None, None)
        self.gender = _View(self, "gender")
        self.age = _View(self, "age") if "age" in self.heads else None

    def embeddings(self, crops):
        # The pipeline asks for gender and then age on the same list of crops:
        # score them once.
        key = tuple(id(c) for c in crops)
        if self._last[0] != key:
            self._last = (key, self.encoder.embed(crops))
        return self._last[1]

    def predict(self, crops):
        X = self.embeddings(crops)
        return {task: head.predict(X) for task, head in self.heads.items()}


class Average:
    """The mean of several models' probabilities for the same labels."""

    def __init__(self, *models, weights=None):
        self.models = [m for m in models if m is not None]
        self.weights = weights or [1.0] * len(self.models)
        self.source = " + ".join(str(getattr(m, "source", type(m).__name__)) for m in self.models)

    def predict(self, crops):
        outs = [m.predict(crops) for m in self.models]
        total = sum(self.weights)
        merged = []
        for rows in zip(*outs):
            labels = rows[0].keys()
            merged.append({k: sum(w * r.get(k, 0.0) for w, r in zip(self.weights, rows)) / total for k in labels})
        return merged


def load_clip_attributes(heads="zero-shot", device="cpu", say=None):
    if heads != "zero-shot" and not Path(heads).exists():
        raise FileNotFoundError(f"{heads} is missing: run scripts/research/clip_probe.py to fit heads.")
    if say:
        say(stage="loading CLIP")
    return ClipAttributes(heads, device=device)
