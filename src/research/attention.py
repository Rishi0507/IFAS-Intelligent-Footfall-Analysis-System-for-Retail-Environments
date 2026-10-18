"""Learned temporal attention over a track's frames: the RQ3 baseline (task A5).

Video attribute papers pool a track with a small attention head trained on
video labels (Chen et al., 2019). This is that baseline, kept small enough to
train on a CPU. Each frame's frozen feature f_t (the ViT feature the gender
head reads, or the histograms the age SVMs score) is standardised and
projected, z_t = W f_t + b. A frame's score is s_t = v . tanh(z_t), the
weights a = softmax(s) average the z_t, and a linear layer classifies the
average.

It needs person-level labels to train. Quality weighting does not, and that
trade-off is what the paper reports. Train it on MARS when the images are
available, or else with grouped cross-validation over the labelled retail
people, the same folds the quality model uses.
"""
from __future__ import annotations

import json

import numpy as np


class AttentionPooler:
    def __init__(self, classes=None, hidden=32, l2=1e-3, seed=0, meta=None):
        self.classes = list(classes) if classes is not None else None
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.meta = meta or {}
        self.mean = self.std = None
        self.params = None                # numpy arrays: W, b, v, U, c

    # ----------------------------------------------------------------- torch
    def _net(self, dim):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, d, h, k):
                super().__init__()
                self.proj = nn.Linear(d, h)
                self.score = nn.Linear(h, 1, bias=False)
                self.out = nn.Linear(h, k)

            def forward(self, x, mask):
                z = self.proj(x)                                  # (n, T, h)
                s = self.score(torch.tanh(z)).squeeze(-1)         # (n, T)
                s = s.masked_fill(~mask, float("-inf"))
                a = torch.softmax(s, dim=1)
                return self.out((a.unsqueeze(-1) * z).sum(1)), a

        torch.manual_seed(self.seed)
        return Net(dim, self.hidden, len(self.classes))

    def _batch(self, tracks):
        import torch
        T = max(len(t) for t in tracks)
        d = tracks[0].shape[1]
        x = np.zeros((len(tracks), T, d), np.float32)
        m = np.zeros((len(tracks), T), bool)
        for i, t in enumerate(tracks):
            x[i, :len(t)] = (t - self.mean) / self.std
            m[i, :len(t)] = True
        return torch.from_numpy(x), torch.from_numpy(m)

    # ------------------------------------------------------------------- fit
    def fit(self, tracks, labels, epochs=300, lr=0.01):
        """tracks: list of (n_frames, dim) arrays, one per person; labels: one
        class per person. Full-batch Adam on class-balanced cross-entropy, so a
        rare class is not ignored (the paper reports balanced accuracy)."""
        import torch
        import torch.nn.functional as F

        tracks = [np.asarray(t, np.float32) for t in tracks]
        keep = [i for i, t in enumerate(tracks) if len(t)]
        tracks = [tracks[i] for i in keep]
        labels = [labels[i] for i in keep]
        if not tracks:
            raise ValueError("no tracks with frames to train on")
        if self.classes is None:
            self.classes = sorted(set(labels))
        frames = np.vstack(tracks)
        self.mean = frames.mean(0)
        std = frames.std(0)
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float32)

        net = self._net(frames.shape[1])
        x, m = self._batch(tracks)
        y = torch.tensor([self.classes.index(c) for c in labels])
        counts = np.bincount(y.numpy(), minlength=len(self.classes)).astype(np.float32)
        w = torch.from_numpy(np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=self.l2)
        for _ in range(epochs):
            opt.zero_grad()
            logits, _ = net(x, m)
            loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            opt.step()
        self.params = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        self.meta.update({"n_people": len(tracks), "n_frames": int(len(frames)),
                          "final_loss": round(float(loss.detach()), 5)})
        return self

    # --------------------------------------------------------------- predict
    def _project(self, track):
        x = (np.asarray(track, np.float32) - self.mean) / self.std
        return x @ self.params["proj.weight"].T + self.params["proj.bias"]

    def predict(self, track):
        """(label, confidence, frame weights) for one person's (n, dim) features."""
        if track is None or len(track) == 0:
            return None, 0.0, np.zeros(0)
        z = self._project(track)
        s = np.tanh(z) @ self.params["score.weight"][0]
        a = np.exp(s - s.max())
        a /= a.sum()
        logits = (a @ z) @ self.params["out.weight"].T + self.params["out.bias"]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(np.argmax(p))
        return self.classes[k], float(p[k]), a

    # ------------------------------------------------------------ save/load
    def to_dict(self):
        return {"kind": "attention", "classes": self.classes, "hidden": self.hidden,
                "l2": self.l2, "seed": self.seed, "meta": self.meta,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "params": {k: v.tolist() for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d):
        if d.get("kind") != "attention":
            raise ValueError("not an attention pooler")
        m = cls(d["classes"], d["hidden"], d["l2"], d["seed"], d.get("meta"))
        m.mean = np.asarray(d["mean"], np.float32)
        m.std = np.asarray(d["std"], np.float32)
        m.params = {k: np.asarray(v, np.float32) for k, v in d["params"].items()}
        return m

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ------------------------------------------------------------ feature files
def save_features(path, items, vectors, **extra):
    """One .npz per run and task: item ids ("run/track/frame") and their vectors.
    `extra` arrays (labels, groups) ride along, e.g. for a MARS training file."""
    np.savez_compressed(path, items=np.asarray(items), vectors=np.asarray(vectors, np.float32),
                        **{k: np.asarray(v) for k, v in extra.items()})


def load_features(path):
    """{item: vector} plus any extra arrays stored with them."""
    with np.load(path, allow_pickle=False) as z:
        lookup = dict(zip(z["items"].tolist(), z["vectors"]))
        extra = {k: z[k] for k in z.files if k not in ("items", "vectors")}
    return lookup, extra


def tracks_from_file(path):
    """A labelled training file (items, vectors, groups, labels) as per-person
    (tracks, labels): the input `AttentionPooler.fit` takes."""
    with np.load(path, allow_pickle=False) as z:
        if "groups" not in z.files or "labels" not in z.files:
            raise ValueError(f"{path} has no groups and labels to train on")
        groups, labels, vectors = z["groups"], z["labels"], z["vectors"]
    tracks, out = [], []
    for g in dict.fromkeys(groups.tolist()):
        idx = np.flatnonzero(groups == g)
        lab = labels[idx[0]]
        if lab in ("", "None"):
            continue
        tracks.append(vectors[idx])
        out.append(str(lab))
    return tracks, out
