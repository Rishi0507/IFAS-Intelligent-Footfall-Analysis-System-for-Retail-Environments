"""Greedy IoU tracker.

Detection gives boxes per frame; analytics needs *people*, not boxes. This
associates boxes across frames so one shopper counts once, accumulates their
per-frame classifier votes, and yields a dwell time and a path.

Deliberately simple (no appearance embedding, no Kalman filter): store footage
is short and the classifiers already dominate the runtime cost.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    """One person, followed across frames, with accumulated classifier votes."""
    id: int
    box: tuple
    first_frame: int
    last_frame: int
    frames: int = 1
    misses: int = 0
    det_conf_sum: float = 0.0
    gender_prob_sum: dict = field(default_factory=dict)   # label -> summed prob
    gender_votes: dict = field(default_factory=dict)      # label -> argmax count
    age_score_sum: dict = field(default_factory=dict)     # label -> summed prob
    age_votes: dict = field(default_factory=dict)
    n_classified: int = 0
    last_vote_frame: int = -10 ** 9   # gates re-classification; see needs_vote()
    path: list = field(default_factory=list)              # [(frame, cx, cy, area_frac)]

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _top(self, votes, probs):
        if not votes:
            return None, 0.0
        label = max(votes, key=lambda k: (votes[k], probs.get(k, 0.0)))
        total = sum(probs.values()) or 1.0
        return label, probs.get(label, 0.0) / total

    def gender(self):
        return self._top(self.gender_votes, self.gender_prob_sum)

    def age(self):
        return self._top(self.age_votes, self.age_score_sum)

    def observe(self, box, frame_idx, det_conf, frame_area):
        self.box = box
        self.last_frame = frame_idx
        self.frames += 1
        self.misses = 0
        self.det_conf_sum += det_conf
        cx, cy = self.center
        area = (box[2] - box[0]) * (box[3] - box[1])
        self.path.append((frame_idx, round(cx, 1), round(cy, 1), round(area / frame_area, 5)))

    def needs_vote(self, frame_idx, every, cap):
        """A person's gender and age do not change between frames.

        Classifying every track on every sampled frame is the pipeline's whole
        cost; a bounded number of spread-out votes gives the same label.
        """
        return self.n_classified < cap and frame_idx - self.last_vote_frame >= every

    def vote(self, gender_probs=None, age_probs=None, frame_idx=None):
        """Accumulate one frame's classifier output. Votes beat any single frame."""
        counted = False
        if gender_probs:
            for k, v in gender_probs.items():
                self.gender_prob_sum[k] = self.gender_prob_sum.get(k, 0.0) + v
            top = max(gender_probs, key=gender_probs.get)
            self.gender_votes[top] = self.gender_votes.get(top, 0) + 1
            counted = True
        if age_probs:
            for k, v in age_probs.items():
                self.age_score_sum[k] = self.age_score_sum.get(k, 0.0) + v
            top = max(age_probs, key=age_probs.get)
            self.age_votes[top] = self.age_votes.get(top, 0) + 1
            counted = True
        if counted:
            self.n_classified += 1
            if frame_idx is not None:
                self.last_vote_frame = frame_idx


class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_misses=30):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.next_id = 1
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def update(self, detections, frame_idx, frame_area):
        """detections: list of (box, det_conf). Returns list of (track, box, conf)."""
        pairs = []
        for ti, t in enumerate(self.active):
            for di, (box, _) in enumerate(detections):
                score = iou(t.box, box)
                if score >= self.iou_thresh:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d, matched = set(), set(), []
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            box, conf = detections[di]
            self.active[ti].observe(box, frame_idx, conf, frame_area)
            matched.append((self.active[ti], box, conf))

        for di, (box, conf) in enumerate(detections):
            if di in used_d:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            area = (box[2] - box[0]) * (box[3] - box[1])
            t = Track(id=self.next_id, box=box, first_frame=frame_idx, last_frame=frame_idx,
                      det_conf_sum=conf, path=[(frame_idx, round(cx, 1), round(cy, 1),
                                                round(area / frame_area, 5))])
            self.next_id += 1
            self.active.append(t)
            matched.append((t, box, conf))

        still = []
        for ti, t in enumerate(self.active):
            if ti in used_t or t.last_frame == frame_idx:
                still.append(t)
                continue
            t.misses += 1
            (still if t.misses <= self.max_misses else self.finished).append(t)
        self.active = still
        return matched

    def close(self):
        self.finished.extend(self.active)
        self.active = []
        return sorted(self.finished, key=lambda t: t.id)
