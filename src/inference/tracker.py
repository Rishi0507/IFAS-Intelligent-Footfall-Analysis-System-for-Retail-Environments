"""Greedy IoU tracker.

Detection gives boxes per frame; analytics needs *people*, not boxes. This
associates boxes across frames so one shopper counts once, accumulates their
per-frame classifier votes, and yields a dwell time and a path.

Deliberately simple (no appearance embedding, no Kalman filter): store footage
is short and the classifiers already dominate the runtime cost.

Three additions deal with a detector that is unsure, or loses someone for a
while (behind glass, shelving, another person):

* **Confidence levels.** A confident detection starts a person outright. A
  weak one may extend a person who is already being followed - weak boxes on a
  known person are usually real.
* **Tentative tracks.** A weak box out of nowhere is usually a shelf, but
  sometimes it is a person sitting behind glass. It opens a *tentative* track,
  which counts for nothing until something confirms it: a confident detection
  later on, or the caller looking closer (`IoUTracker.confirm`). Confirmed, it
  keeps its whole history, so the person's stay starts at the first sighting.
  Never confirmed, it is thrown away.
* **Re-joining.** A box that appears where a person was recently lost is that
  person again, not a new visitor. See `IoUTracker._rejoin` for the rules.
"""
from __future__ import annotations
import math
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


def overlap_of_smaller(a, b):
    """Share of the smaller box that lies inside the other one."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (iw * ih) / float(smaller) if smaller > 0 else 0.0


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
    rejoins: int = 0                  # times this person was lost and picked up again
    confirmed: bool = True            # False while the track is only a weak hunch
    checks: int = 0                   # closer looks spent on a tentative track
    last_check_frame: int = -10 ** 9
    conf_max: float = 0.0
    first_box: tuple = None           # where the track began; used when a hunch is confirmed
    last_vote_frame: int = -10 ** 9   # gates re-classification; see needs_vote()
    path: list = field(default_factory=list)              # [(frame, cx, cy, area_frac)]
    # Every observed box at full resolution, so zone occupancy (zones.py) can be
    # recomputed after the run with other zones or settings, without the models.
    box_log: list = field(default_factory=list)           # [(frame, x1, y1, x2, y2, conf)]
    # Every vote with the crop's quality cues, for pooling strategies other than
    # the majority vote (src/research/pooling.py). At most `max_votes` long.
    vote_log: list = field(default_factory=list)          # [(gender_probs, age_probs, cues)]
    pooling: str = "majority"
    quality_model: object = field(default=None, repr=False)

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

    def _pooled(self, which):
        from ..research.pooling import pool
        votes = [(v[which], v[2] or {}) for v in self.vote_log if v[which]]
        return pool(votes, self.pooling, model=self.quality_model) if votes else (None, 0.0)

    def gender(self):
        if self.pooling != "majority":
            return self._pooled(0)
        return self._top(self.gender_votes, self.gender_prob_sum)

    def age(self):
        if self.pooling != "majority":
            return self._pooled(1)
        return self._top(self.age_votes, self.age_score_sum)

    def observe(self, box, frame_idx, det_conf, frame_area):
        self.box = box
        self.last_frame = frame_idx
        self.frames += 1
        self.misses = 0
        self.det_conf_sum += det_conf
        self.conf_max = max(self.conf_max, det_conf)
        cx, cy = self.center
        area = (box[2] - box[0]) * (box[3] - box[1])
        self.path.append((frame_idx, round(cx, 1), round(cy, 1), round(area / frame_area, 5)))
        self.box_log.append((frame_idx, *box, round(det_conf, 3)))

    def needs_vote(self, frame_idx, every, cap, spread=False, max_gap=None):
        """A person's gender and age do not change between frames.

        Classifying every track on every sampled frame is the pipeline's whole
        cost; a bounded number of spread-out votes gives the same label.

        With `spread`, the gap doubles every four votes (up to `max_gap`), so
        the votes cover the person's whole stay instead of its first seconds:
        one awkward angle at the start no longer decides the label for good.
        """
        gap = every
        if spread:
            gap = every * (2 ** (self.n_classified // 4))
            if max_gap:
                gap = min(gap, max(every, max_gap))
        return self.n_classified < cap and frame_idx - self.last_vote_frame >= gap

    def recent(self, which, n=6):
        """(label, confidence) from the last `n` votes only: what the box shows
        now. If the box ever moves to someone else, this follows; the pooled
        label over the whole stay does not."""
        votes = [v[which] for v in self.vote_log[-n:] if v[which]]
        if not votes:
            return None, 0.0
        labels = votes[0].keys()
        mean = {k: sum(v.get(k, 0.0) for v in votes) / len(votes) for k in labels}
        top = max(mean, key=mean.get)
        return top, mean[top]

    def consistent(self, which, min_votes=6, margin=0.25):
        """False when the first and second halves of the votes confidently name
        different labels: the box most likely changed hands between two people,
        and no single label is right for the whole track."""
        votes = [v[which] for v in self.vote_log if v[which]]
        if len(votes) < min_votes:
            return True
        half = len(votes) // 2

        def lead(part):
            mean = {k: sum(v.get(k, 0.0) for v in part) / len(part) for k in part[0]}
            ranked = sorted(mean.items(), key=lambda kv: -kv[1])
            return ranked[0][0], ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)

        (a, ma), (b, mb) = lead(votes[:half]), lead(votes[half:])
        return a == b or ma < margin or mb < margin

    def vote(self, gender_probs=None, age_probs=None, frame_idx=None, cues=None):
        """Accumulate one frame's classifier output. Votes beat any single frame."""
        if gender_probs or age_probs:
            self.vote_log.append((gender_probs, age_probs, cues))
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
    def __init__(self, iou_thresh=0.3, max_misses=30, rejoin_frames=0,
                 new_track_conf=0.0, rejoin_conf=0.0, tentative_conf=None,
                 frame_size=None, edge_margin=0.12, pooling="majority", quality_model=None):
        """
        rejoin_frames   how long (in video frames) a lost person can still be
                        picked up again; 0 turns re-joining off
        new_track_conf  detections at or above this start a person outright
        rejoin_conf     the least a detection needs to re-join a lost person
        tentative_conf  detections at or above this (and below new_track_conf)
                        open a tentative track; None ignores them
        frame_size      (width, height); needed to tell the interior of the
                        frame from its edges, where people really do leave
        pooling         how a person's votes become one label: "majority" (the
                        default), "mean", "confidence" or "quality"; see
                        src/research/pooling.py
        quality_model   a fitted pooling.QualityModel, needed for "quality"
        """
        if pooling == "quality" and quality_model is None:
            raise ValueError("pooling='quality' needs a quality_model")
        self.pooling = pooling
        self.quality_model = quality_model
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.rejoin_frames = rejoin_frames
        self.new_track_conf = new_track_conf
        self.rejoin_conf = rejoin_conf
        self.tentative_conf = tentative_conf
        self.unconfirmed = 0
        self.frame_size = frame_size
        self.edge_margin = edge_margin
        self.next_id = 1
        self.rejoined = 0
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def _interior(self, cx, cy):
        if not self.frame_size:
            return False
        w, h = self.frame_size
        m = self.edge_margin
        return w * m < cx < w * (1 - m) and h * m < cy < h * (1 - m)

    @staticmethod
    def _settled(track):
        """Whether the person was standing or sitting still when last seen."""
        pts = track.path[-8:]
        if len(pts) < 3:
            return False
        height = track.box[3] - track.box[1]
        return math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) < 0.35 * height

    def _rejoin(self, box, frame_idx, in_place_only=False):
        """The lost track this box most plausibly continues, or None.

        A person cannot vanish in the middle of the frame: if a track ends there
        the detector lost them, and a box turning up nearby soon after is the
        same person. At the edges people do leave and others do arrive, so there
        the only thing re-joined is someone who was sitting or standing still
        and reappears in the very same place.

        `in_place_only` is for weak boxes: they may only pick up a confirmed
        person exactly where that person was, never one who has moved.
        """
        if not self.rejoin_frames:
            return None
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        height = box[3] - box[1]
        best, best_score = None, 0.0
        for t in self.active + self.finished:
            gap = frame_idx - t.last_frame
            if gap <= 0 or gap > self.rejoin_frames:      # matched this frame, or lost too long ago
                continue
            if in_place_only and not t.confirmed:
                continue
            tx, ty = t.center
            t_height = t.box[3] - t.box[1]
            both_inside = self._interior(cx, cy) and self._interior(tx, ty)
            overlap = iou(t.box, box)
            if overlap >= self.iou_thresh and (both_inside or self._settled(t)):
                score = 2.0 + overlap                     # back in the same spot
            elif in_place_only:
                continue
            elif both_inside and 0.5 <= height / max(t_height, 1) <= 2.0:
                reach = math.hypot(cx - tx, cy - ty) / max(height, t_height, 1)
                if reach > 1.0:                           # more than a body height away
                    continue
                score = 1.0 - reach
            else:
                continue
            if score > best_score:
                best, best_score = t, score
        return best

    def update(self, detections, frame_idx, frame_area):
        """detections: list of (box, det_conf). Returns list of (track, box, conf)."""
        pairs = []
        for ti, t in enumerate(self.active):
            for di, (box, _) in enumerate(detections):
                score = iou(t.box, box)
                if score >= self.iou_thresh:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d, matched, newly_sure = set(), set(), [], []
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            box, conf = detections[di]
            self.active[ti].observe(box, frame_idx, conf, frame_area)
            if not self.active[ti].confirmed and conf >= self.new_track_conf:
                newly_sure.append(self.active[ti])        # a clear sighting settles it
            matched.append((self.active[ti], box, conf))

        # Strongest first, so a confident box gets first claim on a lost track.
        leftover = sorted((di for di in range(len(detections)) if di not in used_d),
                          key=lambda di: -detections[di][1])
        for di in leftover:
            box, conf = detections[di]
            # A detector sometimes puts two boxes on one person (the body and
            # its reflection in glass, or the body and just the torso). A box
            # lying mostly on someone who is already matched is that person
            # again, and must neither start nor revive a track.
            # (A confident box is only dropped for a confirmed person: a hunch
            # must not be able to hide someone real.)
            weak = conf < self.new_track_conf
            if any(overlap_of_smaller(box, taken) >= 0.5
                   for t, taken, _c in matched if t.confirmed or weak):
                continue
            if conf >= self.rejoin_conf:
                lost = self._rejoin(box, frame_idx)
            elif self.tentative_conf is not None and conf >= self.tentative_conf:
                lost = self._rejoin(box, frame_idx, in_place_only=True)
            else:
                lost = None
            if lost is not None:
                if lost in self.finished:                 # closed already: bring it back
                    self.finished.remove(lost)
                    self.active.append(lost)
                lost.observe(box, frame_idx, conf, frame_area)
                lost.rejoins += 1
                self.rejoined += 1
                if not lost.confirmed and conf >= self.new_track_conf:
                    newly_sure.append(lost)
                matched.append((lost, box, conf))
                continue
            sure = conf >= self.new_track_conf
            if not sure and (self.tentative_conf is None or conf < self.tentative_conf):
                continue                                  # too weak even for a hunch
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            area = (box[2] - box[0]) * (box[3] - box[1])
            t = Track(id=self.next_id, box=box, first_frame=frame_idx, last_frame=frame_idx,
                      det_conf_sum=conf, confirmed=sure, conf_max=conf, first_box=box,
                      pooling=self.pooling, quality_model=self.quality_model,
                      path=[(frame_idx, round(cx, 1), round(cy, 1),
                             round(area / frame_area, 5))],
                      box_log=[(frame_idx, *box, round(conf, 3))])
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
        for t in newly_sure:
            survivor = self.confirm(t)
            if survivor is not t:
                matched = [(survivor if m is t else m, b, c) for m, b, c in matched]
        return matched

    def confirm(self, track):
        """A tentative track turned out to be a person.

        If it began where a known person was recently lost, it is that person
        coming back into view, picked up weakly at first: fold it into their
        track so they are counted once and their stay has no hole in it.
        Returns the track that lives on, which the caller should use from here.
        """
        if track.confirmed:
            return track
        track.confirmed = True
        older = self._rejoin(track.first_box or track.box, track.first_frame)
        if older is None or older is track or not older.confirmed:
            return track
        older.path.extend(track.path)
        older.box_log.extend(track.box_log)
        older.frames += track.frames
        older.det_conf_sum += track.det_conf_sum
        older.conf_max = max(older.conf_max, track.conf_max)
        older.box, older.last_frame, older.misses = track.box, track.last_frame, track.misses
        older.rejoins += 1
        self.rejoined += 1
        if track in self.active:
            self.active.remove(track)
        if older in self.finished:
            self.finished.remove(older)
        if older not in self.active:
            self.active.append(older)
        return older

    def close(self):
        """Every confirmed person, in order of first appearance. Hunches that
        nothing ever confirmed are dropped and counted in `unconfirmed`."""
        self.finished.extend(self.active)
        self.active = []
        people = [t for t in self.finished if t.confirmed]
        self.unconfirmed = len(self.finished) - len(people)
        self.dropped_hunches = [t for t in self.finished if not t.confirmed]
        return sorted(people, key=lambda t: t.id)
