"""Burn a run's zone occupancy into the footage: a shareable video.

Free zones are drawn green and taken zones red, each with a timer; the zone
taken longest pulses blue (the best single guess for the next to free up),
and a panel lists the counts and the next few candidates, under the
recorder's own clock. The same picture as the dashboard's Zones tab, for a
slide or a demo, without the dashboard.

    python scripts/render_zones.py --run outputs/runs/<run> --video <original footage> \
        --out outputs/runs/<run>/zones.mp4 [--start 0 --end 60] [--width 1280] [--boxes]

The footage is the original recording (the run keeps only the annotated
copy). Zones and states come from the run's analytics.json, so redraw zones in
the dashboard first if they need changing. People's faces are not blurred:
only share renders of footage you have permission to show.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FREE = (20, 160, 106)
TAKEN = (224, 64, 90)
NEXT = (31, 143, 224)
QUEUE = (154, 91, 224)
INK = (255, 255, 255)
PANEL = (12, 13, 26)

FONTS = [
    ROOT / "app" / "static" / "fonts" / "Inter.ttf",
    Path("C:/Windows/Fonts/seguisb.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"), Path("C:/Windows/Fonts/arialbd.ttf"),
]


def font(size):
    for f in FONTS:
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def clock(s):
    s = max(0, int(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def status_at(zones, t):
    out = []
    for z in zones:
        cur = next((s for s in z["sessions"] if s["start"] <= t < s["end"] + 1e-6), None)
        prev = next((s for s in reversed(z["sessions"]) if s["end"] <= t), None)
        waiting = [w for w in (z.get("queue") or {}).get("waits", []) if w["start"] <= t < w["end"]]
        out.append({
            "z": z, "taken": cur is not None,
            "since": (t - cur["start"]) if cur else ((t - prev["end"]) if prev else t),
            "lower": cur["censored_start"] if cur else prev is None,
            "waiting": waiting,
        })
    return out


def centroid(poly):
    a = cx = cy = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        c = x1 * y2 - x2 * y1
        a += c
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    if abs(a) < 1e-12:
        return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)
    return cx / (3 * a), cy / (3 * a)


class Renderer:
    def __init__(self, analytics, size, boxes=None, panel=True):
        self.doc = analytics
        self.Z = analytics["zones"]
        self.W, self.H = size
        k = self.W / 1280.0
        self.k = k
        self.f_chip = font(max(11, round(15 * k)))
        self.f_next = font(max(12, round(17 * k)))
        self.f_head = font(max(12, round(16 * k)))
        self.f_big = font(max(16, round(30 * k)))
        self.f_small = font(max(10, round(13 * k)))
        self.boxes = boxes
        self.panel = panel
        start = analytics["meta"].get("captured_at")
        try:
            self.start = dt.datetime.fromisoformat(start) if start else None
        except ValueError:
            self.start = None
        self.seated = [z for z in self.Z["zones"] if z["kind"] in ("seat", "table")]
        self.kind = ("Seats" if all(z["kind"] == "seat" for z in self.seated) else
                     "Tables" if all(z["kind"] == "table" for z in self.seated) else "Zones")

    def px(self, p):
        return (p[0] * self.W, p[1] * self.H)

    def chip(self, d, placed, text, cx, cy, fill, f, optional=False):
        """A rounded label that moves aside rather than cover one already drawn.
        An optional label with nowhere clear to go is left out."""
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        w, h = r - l + 14 * self.k, b - t + 9 * self.k
        tries = [(cx, cy), (cx, cy - h - 3), (cx, cy + h + 3), (cx - 0.6 * w, cy), (cx + 0.6 * w, cy)]
        pick = None
        for x, y in tries:
            x0 = min(max(2, x - w / 2), self.W - w - 2)
            y0 = min(max(2, y - h / 2), self.H - h - 2)
            if not any(x0 < bx + bw + 2 and bx < x0 + w + 2 and y0 < by + bh + 2 and by < y0 + h + 2
                       for bx, by, bw, bh in placed):
                pick = (x0, y0)
                break
        if pick is None and optional:
            return
        if pick is None:
            x0 = min(max(2, cx - w / 2), self.W - w - 2)
            pick = (x0, min(max(2, cy - h / 2), self.H - h - 2))
        x0, y0 = pick
        placed.append((x0, y0, w, h))
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=h / 2, fill=fill + (235,))
        d.text((x0 + 7 * self.k - l, y0 + 4.5 * self.k - t), text, font=f, fill=INK + (255,))

    def frame(self, bgr, t):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        st = status_at(self.Z["zones"], t)
        taken = [s for s in st if s["taken"] and s["z"]["kind"] in ("seat", "table")]
        nxt = max(taken, key=lambda s: s["since"]) if taken else None

        if self.boxes:
            for box in self.boxes.at(t):
                d.rectangle([v * (self.W / self.boxes.w if i % 2 == 0 else self.H / self.boxes.h)
                             for i, v in enumerate(box)], outline=(255, 255, 255, 110), width=max(1, round(self.k)))
        for s in st:
            z = s["z"]
            col = QUEUE if z["kind"] == "queue" else (TAKEN if s["taken"] else FREE)
            pts = [self.px(p) for p in z["polygon"]]
            d.polygon(pts, fill=col + (62 if s["taken"] else 44,))
            d.line(pts + [pts[0]], fill=col + (255,), width=max(2, round(2.4 * self.k)), joint="curve")
        if nxt:
            cx, cy = self.px(centroid(nxt["z"]["polygon"]))
            xs = [self.px(p)[0] for p in nxt["z"]["polygon"]]
            base = max(14 * self.k, min(46 * self.k, (max(xs) - min(xs)) * 0.55))
            for i in range(3):
                f = ((t / 1.8) + i / 3) % 1
                r = base + f * base * 1.6
                d.ellipse((cx - r, cy - r * 0.62, cx + r, cy + r * 0.62),
                          outline=NEXT + (int(230 * (1 - f)),), width=max(2, round(3 * self.k)))
        placed = []
        # A full room cannot carry a name and a timer on every seat: there,
        # labels are the timer alone, except on the next to free up.
        crowded = len(st) > 24
        for s in ([nxt] if nxt else []) + [s for s in st if s is not nxt]:
            z = s["z"]
            cx, cy = self.px(centroid(z["polygon"]))
            is_next = nxt is not None and s is nxt
            name = "" if crowded and not is_next else f"{z['id']} "
            if z["kind"] == "queue":
                text, col = f"{z['id']} · {len(s['waiting'])} waiting", QUEUE
            elif s["taken"]:
                text, col = f"{name}{'≥' if s['lower'] else ''}{clock(s['since'])}", TAKEN
            else:
                text, col = f"{name}Free", FREE
            self.chip(d, placed, text, cx, cy, NEXT if is_next else col,
                      self.f_next if is_next else (self.f_small if crowded else self.f_chip),
                      optional=crowded and not is_next)

        if self.panel == "overlay":
            self.draw_panel(d, st, taken, nxt, t)
        img = Image.alpha_composite(img, lay)
        if self.panel == "side":
            img = self.side_panel(img, st, taken, nxt, t)
        return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    SIDE = 330

    @classmethod
    def side_width(cls, h):
        return int(round(cls.SIDE * h / 720.0 / 2)) * 2      # even: encoders need it

    def side_panel(self, img, st, taken, nxt, t):
        """The status column beside the frame: nothing in the picture is covered."""
        k = self.H / 720.0
        pw = self.side_width(self.H)
        out = Image.new("RGBA", (self.W + pw, self.H), PANEL + (255,))
        out.paste(img, (0, 0))
        d = ImageDraw.Draw(out)
        x = self.W + 22 * k
        f_s, f_h, f_b = font(round(13 * k)), font(round(17 * k)), font(round(44 * k))
        when = (self.start + dt.timedelta(seconds=t)).strftime("%H:%M:%S") if self.start else clock(t)
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        d.ellipse((x, 26 * k, x + 10 * k, 36 * k), fill=TAKEN + (int(140 + 115 * pulse),))
        d.text((x + 18 * k, 21 * k), f"REC  {when}", font=f_h, fill=INK + (255,))
        if self.start:
            d.text((x, 46 * k), self.start.strftime("%a %d %b %Y"), font=f_s, fill=(150, 150, 175, 255))
        seated = [s for s in st if s["z"]["kind"] in ("seat", "table")]
        free = sorted([s for s in seated if not s["taken"]], key=lambda s: -s["since"])
        ids = [z["id"] for z in self.seated]
        y = 84 * k
        d.text((x, y), f"{self.kind.upper()}  {ids[0]}–{ids[-1]}" if ids else self.kind.upper(),
               font=f_s, fill=(150, 150, 175, 255))
        d.text((x, y + 16 * k), str(len(free)), font=f_b, fill=FREE + (255,))
        fw = d.textlength(str(len(free)), font=f_b)
        d.text((x + fw + 10 * k, y + 22 * k), "free", font=f_h, fill=FREE + (255,))
        d.text((x + fw + 10 * k, y + 42 * k), f"{len(seated) - len(free)} taken", font=f_s, fill=TAKEN + (255,))
        # occupancy bar
        y += 80 * k
        bw = pw - 44 * k
        n = max(1, len(seated))
        d.rounded_rectangle((x, y, x + bw, y + 8 * k), radius=4 * k, fill=(40, 42, 64, 255))
        d.rounded_rectangle((x, y, x + bw * (n - len(free)) / n, y + 8 * k), radius=4 * k, fill=TAKEN + (255,))
        y += 30 * k
        d.text((x, y), "NEXT TO FREE UP", font=f_s, fill=NEXT + (255,))
        y += 20 * k
        if nxt:
            cx, cy = x + 26 * k, y + 26 * k
            for i in range(3):
                f = ((t / 1.8) + i / 3) % 1
                r = (10 + f * 18) * k
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=NEXT + (int(230 * (1 - f)),), width=max(2, round(2.5 * k)))
            d.text((x + 62 * k, y + 2 * k), nxt["z"]["id"], font=font(round(30 * k)), fill=NEXT + (255,))
            d.text((x + 62 * k, y + 38 * k), f"taken {'≥' if nxt['lower'] else ''}{clock(nxt['since'])}, the longest",
                   font=f_s, fill=(200, 200, 220, 255))
            y += 66 * k
            for i, s in enumerate(sorted(taken, key=lambda s: -s["since"])[1:5]):
                d.text((x, y), f"{i + 2}", font=f_s, fill=NEXT + (255,))
                d.text((x + 22 * k, y), s["z"]["id"], font=f_s, fill=INK + (255,))
                d.text((x + 80 * k, y), f"{'≥' if s['lower'] else ''}{clock(s['since'])}", font=f_s,
                       fill=(200, 200, 220, 255))
                y += 21 * k
        else:
            d.text((x, y), "Nothing is taken.", font=f_s, fill=(200, 200, 220, 255))
            y += 24 * k
        y += 14 * k
        d.text((x, y), f"FREE NOW ({len(free)})", font=f_s, fill=FREE + (255,))
        y += 22 * k
        cx = x
        for s in free[:18]:
            label = f"{s['z']['id']} {'≥' if s['lower'] else ''}{clock(s['since'])}"
            w = d.textlength(label, font=f_s) + 14 * k
            if cx + w > self.W + pw - 18 * k:
                cx = x
                y += 26 * k
            if y > self.H - 60 * k:
                break
            d.rounded_rectangle((cx, y, cx + w, y + 21 * k), radius=10 * k, fill=FREE + (60,), outline=FREE + (200,))
            d.text((cx + 7 * k, y + 3 * k), label, font=f_s, fill=INK + (255,))
            cx += w + 6 * k
        d.text((x, self.H - 32 * k), "IFAS · zone occupancy", font=f_s, fill=(110, 110, 140, 255))
        return out

    def draw_panel(self, d, st, taken, nxt, t):
        k = self.k
        x0, y0, w = 18 * k, 18 * k, 250 * k
        seated = [s for s in st if s["z"]["kind"] in ("seat", "table")]
        free = [s for s in seated if not s["taken"]]
        runners = sorted(taken, key=lambda s: -s["since"])[1:4]
        h = (118 + 22 * len(runners)) * k
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=14 * k, fill=PANEL + (190,))
        ids = [z["id"] for z in self.seated]
        d.text((x0 + 14 * k, y0 + 11 * k), f"{self.kind.upper()}  {ids[0]}–{ids[-1]}" if ids else self.kind.upper(),
               font=self.f_small, fill=(190, 190, 210, 255))
        d.text((x0 + 14 * k, y0 + 30 * k), f"{len(free)}", font=self.f_big, fill=FREE + (255,))
        fw = d.textlength(f"{len(free)}", font=self.f_big)
        d.text((x0 + 20 * k + fw, y0 + 42 * k), f"free  ·  {len(seated) - len(free)} taken",
               font=self.f_head, fill=INK + (235,))
        if nxt:
            d.text((x0 + 14 * k, y0 + 80 * k), "NEXT TO FREE UP", font=self.f_small, fill=NEXT + (255,))
            d.text((x0 + 14 * k, y0 + 96 * k), f"{nxt['z']['id']}   taken {'≥' if nxt['lower'] else ''}{clock(nxt['since'])}",
                   font=self.f_head, fill=INK + (255,))
            for i, s in enumerate(runners):
                d.text((x0 + 14 * k, y0 + (120 + 22 * i) * k),
                       f"{i + 2}.  {s['z']['id']}   {'≥' if s['lower'] else ''}{clock(s['since'])}",
                       font=self.f_small, fill=(200, 200, 220, 255))
        # Recorder clock, top right, like a live feed.
        when = (self.start + dt.timedelta(seconds=t)).strftime("%H:%M:%S") if self.start else clock(t)
        tw = d.textlength(when, font=self.f_head)
        bx1 = self.W - 18 * k
        bx0 = bx1 - tw - 46 * k
        d.rounded_rectangle((bx0, 18 * k, bx1, 52 * k), radius=17 * k, fill=PANEL + (190,))
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        d.ellipse((bx0 + 13 * k, 30 * k, bx0 + 23 * k, 40 * k), fill=TAKEN + (int(140 + 115 * pulse),))
        d.text((bx0 + 31 * k, 24 * k), when, font=self.f_head, fill=INK + (255,))


class TrackBoxes:
    """Person boxes at time t, from track_boxes.json (nearest sample)."""

    def __init__(self, path):
        tb = json.loads(Path(path).read_text())
        self.w, self.h = tb["width"], tb["height"]
        self.fps, self.stride = tb["fps"], tb["stride"]
        self.by_frame = {}
        for obs in tb["tracks"].values():
            for f, x1, y1, x2, y2, _c in obs:
                self.by_frame.setdefault(f, []).append((x1, y1, x2, y2))

    def at(self, t):
        f = int(round(t * self.fps / self.stride)) * self.stride
        return self.by_frame.get(f, [])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--video", required=True, help="the original footage the run was made from")
    p.add_argument("--out", default=None, help="default: <run>/zones.mp4")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--boxes", action="store_true", help="also outline every tracked person")
    p.add_argument("--panel", choices=["side", "overlay", "none"], default="side",
                   help="the status panel: a column beside the frame (default), over it, or none")
    a = p.parse_args()

    run = Path(a.run)
    doc = json.loads((run / "analytics.json").read_text(encoding="utf-8"))
    if not doc.get("zones"):
        raise SystemExit("This run has no zones. Draw them in the dashboard's Zones tab first.")
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W0, H0) != (doc["meta"]["width"], doc["meta"]["height"]):
        raise SystemExit("That video is not the one this run was made from (different frame size).")
    W = min(a.width, W0) // 2 * 2
    H = int(round(H0 * W / W0)) // 2 * 2
    out = Path(a.out) if a.out else run / "zones.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    boxes = TrackBoxes(run / "track_boxes.json") if a.boxes and (run / "track_boxes.json").exists() else None
    r = Renderer(doc, (W, H), boxes=boxes, panel=a.panel)
    out_w = W + (Renderer.side_width(H) if a.panel == "side" else 0)
    writer = open_writer(out, fps, (out_w, H))
    start_f = int(a.start * fps)
    end_f = int(a.end * fps) if a.end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        writer.write(r.frame(frame, f / fps))
        f += 1
        if (f - start_f) % 250 == 0:
            print(f"  {100 * (f - start_f) / max(1, end_f - start_f):5.1f}%", end="\r", flush=True)
    writer.release()
    cap.release()
    print(f"\nwritten {out}")


class open_writer:
    """Frames straight into ffmpeg's H.264 encoder when ffmpeg is on PATH (what
    browsers and phones play), else OpenCV's MPEG-4 writer. Either way it fails
    here, before any frame is rendered, if the file cannot be written."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        ffmpeg = shutil.which("ffmpeg")
        self.proc = self.cv = None
        if ffmpeg:
            self.tmp = self.path.with_suffix(".part.mp4")
            self.proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(self.tmp)], stdin=subprocess.PIPE)
        else:
            self.tmp = self.path
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.cv.isOpened():
                raise SystemExit(f"Could not open {self.path} for writing")

    def write(self, frame):
        if self.cv is not None:
            self.cv.write(frame)
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise SystemExit(f"ffmpeg stopped writing {self.tmp}") from None

    def release(self):
        if self.cv is not None:
            self.cv.release()
            return
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(f"ffmpeg could not write {self.tmp}")
        self.tmp.replace(self.path)


if __name__ == "__main__":
    main()
