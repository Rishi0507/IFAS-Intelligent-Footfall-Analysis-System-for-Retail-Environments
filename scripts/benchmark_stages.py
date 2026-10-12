"""Time each stage of the pipeline on this machine.

    python scripts/benchmark_stages.py --video clip.mp4 \
        --gender-ckpt models/hf/abhshkp__footfall-analysis-vit-stage1 --age-bundle models/age

Detection is timed per frame. The two classifiers are timed per person crop, on
crops the detector actually produced from the video. Every figure is the median
over repeated calls after a warm-up, so one slow first call does not skew it.
Prints a JSON document; pass --out to also save it.
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _median_ms(fn, repeats):
    fn()                                    # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 2)


def _cpu_name():
    try:                                    # Windows keeps the marketing name in the registry
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        pass
    try:                                    # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--gender-ckpt", default=None)
    p.add_argument("--age-bundle", default=None)
    p.add_argument("--detector", default="yolov8n.pt")
    p.add_argument("--frames", type=int, default=30, help="frames to time detection on")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO

    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // a.frames)
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read any frame from {a.video}")
    h, w = frames[0].shape[:2]

    detector = YOLO(a.detector)
    detect = lambda f: detector(f, classes=[0], conf=0.4, verbose=False, device=a.device)[0]
    detect(frames[0])                       # warm-up
    det_ms, crops = [], []
    for f in frames:
        t0 = time.perf_counter()
        res = detect(f)
        det_ms.append((time.perf_counter() - t0) * 1000.0)
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if x2 - x1 >= 8 and y2 - y1 >= 16:
                crops.append(Image.fromarray(f[y1:y2, x1:x2][:, :, ::-1]))
    if not crops:
        raise SystemExit("The detector found no people in the sampled frames.")

    result = {
        "machine": {"cpu": _cpu_name(), "threads": torch.get_num_threads(),
                    "device": a.device, "torch": torch.__version__,
                    "python": platform.python_version()},
        "video": {"name": Path(a.video).name, "width": w, "height": h,
                  "frames_timed": len(frames), "crops_found": len(crops),
                  "mean_crop_px": [round(sum(c.size[0] for c in crops) / len(crops)),
                                   round(sum(c.size[1] for c in crops) / len(crops))]},
        "detector": {"weights": a.detector, "ms_per_frame": round(statistics.median(det_ms), 2)},
    }

    batch = lambda n: [crops[i % len(crops)] for i in range(n)]
    if a.gender_ckpt:
        from src.inference.models import GenderModel
        model = GenderModel(a.gender_ckpt, device=a.device)
        result["gender"] = {"ms_per_crop_by_batch": {
            str(n): round(_median_ms(lambda n=n: model.predict(batch(n)), a.repeats) / n, 2)
            for n in (1, 4, 16)}}
    if a.age_bundle:
        from src.inference.models import AgeModel, extract_features
        model = AgeModel(a.age_bundle)
        one = batch(1)
        result["age"] = {
            "ms_per_crop": _median_ms(lambda: model.predict(one), a.repeats),
            "ms_features_only": _median_ms(lambda: extract_features(one[0]), a.repeats),
        }

    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
