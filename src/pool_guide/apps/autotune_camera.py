"""Auto-tune the webcam's image controls to best separate balls from the cloth.

Runs on the machine with the webcam (the sensor). Owns the camera, so stop the
sensor node first (the web panel does this automatically when it runs autotune).

It fixes exposure to manual, then does coordinate-ascent over
exposure/gain/brightness/contrast/saturation/gamma, scoring each setting by how
well balls separate from the army-green felt (see vision/tuning.py). The best
controls are written to capture.controls in config.yaml, and WebcamSource applies
them from then on.

    python -m pool_guide.apps.autotune_camera --config config.yaml
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..capture import camera_controls
from ..config import load_config, write_config
from ..vision.tuning import score_separation

TUNE_ORDER = ["exposure_time_absolute", "gain", "brightness",
              "contrast", "saturation", "gamma"]


def _open(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def _grab(cap, flush=5):
    for _ in range(flush):          # drop buffered frames so the new control took effect
        cap.read()
    ok, frame = cap.read()
    return frame if ok else None


def _evaluate(cap, vision_cfg, avg=2):
    scores = []
    info = None
    for _ in range(avg):
        f = _grab(cap, flush=3)
        if f is None:
            continue
        info = score_separation(f, vision_cfg)
        scores.append(info["score"])
    if not scores:
        return {"score": -1e6, "balls": 0}
    info = dict(info)
    info["score"] = float(np.mean(scores))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Auto-tune camera for ball detection")
    ap.add_argument("--config", default=None)
    ap.add_argument("--samples", type=int, default=6, help="values tried per control")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--save-images", default=None, help="dir to save before/after frames")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    idx = cfg.capture.webcam_index
    device = camera_controls.device_path(idx)
    if not camera_controls.available():
        print("Camera control tuning only works on Linux (v4l2-ctl). Aborting.")
        return 1

    cap = _open(idx)
    if not cap.isOpened():
        print(f"Could not open camera {idx}. Is the sensor node still using it?")
        return 2

    # Stable base: manual exposure + fixed white balance so scoring is repeatable.
    camera_controls.set_controls(device, {"auto_exposure": 1, "white_balance_automatic": 1})
    time.sleep(0.3)
    ranges = camera_controls.list_control_ranges(device)
    if not ranges:
        print("No V4L2 controls found; cannot tune.")
        return 3

    base = _evaluate(cap, cfg.vision)
    print(f"baseline: score={base['score']:.1f} balls={base.get('balls')} "
          f"felt_v={base.get('felt_v', 0):.0f} clip={base.get('clip', 0):.2f}")
    if args.save_images:
        f = _grab(cap)
        if f is not None:
            cv2.imwrite(f"{args.save_images}/before.jpg", f)

    controls: dict[str, int] = {"auto_exposure": 1}
    for p in range(args.passes):
        for name in TUNE_ORDER:
            if name not in ranges:
                continue
            lo, hi = ranges[name]["min"], ranges[name]["max"]
            vals = sorted(set(int(v) for v in np.linspace(lo, hi, args.samples)))
            best_v = controls.get(name, ranges[name].get("default", (lo + hi) // 2))
            best_s = None
            for v in vals:
                camera_controls.set_controls(device, {**controls, name: v})
                r = _evaluate(cap, cfg.vision)
                if best_s is None or r["score"] > best_s:
                    best_s, best_v = r["score"], v
            controls[name] = int(best_v)
            camera_controls.set_controls(device, controls)
            print(f"  pass {p + 1} {name:24s} -> {best_v:5d}  (score {best_s:.1f})")

    final = _evaluate(cap, cfg.vision)
    print(f"tuned:    score={final['score']:.1f} balls={final.get('balls')} "
          f"felt_v={final.get('felt_v', 0):.0f} clip={final.get('clip', 0):.2f}")
    print(f"controls: {controls}")
    if args.save_images:
        f = _grab(cap)
        if f is not None:
            cv2.imwrite(f"{args.save_images}/after.jpg", f)
    cap.release()

    if final["score"] >= base["score"]:
        cfg.capture.controls = controls
        path = args.config or "config.yaml"
        write_config(cfg, path)
        print(f"Saved tuned controls to {path}")
    else:
        print("Tuning did not improve the score; leaving config unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
