"""Apply V4L2 camera controls on Linux via `v4l2-ctl`.

OpenCV's cap.set() mapping for these controls is inconsistent across drivers, so
on Linux we set them with `v4l2-ctl --set-ctrl` against the /dev/videoN device
(reliable, and what the auto-tuner uses). No-ops on non-Linux.

Controls are a plain dict of V4L2 control name -> value, e.g.:
    {"auto_exposure": 1, "exposure_time_absolute": 300, "gain": 40,
     "brightness": 8, "contrast": 40, "saturation": 80, "gamma": 110}
"""
from __future__ import annotations

import subprocess
import sys


def device_path(index: int) -> str:
    return f"/dev/video{int(index)}"


def available() -> bool:
    return sys.platform.startswith("linux")


def set_controls(device: str, controls: dict) -> bool:
    """Set the given controls. Returns True on success (best-effort)."""
    if not controls or not available():
        return False
    pairs = ",".join(f"{k}={int(v)}" for k, v in controls.items())
    try:
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", pairs],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def list_control_ranges(device: str) -> dict[str, dict]:
    """Parse `v4l2-ctl --list-ctrls` into {name: {min, max, step, default, value}}."""
    out: dict[str, dict] = {}
    if not available():
        return out
    try:
        res = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"],
                             capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return out
    for line in res.stdout.splitlines():
        line = line.strip()
        if "0x" not in line or ":" not in line:
            continue
        name = line.split()[0]
        info: dict[str, int] = {}
        for tok in line.split(":", 1)[1].split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                try:
                    info[k] = int(v)
                except ValueError:
                    pass
        if {"min", "max"} <= set(info):
            out[name] = info
    return out
