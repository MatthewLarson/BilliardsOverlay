"""Persist/restore Calibration to JSON (matrices stored as nested lists)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Calibration


def save_calibration(calib: Calibration, path: str) -> None:
    data = {
        "H_cam2proj": calib.H_cam2proj.tolist(),
        "H_cam2table": None if calib.H_cam2table is None else calib.H_cam2table.tolist(),
        "camera_size": list(calib.camera_size),
        "projector_size": list(calib.projector_size),
        "table_size_mm": list(calib.table_size_mm),
        "reproj_error_px": float(calib.reproj_error_px),
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str) -> Calibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    h_table = data.get("H_cam2table")
    return Calibration(
        H_cam2proj=np.array(data["H_cam2proj"], dtype=np.float64),
        H_cam2table=None if h_table is None else np.array(h_table, dtype=np.float64),
        camera_size=tuple(data["camera_size"]),
        projector_size=tuple(data["projector_size"]),
        table_size_mm=tuple(data["table_size_mm"]),
        reproj_error_px=float(data.get("reproj_error_px", 0.0)),
    )
