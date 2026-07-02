"""The calibration data model and the three coordinate frames it relates.

Three coordinate spaces matter:

  camera   pixels in the captured frame (what the vision code sees)
  projector pixels in the projected image (what we draw into)
  table    real-world millimetres on the playing surface (what physics uses)

Two homographies tie them together:

  H_cam2proj  : camera px -> projector px
      Draw an aim line at a ball's camera pixel by mapping it into projector
      space so the light lands back on that ball. This is THE calibration --
      recovered by projecting ArUco markers and seeing where the camera finds
      them (we know both the projector px we drew them at and the camera px we
      detected them at, so we can solve for H).

  H_cam2table : camera px -> table mm
      Lets vision hand real-world positions to the physics engine. Recovered
      from the four table corners (clicked or auto-detected) mapped to a
      rectangle of the known table dimensions.

A homography is a 3x3 matrix; apply with `warp_points`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    H_cam2proj: np.ndarray            # 3x3
    H_cam2table: np.ndarray | None    # 3x3 or None until table corners are set
    camera_size: tuple[int, int]      # (w, h)
    projector_size: tuple[int, int]   # (w, h)
    table_size_mm: tuple[int, int]    # (length, width)
    reproj_error_px: float = 0.0      # mean camera->projector reprojection error

    @property
    def H_proj2cam(self) -> np.ndarray:
        return np.linalg.inv(self.H_cam2proj)


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to Nx2 points. Returns Nx2."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    import cv2

    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)
