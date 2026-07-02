"""Shot controls the player adjusts, plus their projected widgets.

State:
  strength   0..1  -> mapped to cue-ball speed by the physics config
  english    (a, b): contact point on the cue ball face, each in [-1, 1]
               a = horizontal (left/right side spin), b = vertical (top/bottom).
               b > 0 is follow (top), b < 0 is draw (bottom).

Input for v1 is the KEYBOARD (works with a local display window/projector):
  [ ]   strength down / up
  a d   side english   left / right
  w s   follow / draw   up  / down
  c     centre the contact point
This is deliberately behind a small state object so a future input method --
camera-tracked fingertip on the projected slider, or a phone web UI -- can drive
the same state without touching the rest of the app.

The widgets are drawn directly in PROJECTOR pixels at a fixed corner so they land
on the cloth near a rail. Colours are bright on the assumption of a dark overlay.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ShotControls:
    strength: float = 0.5
    a: float = 0.0                    # side english, -1 (left) .. +1 (right)
    b: float = 0.0                    # vertical english, -1 (draw) .. +1 (follow)
    strength_step: float = 0.05
    english_step: float = 0.1

    @classmethod
    def from_config(cls, cfg) -> "ShotControls":
        return cls(strength=cfg.strength, strength_step=cfg.strength_step,
                   english_step=cfg.english_step)

    @property
    def english(self) -> tuple[float, float]:
        return (self.a, self.b)

    def handle_key(self, key: int) -> bool:
        """Update state from a keypress. Returns True if the key was a control."""
        if key in (ord("["), ord("-")):
            self.strength = max(0.0, self.strength - self.strength_step)
        elif key in (ord("]"), ord("=")):
            self.strength = min(1.0, self.strength + self.strength_step)
        elif key == ord("a"):
            self.a = max(-1.0, self.a - self.english_step)
        elif key == ord("d"):
            self.a = min(1.0, self.a + self.english_step)
        elif key == ord("w"):
            self.b = min(1.0, self.b + self.english_step)
        elif key == ord("s"):
            self.b = max(-1.0, self.b - self.english_step)
        elif key == ord("c"):
            self.a = self.b = 0.0
        else:
            return False
        return True

    def describe(self) -> str:
        spin = "centre" if (self.a == 0 and self.b == 0) else f"a={self.a:+.1f} b={self.b:+.1f}"
        return f"strength {self.strength * 100:3.0f}%  |  english {spin}"


def draw_controls(overlay: np.ndarray, controls: ShotControls,
                  origin: tuple[int, int] | None = None, scale: float = 1.0) -> None:
    """Draw the strength meter and cue-ball contact widget onto `overlay`."""
    h, w = overlay.shape[:2]
    s = scale
    if origin is None:
        origin = (int(30 * s), h - int(200 * s))
    ox, oy = origin

    # --- strength meter (vertical bar) ---
    bar_w, bar_h = int(34 * s), int(160 * s)
    cv2.rectangle(overlay, (ox, oy), (ox + bar_w, oy + bar_h), (180, 180, 180), 2)
    fill = int(bar_h * controls.strength)
    # colour ramps green -> red with power
    col = (0, int(255 * (1 - controls.strength)), int(255 * controls.strength))
    cv2.rectangle(overlay, (ox + 2, oy + bar_h - fill),
                  (ox + bar_w - 2, oy + bar_h - 2), col, -1)
    cv2.putText(overlay, f"{controls.strength * 100:.0f}%", (ox - 2, oy - int(8 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), 1)
    cv2.putText(overlay, "PWR", (ox - int(2 * s), oy + bar_h + int(18 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (200, 200, 200), 1)

    # --- cue-ball contact widget (circle with a dot at the contact point) ---
    cr = int(46 * s)
    cx, cy = ox + bar_w + int(60 * s), oy + cr
    cv2.circle(overlay, (cx, cy), cr, (230, 230, 230), 2)
    cv2.line(overlay, (cx - cr, cy), (cx + cr, cy), (70, 70, 70), 1)
    cv2.line(overlay, (cx, cy - cr), (cx, cy + cr), (70, 70, 70), 1)
    # b>0 (follow) is UP on screen, so subtract b from y
    dot = (int(cx + controls.a * cr * 0.8), int(cy - controls.b * cr * 0.8))
    cv2.circle(overlay, dot, max(4, int(7 * s)), (0, 215, 255), -1)
    cv2.putText(overlay, "HIT", (cx - int(14 * s), cy + cr + int(18 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (200, 200, 200), 1)
