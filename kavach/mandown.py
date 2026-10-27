"""Rule-based fall / man-down detection from pose keypoints.

Per track we keep a short history of (time, lying?, keypoints) and run a small
state machine:

    NORMAL --lying for confirm_s--> DOWN --lying and still for man_down_s--> MAN_DOWN
       ^                              |                                       |
       +------ upright for recover_s -+---------------------------------------+

A FALL event is raised on NORMAL -> DOWN only when the person was seen upright in
the last `fall_window_s`; someone already lying when the camera first sees them
(e.g. working under a vehicle) goes straight to DOWN without a FALL alert.
Time comes from the frame index / fps, so a video gives the same result every run.

This is the Phase-1 baseline; Phase 2 replaces `is_lying` with a trained 1D-CNN.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .geometry import KPT_CONF, torso_angle_deg

NORMAL, DOWN, MAN_DOWN = "NORMAL", "DOWN", "MAN_DOWN"


class Posture(Enum):
    UPRIGHT = 0
    LYING = 1
    UNKNOWN = 2


@dataclass
class ManDownConfig:
    lying_angle_deg: float = 60.0  # torso this far from vertical counts as lying
    upright_angle_deg: float = 35.0
    lying_aspect: float = 1.3  # box w/h fallback when shoulders/hips aren't visible
    confirm_s: float = 0.5
    fall_window_s: float = 2.0
    man_down_s: float = 4.0
    still_motion: float = 0.3  # keypoint path length per second, in body diagonals
    recover_s: float = 1.0


def posture(box, kpts, cfg: ManDownConfig) -> Posture:
    ang = torso_angle_deg(kpts)
    if ang is not None:
        if ang >= cfg.lying_angle_deg:
            return Posture.LYING
        if ang <= cfg.upright_angle_deg:
            return Posture.UPRIGHT
        return Posture.UNKNOWN
    w, h = box[2] - box[0], box[3] - box[1]
    if h <= 0:
        return Posture.UNKNOWN
    if w / h >= cfg.lying_aspect:
        return Posture.LYING
    if w / h <= 0.8:
        return Posture.UPRIGHT
    return Posture.UNKNOWN


def motion_rate(history, box) -> float | None:
    """Keypoint path length per second, normalised by body size (box diagonal) so it
    means the same near or far from the camera. Path length, not start-to-end
    displacement: a worker waving for help moves back and forth and must not
    look 'still'."""
    path, span = 0.0, 0.0
    for (t0, _, k0), (t1, _, k1) in zip(history, list(history)[1:]):
        if k0 is None or k1 is None:
            continue
        ok = (k0[:, 2] >= KPT_CONF) & (k1[:, 2] >= KPT_CONF)
        if ok.sum() < 4:
            continue
        path += float(np.linalg.norm(k1[ok, :2] - k0[ok, :2], axis=1).mean())
        span += t1 - t0
    if span <= 0:
        return None
    diag = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
    return path / diag / span


@dataclass
class ManDownTracker:
    cfg: ManDownConfig = field(default_factory=ManDownConfig)
    state: str = NORMAL
    since: float = 0.0  # time the current posture run started
    last_upright: float | None = None
    run_posture: Posture = Posture.UNKNOWN
    run_start: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    def update(self, t: float, box, kpts) -> str | None:
        """Feed one observation; returns "FALL" or "MAN_DOWN" on the frame it happens."""
        p = posture(box, kpts, self.cfg)
        if p is Posture.UPRIGHT:
            self.last_upright = t
        if p is not self.run_posture and p is not Posture.UNKNOWN:
            self.run_posture, self.run_start = p, t
        run_len = t - self.run_start
        self.history.append((t, p, None if kpts is None else kpts.copy()))
        while self.history and t - self.history[0][0] > 1.0:
            self.history.popleft()

        if self.state == NORMAL:
            if self.run_posture is Posture.LYING and run_len >= self.cfg.confirm_s:
                self.state, self.since = DOWN, t
                fell = self.last_upright is not None and self.run_start - self.last_upright <= self.cfg.fall_window_s
                return "FALL" if fell else None
        elif self.run_posture is Posture.UPRIGHT and run_len >= self.cfg.recover_s:
            self.state = NORMAL
        elif self.state == DOWN and t - self.since >= self.cfg.man_down_s:
            m = motion_rate(self.history, box)
            if m is None or m <= self.cfg.still_motion:
                self.state = MAN_DOWN
                return "MAN_DOWN"
        return None
