"""Box, keypoint and polygon helpers. Plain NumPy/OpenCV so every line is explainable.

Boxes are (x1, y1, x2, y2) in pixels. Keypoints are a (17, 3) array of (x, y, conf)
in COCO order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 11-12 hips, 15-16 ankles.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

KPT_CONF = 0.5

# COCO skeleton (pairs of keypoint indices) used for drawing.
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(inner, outer) -> float:
    """Intersection over the area of `inner`. A helmet fully on a head scores 1.0
    even though the head region is much bigger, which IoU would punish."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def _visible(kpts, idx, conf=KPT_CONF) -> bool:
    return kpts is not None and kpts[idx, 2] >= conf


def foot_point(box, kpts=None) -> tuple[float, float]:
    """Where the person touches the floor. Zones are drawn on the floor, so the box
    centre (waist height) would trigger zones the worker is only standing next to."""
    if _visible(kpts, 15) and _visible(kpts, 16):
        return float((kpts[15, 0] + kpts[16, 0]) / 2), float((kpts[15, 1] + kpts[16, 1]) / 2)
    return float((box[0] + box[2]) / 2), float(box[3])


def _expand(box, scale, shift_up=0.0):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    cy -= shift_up * h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def head_region(box, kpts=None):
    """Box where a helmet should be. Built from face keypoints (0-4), enlarged and
    shifted up because a helmet sits above the eyes. Falls back to the top of the
    person box; returns None when there is no head evidence at all."""
    if kpts is not None:
        pts = kpts[:5][kpts[:5, 2] >= KPT_CONF]
        if len(pts) >= 2:
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            side = max(x2 - x1, y2 - y1, 0.12 * (box[2] - box[0]))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return _expand((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), 1.8, 0.3)
        if kpts[:5, 2].max() < 0.2 and not (_visible(kpts, 5) or _visible(kpts, 6)):
            return None
    h = box[3] - box[1]
    return (box[0], box[1], box[2], box[1] + 0.22 * h)


def torso_region(kpts):
    """Box spanning shoulders and hips, where a vest should be. None if under 3 of 4 visible."""
    if kpts is None:
        return None
    idx = [i for i in (5, 6, 11, 12) if _visible(kpts, i)]
    if len(idx) < 3:
        return None
    pts = kpts[idx, :2]
    return _expand((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()), 1.15)


def torso_angle_deg(kpts) -> float | None:
    """Angle between the hip-mid -> shoulder-mid vector and image vertical.
    0 = upright, ~90 = lying. Image y points down, so 'up' is -y."""
    if kpts is None:
        return None
    sh = [i for i in (5, 6) if _visible(kpts, i)]
    hp = [i for i in (11, 12) if _visible(kpts, i)]
    if not sh or not hp:
        return None
    s = kpts[sh, :2].mean(axis=0)
    h = kpts[hp, :2].mean(axis=0)
    dx, dy = s[0] - h[0], s[1] - h[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), -dy))


def point_in_polygon(pt, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def denormalise_polygon(points, width: int, height: int) -> np.ndarray:
    """Zones are stored normalised (0-1) so one YAML works at any resolution."""
    arr = np.asarray(points, dtype=np.float32)
    return (arr * np.array([width, height], dtype=np.float32)).astype(np.int32)
