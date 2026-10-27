"""All drawing, in plain OpenCV: zones, people, skeletons, HUD and alert ticker."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import KPT_CONF, SKELETON

GREEN, AMBER, RED, WHITE, BLACK = (60, 200, 60), (0, 190, 255), (40, 40, 230), (255, 255, 255), (0, 0, 0)
ZONE_COLOURS = {"restricted": (0, 140, 255), "hazard": (40, 40, 230), "walkway": (200, 160, 60)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org, colour, scale=0.5, thick=1):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 6, y), colour, -1)
    cv2.putText(img, text, (x + 3, y - base - 1), FONT, scale, WHITE, thick, cv2.LINE_AA)


def draw_zones(img, zones, occupied: dict[str, bool]):
    """Filled polygons blended with addWeighted so the scene stays visible underneath."""
    overlay = img.copy()
    for z in zones:
        c = RED if occupied.get(z.name) else ZONE_COLOURS.get(z.type, AMBER)
        cv2.fillPoly(overlay, [z.poly], c)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
    for z in zones:
        c = RED if occupied.get(z.name) else ZONE_COLOURS.get(z.type, AMBER)
        cv2.polylines(img, [z.poly], True, c, 2, cv2.LINE_AA)
        top = z.poly[np.argmin(z.poly[:, 1])]
        label(img, f"{z.name} ({z.type})", (top[0], top[1] - 2), c, 0.45)


def draw_skeleton(img, kpts, colour):
    if kpts is None:
        return
    for a, b in SKELETON:
        if kpts[a, 2] >= KPT_CONF and kpts[b, 2] >= KPT_CONF:
            cv2.line(img, (int(kpts[a, 0]), int(kpts[a, 1])), (int(kpts[b, 0]), int(kpts[b, 1])), colour, 2, cv2.LINE_AA)
    for x, y, c in kpts:
        if c >= KPT_CONF:
            cv2.circle(img, (int(x), int(y)), 3, colour, -1, cv2.LINE_AA)


def draw_person(img, tid, box, kpts, foot, status: str, tags: list[str], skeleton=True):
    colour = {"ok": GREEN, "unknown": AMBER, "alert": RED}[status]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
    if skeleton:
        draw_skeleton(img, kpts, colour)
    cv2.circle(img, (int(foot[0]), int(foot[1])), 5, colour, -1, cv2.LINE_AA)
    label(img, " ".join([f"#{tid}", *tags]), (x1, y1), colour)


def draw_hud(img, fps: float, workers: int, active: int, ticker: list[str]):
    h, w = img.shape[:2]
    lines = [f"KAVACH  |  {fps:4.1f} FPS  |  workers: {workers}  |  active alerts: {active}"]
    (tw, th), _ = cv2.getTextSize(lines[0], FONT, 0.6, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 18), BLACK, -1)
    cv2.putText(img, lines[0], (10, th + 8), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    if ticker:
        bar_h = 24 * len(ticker) + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
        for i, t in enumerate(ticker):
            cv2.putText(img, t, (10, h - bar_h + 24 * (i + 1)), FONT, 0.55, (90, 90, 255), 1, cv2.LINE_AA)
