"""PPE association (helmet, vest) and per-track debouncing.

Each frame, every tracked person gets a tri-state per item: OK, MISSING or UNKNOWN.
UNKNOWN means "can't tell" (head not visible, person too small) and never raises an
alert. In industry a false alarm costs more than a late one, because operators stop
trusting a system that cries wolf.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .geometry import head_region, ioa, torso_region

OK, MISSING, UNKNOWN = "OK", "MISSING", "UNKNOWN"
ITEMS = {"helmet": ("helmet", head_region), "vest": ("safety-vest", torso_region)}


@dataclass
class PPEConfig:
    min_ioa: float = 0.30
    min_person_h: int = 60
    window_frames: int = 15
    violation_ratio: float = 0.7
    min_known: int = 8


def _region(item, box, kpts):
    _, fn = ITEMS[item]
    return fn(box, kpts) if item == "helmet" else fn(kpts)


def associate(persons, ppe_dets, cfg: PPEConfig) -> dict[int, dict[str, str]]:
    """persons: {track_id: (box, kpts)}. ppe_dets: list of (class_name, box, conf).

    Greedy one-to-one matching by highest IoA, so one helmet between two people
    can't count for both of them.
    """
    status = {tid: {} for tid in persons}
    for item, (cls, _) in ITEMS.items():
        regions = {tid: _region(item, b, k) for tid, (b, k) in persons.items()}
        boxes = [d[1] for d in ppe_dets if d[0] == cls]
        pairs = sorted(
            ((ioa(pb, r), tid, j) for tid, r in regions.items() if r is not None for j, pb in enumerate(boxes)),
            reverse=True,
        )
        matched_t, matched_b = set(), set()
        for score, tid, j in pairs:
            if score < cfg.min_ioa:
                break
            if tid in matched_t or j in matched_b:
                continue
            matched_t.add(tid)
            matched_b.add(j)
        for tid, (b, _) in persons.items():
            if tid in matched_t:
                status[tid][item] = OK
            elif regions[tid] is None or (b[3] - b[1]) < cfg.min_person_h:
                status[tid][item] = UNKNOWN
            else:
                status[tid][item] = MISSING
    return status


@dataclass
class PPEDebouncer:
    """Sliding window per item: a violation needs `violation_ratio` MISSING among the
    known (non-UNKNOWN) frames, and at least `min_known` known frames."""

    cfg: PPEConfig = field(default_factory=PPEConfig)
    windows: dict = field(default_factory=dict)

    def update(self, item: str, state: str) -> None:
        self.windows.setdefault(item, deque(maxlen=self.cfg.window_frames)).append(state)

    def violating(self, item: str) -> bool:
        known = [s for s in self.windows.get(item, ()) if s != UNKNOWN]
        if len(known) < self.cfg.min_known:
            return False
        return sum(s == MISSING for s in known) / len(known) >= self.cfg.violation_ratio

    def display(self, item: str) -> str:
        if self.violating(item):
            return MISSING
        known = [s for s in self.windows.get(item, ()) if s != UNKNOWN]
        return OK if len(known) >= self.cfg.min_known else UNKNOWN
