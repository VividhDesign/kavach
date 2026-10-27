"""Safety events: JSONL log, snapshot crops, and per-(track, type) cooldown."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2


@dataclass
class Event:
    ts_video: float
    frame_idx: int
    track_id: int
    type: str  # ZONE_INTRUSION | FALL | MAN_DOWN | NO_HELMET | NO_VEST
    zone: str | None = None
    details: dict = field(default_factory=dict)
    snapshot_path: str | None = None

    def label(self) -> str:
        where = f" in {self.zone}" if self.zone else ""
        return f"{self.ts_video:6.1f}s  #{self.track_id}  {self.type}{where}"


class EventLog:
    def __init__(self, out_dir: Path, cooldown_s: float = 10.0):
        self.out_dir = Path(out_dir)
        (self.out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "events.jsonl"
        self.path.write_text("")
        self.cooldown_s = cooldown_s
        self.last: dict[tuple, float] = {}
        self.events: list[Event] = []

    def emit(self, ev: Event, frame=None, box=None) -> bool:
        """Logs the event unless the same (track, type, zone) fired within the cooldown."""
        key = (ev.track_id, ev.type, ev.zone)
        if key in self.last and ev.ts_video - self.last[key] < self.cooldown_s:
            return False
        self.last[key] = ev.ts_video
        if frame is not None and box is not None:
            ev.snapshot_path = self._snapshot(ev, frame, box)
        self.events.append(ev)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")
        return True

    def _snapshot(self, ev: Event, frame, box) -> str:
        h, w = frame.shape[:2]
        pad = 20
        x1, y1 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        x2, y2 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
        p = self.out_dir / "snapshots" / f"{ev.frame_idx:06d}_{ev.track_id}_{ev.type}.jpg"
        cv2.imwrite(str(p), frame[y1:y2, x1:x2])
        return str(p.relative_to(self.out_dir))
