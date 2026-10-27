"""Danger-zone intrusion: a worker's foot point inside a floor polygon for N frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import denormalise_polygon, point_in_polygon


@dataclass
class Zone:
    name: str
    points: list  # normalised [[x, y], ...]
    type: str = "restricted"  # restricted | hazard | walkway
    min_dwell_frames: int = 5
    poly: np.ndarray | None = None  # pixel polygon, set by bind()

    def bind(self, width: int, height: int) -> None:
        self.poly = denormalise_polygon(self.points, width, height)

    def contains(self, pt) -> bool:
        return point_in_polygon(pt, self.poly)


@dataclass
class ZoneMonitor:
    """Counts consecutive frames each track spends in each zone.

    The event fires once, on the frame the count reaches `min_dwell_frames`. That
    debounce stops one jittery foot point on the boundary from spamming alerts.
    Leaving the zone resets the count, so re-entry fires again.
    """

    zones: list[Zone]
    inside: dict = field(default_factory=dict)  # (track_id, zone) -> consecutive frames
    dwell_frames: dict = field(default_factory=dict)  # (track_id, zone) -> total frames inside

    def update(self, track_id: int, foot) -> list[Zone]:
        """Returns the zones this track has just intruded into (on this frame)."""
        entered = []
        for z in self.zones:
            key = (track_id, z.name)
            if z.contains(foot):
                n = self.inside.get(key, 0) + 1
                self.inside[key] = n
                self.dwell_frames[key] = self.dwell_frames.get(key, 0) + 1
                if n == z.min_dwell_frames and z.type != "walkway":
                    entered.append(z)
            else:
                self.inside[key] = 0
        return entered

    def occupied(self, zone: Zone) -> bool:
        """True while someone is inside a zone that alerts (walkways never do)."""
        if zone.type == "walkway":
            return False
        return any(n >= zone.min_dwell_frames for (_, name), n in self.inside.items() if name == zone.name)

    def zones_of(self, track_id: int, alerting_only: bool = True) -> list[str]:
        return [z.name for z in self.zones
                if self.inside.get((track_id, z.name), 0) >= z.min_dwell_frames
                and not (alerting_only and z.type == "walkway")]

    def forget(self, track_id: int) -> None:
        for key in [k for k in self.inside if k[0] == track_id]:
            del self.inside[key]
