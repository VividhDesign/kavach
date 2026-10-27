import numpy as np

from kavach.mandown import DOWN, MAN_DOWN, NORMAL, ManDownConfig, ManDownTracker
from kavach.ppe import MISSING, OK, UNKNOWN, PPEConfig, PPEDebouncer, associate
from kavach.zones import Zone, ZoneMonitor

FPS = 25


def body(lying: bool, jitter: float = 0.0) -> tuple[tuple, np.ndarray]:
    k = np.zeros((17, 3), dtype=np.float32)
    if lying:
        pts = {5: (100, 395), 6: (100, 405), 11: (170, 395), 12: (170, 405)}
        box = (60, 380, 260, 420)
    else:
        pts = {5: (95, 300), 6: (105, 300), 11: (95, 370), 12: (105, 370)}
        box = (80, 260, 120, 440)
    for i, (x, y) in pts.items():
        k[i] = (x + jitter, y, 0.9)
    return box, k


def run(tracker, seconds, lying, start, moving=False):
    events = []
    for f in range(int(seconds * FPS)):
        t = start + f / FPS
        jitter = (f % 2) * 30.0 if moving else 0.0
        box, k = body(lying, jitter)
        ev = tracker.update(t, box, k)
        if ev:
            events.append(ev)
    return events, start + seconds


def test_fall_then_still_becomes_man_down():
    tr = ManDownTracker(ManDownConfig(man_down_s=3.0))
    ev, t = run(tr, 2, lying=False, start=0)
    assert ev == [] and tr.state == NORMAL
    ev, t = run(tr, 1, lying=True, start=t)
    assert ev == ["FALL"] and tr.state == DOWN
    ev, t = run(tr, 3, lying=True, start=t)
    assert ev == ["MAN_DOWN"] and tr.state == MAN_DOWN
    ev, t = run(tr, 2, lying=False, start=t)
    assert tr.state == NORMAL


def test_already_lying_is_not_a_fall():
    tr = ManDownTracker()
    ev, _ = run(tr, 1, lying=True, start=0)
    assert "FALL" not in ev and tr.state == DOWN


def test_moving_on_the_floor_is_not_man_down():
    tr = ManDownTracker(ManDownConfig(man_down_s=2.0))
    _, t = run(tr, 2, lying=False, start=0)
    ev, _ = run(tr, 5, lying=True, start=t, moving=True)
    assert ev == ["FALL"] and tr.state == DOWN


def test_zone_fires_once_after_min_dwell_and_again_after_exit():
    z = Zone("pit", [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "hazard", min_dwell_frames=3)
    z.bind(100, 100)
    mon = ZoneMonitor([z])
    fired = [bool(mon.update(1, (10, 50))) for _ in range(6)]
    assert fired == [False, False, True, False, False, False]
    mon.update(1, (90, 50))  # leaves
    fired = [bool(mon.update(1, (10, 50))) for _ in range(3)]
    assert fired == [False, False, True]


def test_walkway_never_alerts():
    z = Zone("path", [[0, 0], [1, 0], [1, 1], [0, 1]], "walkway", min_dwell_frames=1)
    z.bind(100, 100)
    mon = ZoneMonitor([z])
    assert mon.update(1, (50, 50)) == []
    assert not mon.occupied(z) and mon.zones_of(1) == []


def _person(x):
    k = np.zeros((17, 3), dtype=np.float32)
    for i, (dx, dy) in {0: (0, 20), 1: (-5, 15), 2: (5, 15), 5: (-15, 50), 6: (15, 50)}.items():
        k[i] = (x + dx, dy + 100, 0.9)
    return (x - 30, 100, x + 30, 300), k


def test_helmet_on_own_head_ok_and_not_shared_with_neighbour():
    persons = {1: _person(100), 2: _person(180)}
    dets = [("helmet", (88, 95, 112, 112), 0.9)]
    s = associate(persons, dets, PPEConfig())
    assert s[1]["helmet"] == OK
    assert s[2]["helmet"] == MISSING


def test_small_person_is_unknown():
    box, k = _person(100)
    s = associate({1: ((70, 100, 130, 140), k)}, [], PPEConfig())
    assert s[1]["helmet"] == UNKNOWN


def test_debounce_needs_ratio_and_min_known():
    d = PPEDebouncer(PPEConfig(window_frames=10, violation_ratio=0.7, min_known=5))
    for _ in range(4):
        d.update("helmet", MISSING)
    assert not d.violating("helmet")  # only 4 known frames
    d.update("helmet", MISSING)
    assert d.violating("helmet")
    for _ in range(5):
        d.update("helmet", OK)
    assert not d.violating("helmet")  # 5/10 missing < 0.7
