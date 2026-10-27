"""Kavach CLI: python -m kavach.run --source samples/clip1.mp4 --site configs/sites/demo_site.yaml

Per frame: pose + ByteTrack -> (optional) PPE detection + association -> per-track
state (zones, man-down, PPE debounce) -> events -> render -> write.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from . import render  # noqa: E402
from .config import load_config, load_zones  # noqa: E402
from .events import Event, EventLog  # noqa: E402
from .geometry import foot_point  # noqa: E402
from .mandown import DOWN, MAN_DOWN, ManDownTracker  # noqa: E402
from .ppe import MISSING, OK, PPEDebouncer, associate  # noqa: E402
from .video import VideoSink, VideoSource  # noqa: E402
from .zones import ZoneMonitor  # noqa: E402


@dataclass
class TrackState:
    mandown: ManDownTracker
    ppe: PPEDebouncer
    last_seen: float = 0.0
    tags: list = field(default_factory=list)


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    ap = argparse.ArgumentParser(description="Kavach: vision-based worker safety monitoring")
    ap.add_argument("--source", required=True, help="video file, webcam index (0) or RTSP URL")
    ap.add_argument("--site", help="site YAML with zones")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", help="output video path (default: outputs/<source name>/out.mp4)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", action="store_true", help="show a live window (q to quit)")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--no-ppe", action="store_true", help="skip PPE checks even if weights exist")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    m = cfg["models"]
    device = args.device or default_device()
    Path(m["pose"]).parent.mkdir(parents=True, exist_ok=True)  # Ultralytics auto-downloads the pose weights here
    pose = YOLO(m["pose"])
    ppe_model = None
    if not args.no_ppe and Path(m["ppe"]).exists():
        ppe_model = YOLO(m["ppe"])
    print(f"device={device}  pose={m['pose']}  ppe={'on' if ppe_model else 'off (no weights)'}")

    src = VideoSource(args.source)
    zones = load_zones(args.site)
    for z in zones:
        z.bind(src.width, src.height)
    monitor = ZoneMonitor(zones)

    out_path = Path(args.out or f"outputs/{Path(args.source).stem}/out.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(str(out_path), src.fps, src.width, src.height)
    log = EventLog(out_path.parent, cfg["events"]["cooldown_s"])

    states: dict[int, TrackState] = {}
    stage_t = defaultdict(float)
    frame_idx, fps_ema, t_start = 0, 0.0, time.perf_counter()

    while True:
        frame = src.read()
        if frame is None or (args.max_frames and frame_idx >= args.max_frames):
            break
        t_frame = time.perf_counter()
        t = frame_idx / src.fps

        # 1. People + IDs + keypoints
        t0 = time.perf_counter()
        r = pose.track(frame, persist=True, tracker=m["tracker"], imgsz=m["imgsz"], conf=m["person_conf"],
                       classes=[0], device=device, verbose=False)[0]
        persons = {}
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else [None] * len(ids)
            persons = {int(i): (b, k) for i, b, k in zip(ids, boxes, kpts)}
        stage_t["pose_track"] += time.perf_counter() - t0

        # 2. PPE (optional)
        t0 = time.perf_counter()
        ppe_status = {}
        if ppe_model is not None and persons:
            pr = ppe_model.predict(frame, imgsz=m["imgsz"], conf=m["ppe_conf"], device=device, verbose=False)[0]
            names = pr.names
            dets = [(names[int(c)], b, float(s)) for b, c, s in
                    zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.cls.cpu().numpy(), pr.boxes.conf.cpu().numpy())]
            ppe_status = associate(persons, dets, cfg["ppe_cfg"])
        stage_t["ppe"] += time.perf_counter() - t0

        # 3. Per-track logic
        t0 = time.perf_counter()
        render_jobs = []
        for tid, (box, kp) in persons.items():
            st = states.setdefault(tid, TrackState(ManDownTracker(cfg["mandown_cfg"]), PPEDebouncer(cfg["ppe_cfg"])))
            st.last_seen = t
            foot = foot_point(box, kp)
            alert, tags = False, []

            for z in monitor.update(tid, foot):
                log.emit(Event(round(t, 2), frame_idx, tid, "ZONE_INTRUSION", z.name, {"zone_type": z.type}), frame, box)
            in_zones = monitor.zones_of(tid)
            if in_zones:
                alert = True
                tags.append("IN:" + ",".join(in_zones))

            ev = st.mandown.update(t, box, kp)
            if ev:
                log.emit(Event(round(t, 2), frame_idx, tid, ev), frame, box)
            if st.mandown.state == MAN_DOWN:
                alert = True
                tags.append("MAN DOWN")
            elif st.mandown.state == DOWN:
                tags.append("DOWN")

            for item, s in ppe_status.get(tid, {}).items():
                was = st.ppe.violating(item)
                st.ppe.update(item, s)
                shown = st.ppe.display(item)
                tags.append(f"{item[0].upper()}{'+' if shown == OK else '-' if shown == MISSING else '?'}")
                if st.ppe.violating(item):
                    alert = True
                    if not was:
                        log.emit(Event(round(t, 2), frame_idx, tid, f"NO_{item.upper()}"), frame, box)

            known = any(v != "UNKNOWN" for v in ppe_status.get(tid, {}).values()) or not ppe_status
            render_jobs.append((tid, box, kp, foot, "alert" if alert else "ok" if known else "unknown", tags))

        for tid in [k for k, s in states.items() if t - s.last_seen > cfg["track_ttl_s"]]:
            del states[tid]
            monitor.forget(tid)
        stage_t["logic"] += time.perf_counter() - t0

        # 4. Draw
        t0 = time.perf_counter()
        render.draw_zones(frame, zones, {z.name: monitor.occupied(z) for z in zones})
        for job in render_jobs:
            render.draw_person(frame, *job, skeleton=not args.no_skeleton)
        dt = time.perf_counter() - t_frame
        fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
        active = sum(j[4] == "alert" for j in render_jobs)
        render.draw_hud(frame, fps_ema, len(persons), active, [e.label() for e in log.events[-3:]])
        stage_t["render"] += time.perf_counter() - t0

        sink.write(frame)
        if args.show:
            cv2.imshow("Kavach", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}/{src.frames or '?'}  {fps_ema:.1f} FPS  events={len(log.events)}")

    src.release()
    sink.release()
    cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    counts = defaultdict(int)
    for e in log.events:
        counts[e.type] += 1
    summary = {
        "source": args.source, "frames": frame_idx, "device": device, "pose_model": m["pose"],
        "ppe": bool(ppe_model), "resolution": [src.width, src.height],
        "avg_fps": round(frame_idx / wall, 2) if wall else None,
        "ms_per_frame": {k: round(1000 * v / max(frame_idx, 1), 2) for k, v in stage_t.items()},
        "events": dict(counts),
    }
    (out_path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} and {log.path}")


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main()
