"""Video input/output. Live sources use a reader thread that keeps only the newest
frames, so a slow model shows the current scene instead of a growing backlog."""

from __future__ import annotations

import queue
import threading

import cv2


class VideoSource:
    def __init__(self, source: str):
        self.live = source.isdigit() or source.startswith(("rtsp://", "http://", "https://"))
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not 1 <= self.fps <= 120:
            self.fps = 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.live else 0
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        if self.live:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._q.put(None)
                return
            if self._q.full():
                try:
                    self._q.get_nowait()  # drop the stale frame
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self):
        if self.live:
            return self._q.get()
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._stop.set()
        self.cap.release()


class VideoSink:
    def __init__(self, path: str, fps: float, width: int, height: int):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
