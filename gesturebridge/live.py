"""Smooth live video: the camera loop runs in a background thread and the browser
watches it as an MJPEG stream (like an IP camera) from a tiny local HTTP server.

The camera is only used while the site is open and visible: when no browser is
watching the stream any more (tab closed, page left, browser tab in the background -
the page stops watching then) the camera is released after a few seconds. If the
same page becomes visible again it reconnects and the camera resumes by itself.
When the app itself stops, the camera is released too.

Before, every frame was pushed through Streamlit's page updates, which made the
video stutter and fall behind. Now the Streamlit page only shows
<img src="http://127.0.0.1:8503/stream.mjpg"> and refreshes the small text panels
a few times per second; recognition runs on every camera frame in this thread.
"""

from __future__ import annotations

import atexit
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .hand_tracker import open_camera

STREAM_PORT = 8503
STREAM_WIDTH = 640
IDLE_STOP_SECONDS = 5.0      # release the camera when no page has watched the stream for this long
RESUME_WINDOW_SECONDS = 600  # a page that comes back within this time resumes the camera by itself


def _placeholder(text: str) -> bytes:
    image = np.full((360, 640, 3), (36, 22, 18), dtype=np.uint8)
    cv2.putText(image, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 220, 255), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


class LiveEngine:
    def __init__(self, port: int = STREAM_PORT):
        self.port = port
        self.agent = None
        self.fps = 0.0
        self.error = ""
        self.notice = ""             # e.g. why the camera was switched off automatically
        self.idle_stop_seconds = IDLE_STOP_SECONDS
        self._viewers = 0
        self._camera = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._jpeg = _placeholder("Camera off - press Start")
        self._seq = 0
        self._cond = threading.Condition()
        self._server: ThreadingHTTPServer | None = None
        self._lock = threading.Lock()
        self._settings = None
        self.auto_paused = False     # switched off because nobody watched (not by the Stop button)
        self._paused_at = 0.0
        atexit.register(self.stop)   # the app is closing: let go of the camera

    # -- HTTP stream ------------------------------------------------------------
    def ensure_server(self) -> None:
        if self._server is not None:
            return
        engine = self

        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                if not self.path.startswith("/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                engine._count_viewer(+1)
                try:
                    for jpeg in engine.frames():
                        engine.maybe_resume()           # the page is back: switch the camera on again
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return                      # the browser tab was closed or reloaded
                finally:
                    engine._count_viewer(-1)

            def log_message(self, *args):  # keep the console quiet
                pass

        for port in range(self.port, self.port + 20):   # another copy of the app may hold the first port
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", port), StreamHandler)
                self.port = port
                break
            except OSError:
                continue
        else:
            raise RuntimeError("No free port for the video stream (8503-8522)")
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="gb-stream").start()

    def _count_viewer(self, delta: int) -> None:
        with self._cond:
            self._viewers += delta

    @property
    def viewers(self) -> int:
        return self._viewers

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream.mjpg"

    def frames(self):
        """Yield the newest JPEG whenever it changes (and at least once a second)."""
        seq = -1
        while True:
            with self._cond:
                if self._seq == seq:
                    self._cond.wait(timeout=1.0)
                seq, jpeg = self._seq, self._jpeg
            yield jpeg

    def _publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    # -- camera loop ---------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    def start(self, agent, settings) -> None:
        with self._lock:
            self.ensure_server()
            if self._running:
                self.agent = agent
                return
            self.error = ""
            self.notice = ""
            self.auto_paused = False
            self.agent = agent
            self._settings = settings
            self._camera = open_camera(settings.camera.index, settings.camera.width, settings.camera.height)
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True, name="gb-camera")
            self._thread.start()

    def maybe_resume(self) -> None:
        """Restart the camera if it was switched off only because the page was hidden for a while."""
        if not self.auto_paused or self._running or time.time() - self._paused_at > RESUME_WINDOW_SECONDS:
            return
        try:
            self.start(self.agent, self._settings)
        except Exception as exc:  # e.g. another program took the camera meanwhile
            self.auto_paused = False
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        with self._lock:
            self.auto_paused = False
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=3)    # the camera thread releases the camera itself
                self._thread = None
            self._release()

    def _release(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self.fps = 0.0
        self._publish(_placeholder("Camera off - press Start"))

    def _loop(self) -> None:
        frames, tick = 0, time.time()
        last_watched = time.time()          # starting counts as watched: the page needs a moment to connect
        idle = False
        try:
            if self.agent is not None:
                self.agent.ensure_tracker()
            while self._running:
                if self._viewers > 0:
                    last_watched = time.time()
                elif time.time() - last_watched > self.idle_stop_seconds:
                    self.notice = ("The camera was switched off because the page was closed or hidden "
                                   "(browser tab in the background).")
                    idle = True
                    break
                ok, frame = self._camera.read()
                if not ok:
                    self.error = "Could not read from the camera. Is another program using it?"
                    break
                agent = self.agent
                result = agent.step(frame) if agent is not None else None
                view = result.frame if result is not None else frame
                if view.shape[1] > STREAM_WIDTH:
                    view = cv2.resize(view, (STREAM_WIDTH, round(view.shape[0] * STREAM_WIDTH / view.shape[1])),
                                      interpolation=cv2.INTER_AREA)
                self._publish(cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                frames += 1
                if time.time() - tick >= 1.0:
                    self.fps, frames, tick = frames / (time.time() - tick), 0, time.time()
        except Exception as exc:  # show the problem in the page instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._release()                     # the camera light goes off as soon as the loop ends
            self._running = False
            if idle:
                self.auto_paused, self._paused_at = True, time.time()
