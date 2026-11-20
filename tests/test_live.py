import threading
import time

import cv2
import numpy as np
import pytest
import requests

from gesturebridge.config import Settings
from gesturebridge.live import LiveEngine


@pytest.fixture
def fake_camera(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "frame.jpg"), np.full((480, 640, 3), 90, dtype=np.uint8))
    monkeypatch.setenv("GESTUREBRIDGE_FAKE_CAMERA", str(tmp_path))


def wait_until(condition, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_live_engine_streams_mjpeg(fake_camera):
    engine = LiveEngine(port=8560)
    engine.start(None, Settings())            # no agent: the stream still shows the camera
    try:
        assert engine.running
        with requests.get(engine.url, stream=True, timeout=5) as response:
            assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            data, end = b"", time.time() + 3
            while data.count(b"--frame") < 5 and time.time() < end:
                data += next(response.iter_content(8192))
        assert data.count(b"--frame") >= 5 and b"Content-Type: image/jpeg" in data
    finally:
        engine.stop()
    assert not engine.running


def test_camera_is_released_when_nobody_watches(fake_camera):
    engine = LiveEngine(port=8570)
    engine.idle_stop_seconds = 1.0
    engine.start(None, Settings())
    assert engine.running
    assert wait_until(lambda: not engine.running, timeout=4)        # no page ever connected
    assert engine._camera is None and "page was closed" in engine.notice


def test_camera_runs_while_watched_and_stops_after_the_tab_closes(fake_camera):
    engine = LiveEngine(port=8580)
    engine.idle_stop_seconds = 1.0
    engine.start(None, Settings())
    closed = threading.Event()

    def viewer():                           # a browser tab watching the stream for 2.5 s
        with requests.get(engine.url, stream=True, timeout=5) as response:
            end = time.time() + 2.5
            for _ in response.iter_content(8192):
                if time.time() > end:
                    break
        closed.set()

    threading.Thread(target=viewer, daemon=True).start()
    time.sleep(2.2)
    assert engine.running and engine.viewers == 1                   # watched: keeps running past the 1 s limit
    assert closed.wait(5)
    assert wait_until(lambda: not engine.running, timeout=4)        # tab closed -> camera released
    assert engine._camera is None


def test_camera_resumes_when_the_hidden_page_comes_back(fake_camera):
    engine = LiveEngine(port=8590)
    engine.idle_stop_seconds = 1.0
    engine.start(None, Settings())
    try:
        assert wait_until(lambda: not engine.running, timeout=4)    # page hidden: nobody watches
        assert engine.auto_paused and engine._camera is None
        with requests.get(engine.url, stream=True, timeout=5) as response:   # page visible again
            next(response.iter_content(8192))
            assert wait_until(lambda: engine.running, timeout=4)
            assert not engine.auto_paused
    finally:
        engine.stop()


def test_stop_button_is_not_undone_by_a_returning_page(fake_camera):
    engine = LiveEngine(port=8600)
    engine.start(None, Settings())
    engine.stop()                                                   # the user pressed Stop
    with requests.get(engine.url, stream=True, timeout=5) as response:
        next(response.iter_content(8192))
        time.sleep(1.0)
        assert not engine.running


def test_two_copies_of_the_app_never_share_a_stream_port():
    first, second = LiveEngine(port=8610), LiveEngine(port=8610)
    first.ensure_server()
    second.ensure_server()
    assert first.port != second.port            # Windows would otherwise let both bind 8610


def test_frames_shown_from_outside_reach_the_stream():
    engine = LiveEngine(port=8620)
    engine.ensure_server()
    got = []

    def viewer():
        with requests.get(engine.url, stream=True, timeout=5) as response:
            end, buf = time.time() + 1.5, b""
            for chunk in response.iter_content(4096):
                buf += chunk
                got.extend([1] * buf.count(b"--frame"))
                buf = buf.rsplit(b"--frame", 1)[-1]
                if time.time() > end:
                    break

    t = threading.Thread(target=viewer)
    t.start()
    time.sleep(0.3)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(30):                          # the recording preview publishes like this
        frame[:] = i
        engine.show(frame)
        time.sleep(1 / 30)
    t.join()
    assert len(got) >= 15
