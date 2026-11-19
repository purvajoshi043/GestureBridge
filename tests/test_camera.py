import time

import numpy as np

from gesturebridge.hand_tracker import LatestFrameCamera


class FakeCapture:
    """A 30 FPS camera that numbers its frames."""

    def __init__(self):
        self.n = 0
        self.released = False

    def read(self):
        time.sleep(1 / 30)
        self.n += 1
        return True, np.full((4, 4, 3), self.n, dtype=np.uint8)

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


def test_slow_consumer_always_gets_the_newest_frame():
    fake = FakeCapture()
    cam = LatestFrameCamera(fake)
    try:
        ok, first = cam.read()
        time.sleep(0.5)                  # slow processing: ~15 frames arrive meanwhile
        ok, frame = cam.read()
        assert ok and int(frame[0, 0, 0]) >= int(first[0, 0, 0]) + 10   # no backlog of old frames
        assert int(frame[0, 0, 0]) >= fake.n - 2
    finally:
        cam.release()
    assert fake.released and not cam.isOpened()


def test_each_frame_is_returned_once():
    cam = LatestFrameCamera(FakeCapture())
    try:
        values = [int(cam.read()[1][0, 0, 0]) for _ in range(5)]
        assert values == sorted(set(values))
    finally:
        cam.release()
