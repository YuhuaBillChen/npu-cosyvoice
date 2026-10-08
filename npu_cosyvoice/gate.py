"""NPU priority gate shared between processes.

The NPU DiT library checks a 16-byte file at every layer boundary:

    int32 holders | int32 pad | int64 expiry_ms (CLOCK_REALTIME, ms since epoch)

While holders > 0 and the expiry is in the future it waits, so a higher-priority NPU user (for
example a speech-to-text service) gets the device to itself for the duration of its request. The
expiry bounds the pause if a holder dies without releasing.

Holder side, in the higher-priority service:

    gate = NpuGate("/dev/shm/npu-cosyvoice.gate")
    with gate.hold():
        transcribe(...)

The file only needs to be visible to both processes (bind-mount its directory into containers).
This module is stdlib-only so it can be vendored into another service.
"""
import contextlib
import fcntl
import os
import struct
import time

_FMT = "<iiq"


class NpuGate:
    def __init__(self, path, max_hold_s=120.0):
        self.path = path
        self.max_hold_s = max_hold_s
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if os.fstat(fd).st_size < 16:
                os.ftruncate(fd, 16)
        finally:
            os.close(fd)

    def _update(self, delta):
        with open(self.path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            holders, _, expiry = struct.unpack(_FMT, f.read(16))
            now_ms = int(time.time() * 1000)
            if expiry <= now_ms:
                holders = 0  # stale count from a holder that died
            holders = max(0, holders + delta)
            expiry = now_ms + int(self.max_hold_s * 1000) if holders else 0
            f.seek(0)
            f.write(struct.pack(_FMT, holders, 0, expiry))
            f.flush()

    @contextlib.contextmanager
    def hold(self):
        self._update(+1)
        try:
            yield
        finally:
            self._update(-1)
