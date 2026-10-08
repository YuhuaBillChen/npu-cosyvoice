"""Check the compiled GEMM overlay on this machine's NPU, bit-exact against numpy.

    python3 tests/check_gemm.py --aot aot

int8 x int8 -> int32 has no rounding, so every element must match. This is the first thing to run
on a new NPU (for example an XDNA2 build from scripts/compile_xclbin.py --device npu2); then
tests/check_estimator.py checks the whole DiT.
"""
import argparse
import json
import time

import numpy as np
import pyxrt

M, K, N = 512, 1024, 1024
NAME = "mmi8i32_512x1024x1024_64_64_64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aot", required=True)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(f"{args.aot}/{NAME}.xclbin")
    try:
        ctx = pyxrt.hw_context(dev, dev.register_xclbin(xb))
    except RuntimeError as e:
        # XDNA1 has 6 hardware contexts; a full NPU reports EINVAL, not EBUSY
        raise SystemExit(f"{e}\nno free NPU context? stop other NPU services (STT, TTS) and retry")
    k = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    ins = np.fromfile(f"{args.aot}/{NAME}.bin", dtype=np.uint32)

    def bo(nbytes, g, flags=pyxrt.bo.host_only):
        return pyxrt.bo(dev, nbytes, flags, k.group_id(g))

    ibo = bo(ins.nbytes, 1, pyxrt.bo.cacheable)
    ibo.write(ins.tobytes(), 0)
    ibo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    abo, bbo, cbo = bo(M * K, 3), bo(K * N, 4), bo(M * N * 4, 5)

    rng = np.random.default_rng(0)
    cases = {"random": (rng.integers(-128, 128, (M, K)), rng.integers(-127, 128, (K, N))),
             "extremes": (np.full((M, K), -128), np.full((K, N), 127))}  # largest |sum|: 2^27, fits int32
    result = {}
    for name, (a, b) in cases.items():
        a, b = a.astype(np.int8), b.astype(np.int8)
        abo.write(a.tobytes(), 0)
        bbo.write(b.tobytes(), 0)
        for x in (abo, bbo):
            x.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            k(3, ibo, len(ins), abo, bbo, cbo).wait()
            ts.append(time.perf_counter() - t0)
        cbo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        c = np.frombuffer(cbo.read(M * N * 4, 0), dtype=np.int32).reshape(M, N)
        ref = a.astype(np.int64) @ b.astype(np.int64)
        result[name] = dict(mismatches=int((c != ref).sum()), ms=round(sorted(ts)[len(ts) // 2] * 1e3, 3))
    result["tops"] = round(2 * M * K * N / (result["random"]["ms"] / 1e3) / 1e12, 2)
    result["ok"] = all(v["mismatches"] == 0 for v in result.values() if isinstance(v, dict))
    print(json.dumps(result))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
