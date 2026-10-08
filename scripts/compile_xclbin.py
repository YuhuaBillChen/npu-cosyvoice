"""AOT-compile the one NPU overlay libdit_npu uses: a 512x1024x1024 int8 x int8 -> int32 GEMM
(64x64x64 tiles, whole array), with MLIR-AIE's whole_array matrix-multiplication generator.
Writes mmi8i32_512x1024x1024_64_64_64.{xclbin,bin} to $NPU_AOT_DIR.

    NPU_AOT_DIR=$PWD/aot python3 scripts/compile_xclbin.py              # XDNA1 (Ryzen AI 7040/8040)
    NPU_AOT_DIR=$PWD/aot python3 scripts/compile_xclbin.py --device npu2  # XDNA2 (Ryzen AI 300), untested

Then check the overlay on the NPU with tests/check_gemm.py.
"""
import argparse
import os
import subprocess
import sys

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", "/opt/mlir-aie")
WA = os.path.join(MLIR_AIE, "programming_examples/basic/matrix_multiplication/whole_array/whole_array.py")
IRONENV = os.path.join(MLIR_AIE, "ironenv/bin/python3")
PEANO = os.environ.get("PEANO_INSTALL_DIR", os.path.join(MLIR_AIE, "ironenv/lib/python3.13/site-packages/llvm-aie"))
COLS = {"npu": 4, "npu2": 8}
NAME = "mmi8i32_512x1024x1024_64_64_64"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["npu1", "npu2"], default="npu1",
                    help="npu1 = XDNA1 (Phoenix, Hawk Point); npu2 = XDNA2 (Strix, Strix Halo, Krackan)")
    ap.add_argument("--out", default=os.environ.get("NPU_AOT_DIR", "aot"))
    args = ap.parse_args()
    dev = "npu" if args.device == "npu1" else "npu2"  # the generator's names
    os.makedirs(args.out, exist_ok=True)
    xcl, ins = f"{args.out}/{NAME}.xclbin", f"{args.out}/{NAME}.bin"
    for f in (xcl, ins):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, PYTHONPATH="/usr/lib/python3/dist-packages", PEANO_INSTALL_DIR=PEANO)
    cmd = [IRONENV, WA, "--dev", dev, "--n-aie-cols", str(COLS[dev]), "--dtype_in", "i8", "--dtype_out", "i32",
           "-M", "512", "-K", "1024", "-N", "1024", "-m", "64", "-k", "64", "-n", "64",
           f"--xclbin-path={os.path.abspath(xcl)}", f"--insts-path={os.path.abspath(ins)}"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=os.path.dirname(WA))
    if not (os.path.exists(xcl) and os.path.exists(ins)):
        sys.exit(f"compile failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    print(f"{args.device}: {xcl}")


if __name__ == "__main__":
    main()
