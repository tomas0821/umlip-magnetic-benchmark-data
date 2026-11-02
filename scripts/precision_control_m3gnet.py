#!/usr/bin/env python3
"""Controlled precision test for M3GNet-MatPES, companion to precision_control.py.

Evaluates the same trained M3GNet-PES-MatPES-PBE-2025.2 weights at float32 and float64
(matgl.set_default_dtype) on the 8-atom cubic CrN cell, with the noise-floor protocol of
precision_control.py (repeat, five random rotations, five random translations, mirror). If the
floor were a floating-point-width artefact, it would collapse in float64 as GRACE's does.

Run in ~/.venvs/umlip. Writes results/precision_control_m3gnet.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from precision_control import noise_floor  # noqa: E402  (same protocol, same seed)


def floor_at(dtype):
    import matgl
    from ase.io import read
    from matgl.ext.ase import PESCalculator
    matgl.set_default_dtype("float", int(dtype.replace("float", "")))
    # pinned: the Hub name was re-uploaded after the production runs (Aug 2026)
    calc = PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2", revision="8414b23e68b173d6e36271e4366ee3717364051c"))
    base = read(ROOT / "structures" / "CrN_cubic_conv.extxyz")
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")
    return noise_floor(calc, base, n_fu)


def main():
    dtype = sys.argv[1] if len(sys.argv) > 1 else None
    if dtype:                      # one precision per process: the default dtype is global
        print(json.dumps(floor_at(dtype)))
        return
    import subprocess
    res = {}
    for dt in ("float32", "float64"):
        out = subprocess.run([sys.executable, __file__, dt], capture_output=True, text=True, check=True)
        res[dt] = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"M3GNet-MatPES {dt}: floor = {res[dt]['noise_floor_meV_per_fu']:.4f} meV/f.u. "
              f"(rot {res[dt]['rotation']:.4f}, trans {res[dt]['translation']:.4f}, "
              f"mirror {res[dt]['mirror']:.4f})", flush=True)
    f32, f64 = res["float32"]["noise_floor_meV_per_fu"], res["float64"]["noise_floor_meV_per_fu"]
    res["ratio_f32_over_f64"] = f32 / f64 if f64 > 0 else None
    (ROOT / "results" / "precision_control_m3gnet.json").write_text(json.dumps(res, indent=1))
    print(f"ratio = {res['ratio_f32_over_f64']}")


if __name__ == "__main__":
    main()
