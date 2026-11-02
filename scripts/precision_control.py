#!/usr/bin/env python3
"""Controlled test of the numerical-noise-floor mechanism.

The numerical noise floors vary by three orders of magnitude between models.
This script tests the hypothesis that they are a floating-point-precision artefact rather
than an intrinsic property of an architecture, using the one case where the same trained
model is distributed at two precisions: GRACE-2L-OMAT-medium-base (float32) and
GRACE-2L-OMAT-medium-base-fp64 (float64). Same weights, same architecture, same graph
construction; only the arithmetic differs.

Protocol identical to the production noise-floor measurement (batch1_fixes.py): evaluate one cell under
operations that leave the energy of a rotation- and inversion-invariant potential exactly
unchanged, and report the spread.

Run in ~/.venvs/umlip2. Writes results/precision_control.json.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).parent.parent
PAIR = ["GRACE-2L-OMAT-medium-base", "GRACE-2L-OMAT-medium-base-fp64"]


def noise_floor(calc, base, n_fu, seed=0):
    rng = np.random.default_rng(seed)

    def E(at):
        a = at.copy(); a.calc = calc
        return a.get_potential_energy()

    e0 = E(base)
    rep = [E(base) for _ in range(5)]
    rots = []
    for _ in range(5):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        a = base.copy(); a.rotate(float(rng.uniform(0, 360)), v, rotate_cell=True)
        rots.append(E(a))
    trs = []
    for _ in range(5):
        a = base.copy(); a.translate(rng.uniform(-3, 3, size=3)); a.wrap()
        trs.append(E(a))
    m = base.copy(); M = np.diag([-1.0, 1.0, 1.0])
    m.set_cell(np.array(m.get_cell()) @ M, scale_atoms=False)
    m.set_positions(m.get_positions() @ M); m.wrap()

    def sp(v):
        return (max(v) - min(v)) / n_fu * 1000

    out = {"repeat": sp(rep), "rotation": sp(rots + [e0]),
           "translation": sp(trs + [e0]), "mirror": abs(E(m) - e0) / n_fu * 1000}
    out["noise_floor_meV_per_fu"] = max(out.values())
    return out


def main():
    from tensorpotential.calculator import grace_fm
    base = read(ROOT / "structures" / "CrN_cubic_conv.extxyz")
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")
    res = {}
    for name in PAIR:
        calc = grace_fm(name)
        res[name] = noise_floor(calc, base, n_fu)
        print(f"{name}: floor = {res[name]['noise_floor_meV_per_fu']:.5f} meV/f.u. "
              f"(rot {res[name]['rotation']:.5f}, trans {res[name]['translation']:.5f})",
              flush=True)
    f32, f64 = (res[PAIR[0]]["noise_floor_meV_per_fu"], res[PAIR[1]]["noise_floor_meV_per_fu"])
    res["ratio_f32_over_f64"] = (f32 / f64) if f64 > 0 else None
    (ROOT / "results" / "precision_control.json").write_text(json.dumps(res, indent=1))
    print(f"ratio = {res['ratio_f32_over_f64']}")


if __name__ == "__main__":
    main()
