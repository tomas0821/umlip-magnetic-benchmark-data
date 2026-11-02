#!/usr/bin/env python3
"""Transfer the CrN protocol to MnO: does the benchmark's verdict generalise?

Reference: Eckhoff & Behler, npj Comput. Mater. 7, 170 (2021), HSE06 values. No new DFT.

Three measurements per model, mirroring the CrN pipeline exactly:

  (1) PHASE PROBE (single-point at the published DFT geometries). uMLIPs take no spin input,
      so the only channel through which they can see magnetic order is the geometry it
      induces. Two contrasts:
        ordering   E(FM cubic) - E(AFM-II rhombohedral)          DFT: 45.5 meV/atom
        distortion E(AFM-II cubic) - E(AFM-II rhombohedral)      DFT:  2.1 meV/atom
      both evaluated at fixed published lattice parameters.

  (2) SYMMETRY RETENTION (free cell relaxation from each DFT geometry). The ground state is
      rhombohedral with alpha = 90.89 deg. A model that carries the magnetostructural
      coupling keeps the distortion; a model that does not collapses to alpha = 90 deg. This
      is the trigonal analogue of the CrN [110] shear test, and the elastic-stability
      diagnostic predicts the outcome in advance.

  (3) CUBIC-CONSTRAINED EQUILIBRIUM (energy vs a, cubic cell). In rocksalt the ions sit on
      fixed Wyckoff positions under both trigonal and tetragonal strain, so no ionic
      relaxation is possible and a scan over `a` is exact, not a fit artefact.

WHY THIS IS A REAL TEST AND NOT A REPEAT. The distortion signal here is 2.1 meV/atom
(= 4.2 meV/f.u.), roughly half the CrN signal of 8.0 meV/f.u., and the distortion mode is
trigonal rather than a [110] shear. The resolution diagnostic therefore makes a falsifiable
prediction BEFORE the run: any model whose measured noise floor approaches 4.2 meV/f.u.
cannot resolve this quantity at all, whatever number it prints.

One model per process (see run_validation.sh) -- the VM has 7 GB.
Writes results/mno_results.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase.cell import Cell
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE, LBFGS

torch.set_num_threads(4)

_TL = torch.load
torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _TL(*a, **k))[1]

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
BASE = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]
EXTRA = ["Orb-v3", "GRACE-2L-OMAT"]

# HSE06, meV/atom
DFT = {"ordering_FM_minus_AFM2": 45.5, "distortion_cubic_minus_rhomb": 2.1,
       "alpha_AFM2_deg": 90.89, "a_AFM2_A": 4.434, "a_FM_A": 4.462}

sys.path.insert(0, str(Path(__file__).parent))
from diagnostic_validation import make_calc  # noqa: E402  (shared model loader)


def _json_default(o):
    """CHGNet and MatterSim return float32; json refuses it. round() on a numpy scalar
    returns a numpy scalar, so this has to be handled at serialisation time, not at the
    call sites."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def dump(obj):
    return json.dumps(obj, indent=1, default=_json_default)


def relax(atoms, calc, cell=True, fmax=0.005, steps=2000):
    atoms.calc = calc
    tgt = FrechetCellFilter(atoms) if cell else atoms
    o = FIRE(tgt, logfile=None)
    o.run(fmax=fmax, steps=steps)
    if not o.converged():
        o = LBFGS(tgt, logfile=None)
        o.run(fmax=fmax, steps=steps)
    return atoms, bool(o.converged())


def phase_probe(calc):
    """(1) Single-point energies at the published DFT geometries, meV/atom."""
    e = {}
    for tag in ("MnO_FM_cubic", "MnO_AFM1_tetra", "MnO_AFM2_rhomb",
                "MnO_AFM2_cubic_seed", "MnO_AFM2_rhomb_exp"):
        at = read(SDIR / f"{tag}.extxyz")
        at.calc = calc
        e[tag] = at.get_potential_energy() / len(at) * 1000.0  # meV/atom
    out = {"energies_meV_per_atom": {k: round(v, 6) for k, v in e.items()}}
    out["ordering_FM_minus_AFM2"] = round(e["MnO_FM_cubic"] - e["MnO_AFM2_rhomb"], 4)
    out["distortion_cubic_minus_rhomb"] = round(
        e["MnO_AFM2_cubic_seed"] - e["MnO_AFM2_rhomb"], 4)
    out["DFT_ordering"] = DFT["ordering_FM_minus_AFM2"]
    out["DFT_distortion"] = DFT["distortion_cubic_minus_rhomb"]
    return out


def symmetry_retention(calc):
    """(2) Free cell relaxation from each DFT geometry: is the distortion preserved?"""
    out = {}
    for tag in ("MnO_AFM2_rhomb", "MnO_AFM1_tetra", "MnO_FM_cubic"):
        at = read(SDIR / f"{tag}.extxyz")
        at, conv = relax(at, calc, cell=True)
        a, b, c, al, be, ga = [float(x) for x in at.cell.cellpar()]
        dev = max(abs(al - 90), abs(be - 90), abs(ga - 90))
        out[tag] = {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4),
                    "alpha": round(al, 4), "beta": round(be, 4), "gamma": round(ga, 4),
                    "max_angle_deviation_deg": round(dev, 4),
                    "c_over_a": round(c / a, 5),
                    "volume_A3": round(float(at.get_volume()), 4),
                    "E_meV_per_atom": round(at.get_potential_energy() / len(at) * 1000, 6),
                    "converged": conv}
    r = out["MnO_AFM2_rhomb"]
    # DFT keeps 0.89 deg; call the distortion "retained" if at least a third of it survives
    out["_retains_rhombohedral"] = bool(r["max_angle_deviation_deg"] > 0.30)
    out["_alpha_DFT"] = DFT["alpha_AFM2_deg"]
    out["_verdict"] = ("RETAINS the rhombohedral distortion"
                       if out["_retains_rhombohedral"] else
                       "COLLAPSES to cubic - magnetostructural coupling absent")
    return out


def cubic_eos(calc, lo=4.30, hi=4.60, n=13):
    """(3) Energy vs a on a cubic lattice. Ions are symmetry-fixed, so this is exact."""
    base = read(SDIR / "MnO_AFM2_cubic_seed.extxyz")
    xs, ys = [], []
    for a in np.linspace(lo, hi, n):
        at = base.copy()
        at.set_cell(Cell.fromcellpar([a, a, a, 90, 90, 90]), scale_atoms=True)
        at.calc = calc
        xs.append(float(a))
        ys.append(at.get_potential_energy() / len(at) * 1000.0)
    x, y = np.array(xs), np.array(ys)
    k = int(y.argmin())
    # parabola through the three points bracketing the minimum
    if 0 < k < len(x) - 1:
        c2, c1, c0 = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2)
        a_eq = float(-c1 / (2 * c2))
        e_eq = float(c0 + c1 * a_eq + c2 * a_eq ** 2)
    else:
        a_eq, e_eq = float(x[k]), float(y[k])
    return {"a_grid": [round(v, 4) for v in xs],
            "E_meV_per_atom": [round(v, 6) for v in ys],
            "a_eq_cubic_A": round(a_eq, 4), "E_eq_meV_per_atom": round(e_eq, 6),
            "a_eq_at_grid_edge": not (0 < k < len(x) - 1),
            "DFT_a_AFM2": DFT["a_AFM2_A"], "DFT_a_FM": DFT["a_FM_A"]}


def main():
    extra = "--extra" in sys.argv
    models = EXTRA if extra else BASE
    if "--only" in sys.argv:
        models = [sys.argv[sys.argv.index("--only") + 1]]
    out_path = RDIR / ("mno_results_extra.json" if extra else "mno_results.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out = {k: v for k, v in out.items() if "error" not in v}

    for name in models:
        print(f"=== {name} ===", flush=True)
        blk = out.setdefault(name, {})
        calc = None
        for key, fn in [("phase_probe", phase_probe),
                        ("symmetry_retention", symmetry_retention),
                        ("cubic_eos", cubic_eos)]:
            if key in blk and "error" not in blk[key]:
                print(f"  {key}: cached", flush=True)
                continue
            try:
                if calc is None:
                    calc = make_calc(name)
                blk[key] = fn(calc)
                if key == "phase_probe":
                    print(f"  probe:  ordering {blk[key]['ordering_FM_minus_AFM2']:+8.3f}"
                          f" (DFT {DFT['ordering_FM_minus_AFM2']:+.1f}) |"
                          f" distortion {blk[key]['distortion_cubic_minus_rhomb']:+7.4f}"
                          f" (DFT {DFT['distortion_cubic_minus_rhomb']:+.1f}) meV/atom",
                          flush=True)
                elif key == "symmetry_retention":
                    print(f"  relax:  alpha -> "
                          f"{blk[key]['MnO_AFM2_rhomb']['alpha']:.3f} deg"
                          f" (DFT {DFT['alpha_AFM2_deg']}) | {blk[key]['_verdict']}",
                          flush=True)
                else:
                    print(f"  eos:    a_eq(cubic) = {blk[key]['a_eq_cubic_A']:.4f} A"
                          f" (DFT AFM-II {DFT['a_AFM2_A']}, FM {DFT['a_FM_A']})", flush=True)
            except Exception:
                traceback.print_exc()
                blk[key] = {"error": traceback.format_exc(limit=2)}
            out_path.write_text(dump(out))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
