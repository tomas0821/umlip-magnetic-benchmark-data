#!/usr/bin/env python3
"""Predictive validation of two of the three preflight diagnostics.

A diagnostic is only useful if firing it predicts a concrete failure. Two tests:

(A) shear_stability  -- the elastic-stability check says SevenNet-0 has C44 < 0, i.e. its
    cubic CrN is unstable to trigonal shear. PREDICTION: perturb pristine, defect-free cubic
    CrN slightly off 90 degrees and relax the cell freely; models with C44 > 0 must return to
    90 degrees, SevenNet-0 must run away. No defect, no magnetism -- just an unstable lattice
    that would silently corrupt any simulation run with that model.

(B) mev_resolution   -- the resolution check says M3GNet-MatPES carries a 2.0 meV/f.u. noise
    floor. PREDICTION: rattle pristine CrN by a tiny amount and relax back; a model with a
    clean surface returns the same energy every time, one at its noise floor scatters. This
    is a task with no magnetism in it at all, so it tests the diagnostic's general claim.

Run with ~/.venvs/umlip/bin/python (5 models); --extra with ~/.venvs/umlip2 for Orb/GRACE.
Writes results/diagnostic_validation{,_extra}.json
"""
import json
import sys
import traceback
from pathlib import Path

import os

# Cap threading BEFORE torch/numpy import. The WSL2 VM has 7 GB and 12 cores; letting
# torch+OpenMP take every core while a 64-atom relaxation runs froze the VM (2026-08-03).
# Small-cell CPU inference does not scale past ~4 threads anyway.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import torch

torch.set_num_threads(4)
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE, LBFGS

_TL = torch.load
torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _TL(*a, **k))[1]

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
BASE = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]
EXTRA = ["Orb-v3", "GRACE-2L-OMAT"]
PERTURB_DEG = 1.0     # small shear away from cubic; a stable lattice must undo it


def make_calc(name):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "MACE-MP-0":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device=dev, default_dtype="float64")
    if name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=dev)
    if name == "SevenNet-0":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model="7net-0", device=dev)
    if name == "M3GNet-MatPES":
        import matgl
        from matgl.ext.ase import PESCalculator
        return PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2", revision="8414b23e68b173d6e36271e4366ee3717364051c"))
    if name == "MatterSim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=dev)
    if name == "Orb-v3":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        return ORBCalculator(pretrained.orb_v3_conservative_inf_omat(
            device=dev, precision="float32-high"), device=dev)
    if name == "GRACE-2L-OMAT":
        from tensorpotential.calculator import grace_fm
        return grace_fm("GRACE-2L-OMAT")
    raise ValueError(name)


def relax(atoms, calc, cell=False, fmax=0.01, steps=1500):
    atoms.calc = calc
    tgt = FrechetCellFilter(atoms) if cell else atoms
    o = FIRE(tgt, logfile=None)
    o.run(fmax=fmax, steps=steps)
    if not o.converged():
        LBFGS(tgt, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def sheared_cell(a, gamma_deg):
    r = np.radians(gamma_deg)
    return np.array([[a, 0, 0], [a * np.cos(r), a * np.sin(r), 0], [0, 0, a]])


def shear_stability(name, calc, a0):
    """(A) Does the cubic lattice survive a small shear perturbation?"""
    base = read(SDIR / "CrN_cubic_conv.extxyz")
    out = {}
    for tag, g0 in [("control_90deg", 90.0), (f"perturbed_{90-PERTURB_DEG:g}deg", 90.0 - PERTURB_DEG),
                    (f"perturbed_{90+PERTURB_DEG:g}deg", 90.0 + PERTURB_DEG)]:
        at = base.copy()
        at.set_cell(sheared_cell(a0, g0), scale_atoms=True)
        relax(at, calc, cell=True)
        ang = [float(x) for x in np.round(at.get_cell().angles(), 3)]
        # the shear angle of interest is the one between a1 and a2 -> angles()[2] (gamma)
        out[tag] = {"start_gamma": g0, "final_angles": ang, "final_gamma": ang[2],
                    "returned_to_cubic": bool(abs(ang[2] - 90.0) < 0.5),
                    "final_volume_A3": float(at.get_volume())}
    pert = [v for k, v in out.items() if k.startswith("perturbed")]
    out["_verdict"] = ("STABLE - undoes the perturbation" if all(p["returned_to_cubic"] for p in pert)
                       else "UNSTABLE - runs away from cubic")
    out["_max_departure_deg"] = round(max(abs(p["final_gamma"] - 90.0) for p in pert), 3)
    return out


def mev_resolution(name, calc, n=8, amp=0.01, seed=0):
    """(B) Rattle pristine CrN by `amp` A and relax back; spread in the final energy is the
    resolution the model can actually deliver on a task with no magnetism in it."""
    rng = np.random.default_rng(seed)
    base = read(SDIR / "CrN_cubic_2x2x2_pristine.extxyz")
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")
    energies = []
    for _ in range(n):
        at = base.copy()
        at.positions += rng.normal(scale=amp, size=at.positions.shape)
        relax(at, calc, cell=False)
        energies.append(float(at.get_potential_energy()))
    e = np.array(energies)
    return {"n_trials": n, "rattle_amplitude_A": amp,
            "energies_eV": [round(x, 6) for x in energies],
            "spread_meV_per_fu": round(float(np.ptp(e)) / n_fu * 1000, 4),
            "std_meV_per_fu": round(float(e.std()) / n_fu * 1000, 4)}


def main():
    extra = "--extra" in sys.argv
    models = EXTRA if extra else BASE
    # --only MODEL: run a single model per process so its memory is fully released on exit.
    # The driver loop in run_validation.sh calls this once per model; results accumulate in
    # the shared JSON via the resume logic below.
    if "--only" in sys.argv:
        models = [sys.argv[sys.argv.index("--only") + 1]]
    claims = json.loads((RDIR / "claims.json").read_text())
    out_path = RDIR / ("diagnostic_validation_extra.json" if extra else "diagnostic_validation.json")
    # resume: keep any block already completed in an earlier run
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out = {k: v for k, v in out.items() if "error" not in v}
    for name in models:
        print(f"=== {name} ===", flush=True)
        blk = out.get(name, {})
        try:
            a0 = claims["elastic"]["per_model"][name]["a_eq_A"]
            c44 = claims["elastic"]["per_model"][name]["C44_GPa"]
            floor = claims["noise"]["floor_meV_per_fu"][name]
            blk.update({"a_eq_A": a0, "C44_GPa": c44, "noise_floor_meV_per_fu": floor,
                        "prediction_shear": "unstable" if c44 < 0 else "stable",
                        "prediction_resolution": "scatters" if floor > 0.08 else "reproducible"})
        except Exception:
            traceback.print_exc()
            out[name] = {"error": traceback.format_exc(limit=2)}
            out_path.write_text(json.dumps(out, indent=1))
            continue
        out[name] = blk
        calc = None
        # each sub-test fails independently: a crash in one must not discard the other
        for key, fn, report in [
            ("shear_stability", lambda c: shear_stability(name, c, blk["a_eq_A"]),
             lambda r: f"  shear:  C44={blk['C44_GPa']:+.0f} GPa -> predicted "
                       f"{blk['prediction_shear']:9s} | observed {r['_verdict']}"),
            ("mev_resolution", lambda c: mev_resolution(name, c),
             lambda r: f"  rattle: floor={blk['noise_floor_meV_per_fu']:.4f} -> predicted "
                       f"{blk['prediction_resolution']:12s} | observed spread "
                       f"{r['spread_meV_per_fu']} meV/f.u.")]:
            if key in blk:
                print(f"  {key}: cached", flush=True)
                continue
            try:
                if calc is None:
                    calc = make_calc(name)
                blk[key] = fn(calc)
                print(report(blk[key]), flush=True)
            except Exception:
                traceback.print_exc()
                blk[key] = {"error": traceback.format_exc(limit=2)}
            out_path.write_text(json.dumps(out, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
