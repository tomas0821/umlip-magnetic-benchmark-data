#!/usr/bin/env python3
"""Provenance quantities: force noise, residual forces and the SevenNet-0 continuation test.

Computes quantities that previously had no backing script or results file and writes them to
results/phase1_provenance.json so every number is reproducible.

  (a) force_noise      - spread in FORCES under rigid rotation. A rotation-invariant potential
                         must return forces that rotate rigidly with the structure, so the
                         residual after rotating them back is purely numerical. This is the
                         force analogue of the energy noise floor, and it is what bounds how
                         tightly any relaxation with that model can converge.
  (b) residual_forces  - max |F| on the released relaxed geometries, so "did not reach
                         0.01 eV/A" can be quoted with the actual value.
  (c) e2_continuation  - SevenNet-0 orthorhombic V_Cr relaxed at successively tighter fmax,
                         the evidence for exclusion rule E2 (energy still falling => the
                         formation energy is not a well-defined quantity in that host).

Run with ~/.venvs/umlip/bin/python (5 models); pass --extra and run with ~/.venvs/umlip2 for
Orb-v3 and GRACE-2L-OMAT.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from ase.optimize import FIRE, LBFGS

_TL = torch.load
torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _TL(*a, **k))[1]

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
BASE_MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]
EXTRA_MODELS = ["Orb-v3", "GRACE-2L-OMAT"]


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


def force_noise(calc, atoms, n=5, seed=0):
    """Max |F_rotated_back - F_reference| over random rigid rotations (eV/A)."""
    rng = np.random.default_rng(seed)
    a0 = atoms.copy(); a0.calc = calc
    F0 = a0.get_forces()
    cell0 = np.array(atoms.get_cell())
    devs = []
    for _ in range(n):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        a = atoms.copy()
        a.rotate(float(rng.uniform(0, 360)), v, rotate_cell=True)
        a.calc = calc
        R = np.array(a.get_cell()).T @ np.linalg.inv(cell0.T)
        devs.append(float(np.abs(a.get_forces() @ np.linalg.inv(R).T - F0).max()))
    return max(devs), devs


def main():
    extra = "--extra" in sys.argv
    models = EXTRA_MODELS if extra else BASE_MODELS
    out_path = RDIR / ("phase1_provenance_extra.json" if extra else "phase1_provenance.json")
    out = {}

    for name in models:
        print(f"=== {name} ===", flush=True)
        try:
            calc = make_calc(name)
            blk = {}
            # (a) force noise on the pristine cubic conventional cell and on a defect cell
            for tag, f in [("cubic_conv", SDIR / "CrN_cubic_conv.extxyz"),
                           ("cubic_2x2x2_V_N", SDIR / "CrN_cubic_2x2x2_V_N.extxyz")]:
                mx, all_ = force_noise(calc, read(f))
                blk[f"force_noise_{tag}_eV_per_A"] = round(mx, 5)
            print(f"  force noise: {blk}", flush=True)

            # (b) residual forces on the released relaxed geometries
            rel_dir = (RDIR / "relaxed_extra" / name if extra else
                       (RDIR / "relaxed_batch1" / name if name == "M3GNet-MatPES"
                        else RDIR / "relaxed" / name))
            resid = {}
            for tag in ["cubic", "sheared"]:
                for d in ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]:
                    p = rel_dir / f"CrN_{tag}_2x2x2_{d}.extxyz"
                    if not p.exists():
                        continue
                    at = read(p); at.calc = calc
                    resid[f"{tag}|{d}"] = round(float(
                        np.linalg.norm(at.get_forces(), axis=1).max()), 5)
            blk["residual_fmax_eV_per_A"] = resid
            worst = max(resid, key=resid.get) if resid else None
            if worst:
                print(f"  residual fmax worst: {worst} = {resid[worst]}", flush=True)
            out[name] = blk
        except Exception:
            traceback.print_exc()
            out[name] = {"error": traceback.format_exc(limit=2)}
        out_path.write_text(json.dumps(out, indent=1))

    # (c) E2 continuation - base env only, SevenNet
    if not extra:
        print("=== E2 continuation (SevenNet-0 orthorhombic V_Cr) ===", flush=True)
        try:
            calc = make_calc("SevenNet-0")
            mu = json.loads((RDIR / "stage3_results.json").read_text())
            mu = next(m for m in mu if m["model"] == "SevenNet-0")["reservoirs"]
            ideal_d = read(SDIR / "CrN_sheared_2x2x2_V_Cr.extxyz")
            ideal_p = read(SDIR / "CrN_sheared_2x2x2_pristine.extxyz")
            series = {}
            for fmax in [1e-2, 3e-3, 1e-3]:
                p = ideal_p.copy(); p.calc = calc
                o = FIRE(p, logfile=None); o.run(fmax=fmax, steps=6000)
                if not o.converged():
                    LBFGS(p, logfile=None).run(fmax=fmax, steps=6000)
                d = ideal_d.copy(); d.calc = calc
                o = FIRE(d, logfile=None); o.run(fmax=fmax, steps=6000)
                conv = o.converged()
                if not conv:
                    o2 = LBFGS(d, logfile=None); o2.run(fmax=fmax, steps=6000)
                    conv = o2.converged()
                cell = d.get_cell()
                dd = d.get_positions() - ideal_d.get_positions()
                fr = dd @ np.linalg.inv(cell); fr -= np.round(fr)
                maxdisp = float(np.linalg.norm(fr @ cell, axis=1).max())
                ef = d.get_potential_energy() - p.get_potential_energy() + mu["mu_Cr"]
                series[f"{fmax:g}"] = {"E_defect": round(float(d.get_potential_energy()), 4),
                                       "E_pristine": round(float(p.get_potential_energy()), 4),
                                       "E_f_eV": round(float(ef), 4),
                                       "max_disp_A": round(maxdisp, 4),
                                       "converged": bool(conv)}
                print(f"  fmax={fmax:g}: E_f={ef:.4f} max|d|={maxdisp:.3f} conv={conv}", flush=True)
            out["_e2_continuation_SevenNet_sheared_V_Cr"] = series
            evs = [v["E_f_eV"] for v in series.values()]
            out["_e2_still_falling"] = bool(evs[-1] < evs[-2] - 1e-3)
        except Exception:
            traceback.print_exc()
        out_path.write_text(json.dumps(out, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
