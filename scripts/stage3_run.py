#!/usr/bin/env python3
"""Stage 3 — inference runs: all models x all CrN structures.

Per model:
  reservoirs   : N2 (bond relax), bcc Cr (full cell relax)  -> mu_N, mu_Cr
  phase probe  : cubic + sheared conventional cells
                 A: single-point at DFT geometry (a_c = 4.14 A, alpha = 88 deg)
                 B: internal-coordinate relax at fixed cell
                 C: full relax from the sheared cell (does the distortion survive?)
  defect probe : 2x2x2 supercells, position-only relax at fixed cell (paper protocol),
                 formation energies vs mu_N = E(N2)/2, mu_Cr = E(Cr_bcc)/atom
                 (neutral defects; paper Eq. 1 with q = 0)

Outputs: results/stage3_results.json (energies + derived numbers)
         results/relaxed/<model>/<structure>.extxyz (for displacement analysis later)
Force convergence: fmax = 0.01 eV/A (paper: 1e-4 a.u. ~ 0.003 eV/A; noted in Methods).
"""
import json
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
FMAX = 0.01
DEFECTS = ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]
# formation-energy bookkeeping: E_f = E_def - E_prist + n_N*mu_N + n_Cr*mu_Cr
DEFECT_MU = {"V_N": (1, 0), "V_Cr": (0, 1), "A_N": (-1, 1), "A_Cr": (1, -1), "I_N": (-1, 0)}


def relax(atoms, cell=False, fmax=FMAX, steps=500):
    target = FrechetCellFilter(atoms) if cell else atoms
    opt = FIRE(target, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms, bool(opt.converged())


def get_calculators():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mace.calculators import mace_mp
    yield "MACE-MP-0", lambda: mace_mp(model="medium", device=device, default_dtype="float64")
    from chgnet.model.dynamics import CHGNetCalculator
    yield "CHGNet", lambda: CHGNetCalculator(use_device=device)
    from sevenn.calculator import SevenNetCalculator
    yield "SevenNet-0", lambda: SevenNetCalculator(model="7net-0", device=device)
    import matgl
    from matgl.ext.ase import PESCalculator
    yield "M3GNet-MatPES", lambda: PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2", revision="8414b23e68b173d6e36271e4366ee3717364051c"))
    from mattersim.forcefield import MatterSimCalculator
    yield "MatterSim", lambda: MatterSimCalculator(device=device)


def run_model(name, make_calc):
    out = {"model": name}
    rel_dir = RDIR / "relaxed" / name
    rel_dir.mkdir(parents=True, exist_ok=True)

    def sp(fname):
        at = read(SDIR / f"{fname}.extxyz")
        at.calc = make_calc()
        return at.get_potential_energy()

    def rx(fname, cell=False):
        at = read(SDIR / f"{fname}.extxyz")
        at.calc = make_calc()
        at, conv = relax(at, cell=cell)
        write(rel_dir / f"{fname}.extxyz", at)
        return at.get_potential_energy(), conv

    # reservoirs
    e_n2, c1 = rx("N2_molecule")
    e_cr, c2 = rx("Cr_bcc", cell=True)
    mu_n, mu_cr = e_n2 / 2, e_cr / 2  # Cr_bcc cell has 2 atoms
    out["reservoirs"] = {"E_N2": e_n2, "mu_N": mu_n, "mu_Cr": mu_cr,
                         "converged": c1 and c2}

    # phase probe (8-atom cells, 4 f.u.)
    a_cub_sp = sp("CrN_cubic_conv")
    a_she_sp = sp("CrN_sheared_conv")
    # B: positions-only relax at fixed cell
    at = read(SDIR / "CrN_cubic_conv.extxyz"); at.calc = make_calc()
    at, _ = relax(at); e_cub_posrx = at.get_potential_energy()
    at = read(SDIR / "CrN_sheared_conv.extxyz"); at.calc = make_calc()
    at, _ = relax(at); e_she_posrx = at.get_potential_energy()
    # C: full relax from sheared start
    at = read(SDIR / "CrN_sheared_conv.extxyz"); at.calc = make_calc()
    at, conv = relax(at, cell=True)
    write(rel_dir / "CrN_sheared_fullrelax.extxyz", at)
    cell = at.get_cell()
    ang = cell.angles()
    out["phase"] = {
        "E_cubic_sp": a_cub_sp, "E_sheared_sp": a_she_sp,
        "dE_sp_meV_per_fu": (a_she_sp - a_cub_sp) / 4 * 1000,
        "E_cubic_posrelax": e_cub_posrx, "E_sheared_posrelax": e_she_posrx,
        "dE_posrelax_meV_per_fu": (e_she_posrx - e_cub_posrx) / 4 * 1000,
        "fullrelax_from_sheared_angles_deg": list(np.round(ang, 3)),
        "fullrelax_converged": conv,
    }

    # defect probe, both phase geometries
    out["defects"] = {}
    for tag in ["cubic", "sheared"]:
        e_prist, _ = rx(f"CrN_{tag}_2x2x2_pristine")
        block = {"E_pristine": e_prist}
        for d in DEFECTS:
            e_def, conv = rx(f"CrN_{tag}_2x2x2_{d}")
            n_n, n_cr = DEFECT_MU[d]
            ef = e_def - e_prist + n_n * mu_n + n_cr * mu_cr
            block[d] = {"E": e_def, "E_f_eV": ef, "converged": conv}
        out["defects"][tag] = block
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def main():
    RDIR.mkdir(exist_ok=True)
    out_path = RDIR / "stage3_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["model"] for r in results if "error" not in r}
    for name, make_calc in get_calculators():
        if name in done:
            print(f"SKIP {name} (already done)", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        try:
            results = [r for r in results if r["model"] != name]
            results.append(run_model(name, make_calc))
            print(f"OK {name}", flush=True)
        except Exception:
            traceback.print_exc()
            results.append({"model": name, "error": traceback.format_exc(limit=3)})
        out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
