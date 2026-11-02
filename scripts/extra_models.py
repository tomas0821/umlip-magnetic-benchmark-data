#!/usr/bin/env python3
"""Extend the roster beyond the original five models.

Runs the full CrN protocol (phase probe, defects in both phases, elastic constants,
numerical-resolution test, E(gamma) scan) for models that live in a separate environment
(~/.venvs/umlip2) so that the pinned five-model environment stays reproducible.

Usage:  ~/.venvs/umlip2/bin/python scripts/extra_models.py [model ...]
Writes results/extra_<model>.json
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE, LBFGS

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
A_C = 4.14
EV_A3_TO_GPA = 160.21766
DEFECTS = ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]
DEFECT_MU = {"V_N": (1, 0), "V_Cr": (0, 1), "A_N": (-1, 1), "A_Cr": (1, -1), "I_N": (-1, 0)}


def make_calc(name):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("Orb"):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        ff = pretrained.orb_v3_conservative_inf_omat(device=dev, precision="float32-high")
        return ORBCalculator(ff, device=dev)
    if name.startswith("GRACE"):
        from tensorpotential.calculator import grace_fm
        return grace_fm("GRACE-2L-OMAT")
    raise ValueError(name)


def relax(atoms, calc, cell=False, fmax=0.01, steps=2000):
    atoms.calc = calc
    tgt = FrechetCellFilter(atoms) if cell else atoms
    o = FIRE(tgt, logfile=None)
    o.run(fmax=fmax, steps=steps)
    if o.converged():
        return True
    o2 = LBFGS(tgt, logfile=None)
    o2.run(fmax=fmax, steps=steps)
    return bool(o2.converged())


def run(name):
    calc = make_calc(name)
    out = {"model": name}
    rel = RDIR / "relaxed_extra" / name
    rel.mkdir(parents=True, exist_ok=True)
    base = read(SDIR / "CrN_cubic_conv.extxyz")
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")

    def E(at):
        a = at.copy(); a.calc = calc
        return a.get_potential_energy()

    # --- numerical resolution ---
    rng = np.random.default_rng(0)
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
    sp = lambda v: (max(v) - min(v)) / n_fu * 1000
    out["numerical_precision"] = {
        "repeat": sp(rep), "rotation": sp(rots + [e0]), "translation": sp(trs + [e0]),
        "mirror": abs(E(m) - e0) / n_fu * 1000}
    out["numerical_precision"]["noise_floor_meV_per_fu"] = max(
        out["numerical_precision"].values())
    print(f"  noise floor: {out['numerical_precision']['noise_floor_meV_per_fu']:.4f} meV/f.u.",
          flush=True)

    # --- phase probe ---
    cub, she = read(SDIR / "CrN_cubic_conv.extxyz"), read(SDIR / "CrN_sheared_conv.extxyz")
    e_c_sp, e_s_sp = E(cub), E(she)
    a1 = cub.copy(); relax(a1, calc); e_c_rx = a1.get_potential_energy()
    a2 = she.copy(); relax(a2, calc); e_s_rx = a2.get_potential_energy()
    a3 = she.copy(); conv = relax(a3, calc, cell=True)
    write(rel / "CrN_sheared_fullrelax.extxyz", a3)
    out["phase"] = {"dE_sp_meV_per_fu": (e_s_sp - e_c_sp) / n_fu * 1000,
                    "dE_posrelax_meV_per_fu": (e_s_rx - e_c_rx) / n_fu * 1000,
                    "fullrelax_angles_deg": [float(x) for x in np.round(a3.get_cell().angles(), 3)],
                    "fullrelax_converged": conv}
    print(f"  phase: dE_sp={out['phase']['dE_sp_meV_per_fu']:+.2f} "
          f"angles={out['phase']['fullrelax_angles_deg']}", flush=True)

    # --- gamma scan ---
    angles = list(np.arange(82, 99, 2.0))
    es = []
    for g in angles:
        at = base.copy(); r = np.radians(g)
        at.set_cell(np.array([[A_C, 0, 0], [A_C*np.cos(r), A_C*np.sin(r), 0], [0, 0, A_C]]),
                    scale_atoms=True)
        es.append(E(at))
    es = np.array(es); i90 = angles.index(90.0)
    relE = (es - es[i90]) / n_fu * 1000
    out["gamma_scan"] = {"angles_deg": [float(g) for g in angles],
                         "dE_meV_per_fu_vs_90": [float(round(x, 4)) for x in relE],
                         "mirror_asymmetry_meV_per_fu": float(round(
                             max(abs(relE[i] - relE[len(relE)-1-i]) for i in range(len(relE)//2)), 4))}

    # --- reservoirs + defects ---
    n2 = read(SDIR / "N2_molecule.extxyz"); relax(n2, calc); mu_n = n2.get_potential_energy()/2
    cr = read(SDIR / "Cr_bcc.extxyz"); relax(cr, calc, cell=True); mu_cr = cr.get_potential_energy()/2
    out["reservoirs"] = {"mu_N": mu_n, "mu_Cr": mu_cr}
    out["defects"] = {}
    for tag in ["cubic", "sheared"]:
        pr = read(SDIR / f"CrN_{tag}_2x2x2_pristine.extxyz")
        cp = relax(pr, calc); e_p = pr.get_potential_energy()
        write(rel / f"CrN_{tag}_2x2x2_pristine.extxyz", pr)
        blk = {"E_pristine": e_p, "pristine_converged": cp}
        for d in DEFECTS:
            at = read(SDIR / f"CrN_{tag}_2x2x2_{d}.extxyz")
            cd = relax(at, calc); e_d = at.get_potential_energy()
            write(rel / f"CrN_{tag}_2x2x2_{d}.extxyz", at)
            nn, ncr = DEFECT_MU[d]
            blk[d] = {"E": e_d, "E_f_eV": e_d - e_p + nn*mu_n + ncr*mu_cr, "converged": cd}
            print(f"  {tag} {d}: E_f={blk[d]['E_f_eV']:.3f} conv={cd}", flush=True)
        out["defects"][tag] = blk

    # --- elastic ---
    a0 = base.cell[0, 0]
    def Ecell(cm):
        a = base.copy(); a.set_cell(cm, scale_atoms=True); a.calc = calc
        return a.get_potential_energy()
    sc = np.linspace(0.97, 1.03, 7)
    p = np.polyfit(sc, [Ecell(np.eye(3)*a0*x) for x in sc], 2)
    a_eq = a0 * (-p[1]/(2*p[0])); c0 = np.eye(3)*a_eq; v0 = a_eq**3; e_eq = Ecell(c0)
    vols, ev = [], []
    for x in np.linspace(0.98, 1.02, 5):
        vols.append(v0*x**3); ev.append(Ecell(c0*x))
    B = 2*np.polyfit(vols, ev, 2)[0]*v0*EV_A3_TO_GPA
    ds = [-0.02, -0.01, 0.01, 0.02]
    cp_ = np.polyfit(ds, [Ecell(c0 @ np.diag([1+d, 1-d, 1/(1-d*d)])) - e_eq for d in ds], 2)[0]/v0*EV_A3_TO_GPA
    c44 = 2*np.polyfit(ds, [Ecell(c0 @ np.array([[1, d/2, 0], [d/2, 1, 0], [0, 0, 4/(4-d*d)]])) - e_eq
                            for d in ds], 2)[0]/v0*EV_A3_TO_GPA
    out["elastic"] = {"a_eq_A": float(a_eq), "B_GPa": float(B),
                      "C11_GPa": float(B + 2*cp_/3), "C12_GPa": float(B - cp_/3),
                      "C44_GPa": float(c44)}
    print(f"  elastic: {out['elastic']}", flush=True)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["Orb-v3"]):
        print(f"=== {name} ===", flush=True)
        try:
            r = run(name)
            (RDIR / f"extra_{name}.json").write_text(json.dumps(r, indent=1, default=float))
            print(f"OK {name}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name}", flush=True)
    print("DONE", flush=True)
