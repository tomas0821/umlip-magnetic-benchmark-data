#!/usr/bin/env python3
"""Batch 1: M3GNet rerun, force-tolerance sensitivity, numerical noise floors, E(gamma) scan.

(A) rerun_m3gnet. M3GNet's reservoirs, phase full-relax and four defect cells
    carried converged:false. Rerun every M3GNet relaxation with a 10x step budget and an
    LBFGS fallback after FIRE; record convergence for the pristine cells too (the original
    driver discarded that flag even though E_pristine enters every formation energy).

(B) fmax_sensitivity. The reference used 1e-4 a.u. = 5.1e-3 eV/A; production runs
    used 1e-2 eV/A. Re-relax the cases whose conclusions sit near that resolution at
    3e-3 eV/A and report the shift.

(C) numerical_precision. Separates *numerical* precision
    of the energy evaluation from physical PES structure, by evaluating the SAME cell under
    operations that cannot change the energy of an O(3)-invariant potential:
      repeat   : identical input, 5x            -> nondeterminism
      rotate   : 5 random proper rotations      -> graph/numerics noise
      translate: 5 random rigid translations    -> PBC/graph-construction noise
      mirror   : x -> -x                        -> the E(gamma) test's own operation
    The max spread over these is the model's numerical noise floor: a hard lower bound on
    any energy difference it can resolve. This is what the E(gamma) asymmetry measures: mirror
    symmetry is architecturally guaranteed, so any nonzero asymmetry is numerical.

(D) gamma_scan. The E(gamma) scan generator.
    Cell construction: conventional 8-atom rocksalt cell at the published a_c = 4.14 A,
    vectors a1 = a(1,0,0), a2 = a(cos g, sin g, 0), a3 = a(0,0,1), atoms held at fixed
    fractional coordinates. Edge lengths are therefore gamma-independent and V proportional
    to sin(gamma), so the gamma and 180-gamma cells are exact mirror images (x -> -x) with
    identical volume: an O(3)-invariant potential must return identical energies.

Writes results/batch1_{m3gnet_rerun,fmax_sensitivity,numerical_precision,gamma_scan}.json
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE, LBFGS

# e3nn's constants.pt fails to load under torch>=2.6's weights_only=True default unless
# mace-torch happens to have registered its safe globals first, which makes model-loading
# order-dependent. These are local, trusted checkpoints; pin the old behaviour explicitly.
_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
A_C = 4.14
DEFECTS = ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]
DEFECT_MU = {"V_N": (1, 0), "V_Cr": (0, 1), "A_N": (-1, 1), "A_Cr": (1, -1), "I_N": (-1, 0)}


def make_calc(name):
    import torch
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
    raise ValueError(name)


def relax_hard(atoms, cell=False, fmax=0.01, steps=2000):
    """FIRE, then LBFGS fallback if unconverged. Returns (converged, n_steps, optimizer)."""
    target = FrechetCellFilter(atoms) if cell else atoms
    opt = FIRE(target, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    if opt.converged():
        return True, opt.get_number_of_steps(), "FIRE"
    n1 = opt.get_number_of_steps()
    opt2 = LBFGS(target, logfile=None)
    opt2.run(fmax=fmax, steps=steps)
    return bool(opt2.converged()), n1 + opt2.get_number_of_steps(), "FIRE+LBFGS"


# ---------------------------------------------------------------- (A)
def rerun_m3gnet():
    name = "M3GNet-MatPES"
    calc = make_calc(name)
    out = {"model": name, "fmax": 0.01, "steps_cap": 2000}
    rel_dir = RDIR / "relaxed_batch1" / name
    rel_dir.mkdir(parents=True, exist_ok=True)

    def rx(fname, cell=False):
        at = read(SDIR / f"{fname}.extxyz")
        at.calc = calc
        conv, nsteps, how = relax_hard(at, cell=cell)
        write(rel_dir / f"{fname}.extxyz", at)
        return at.get_potential_energy(), conv, nsteps, how

    e_n2, c1, s1, h1 = rx("N2_molecule")
    e_cr, c2, s2, h2 = rx("Cr_bcc", cell=True)
    mu_n, mu_cr = e_n2 / 2, e_cr / 2
    out["reservoirs"] = {"E_N2": e_n2, "mu_N": mu_n, "mu_Cr": mu_cr,
                         "N2_converged": c1, "N2_steps": s1, "N2_opt": h1,
                         "Cr_converged": c2, "Cr_steps": s2, "Cr_opt": h2}
    print(f"  reservoirs: N2 conv={c1} ({s1} steps, {h1}); Cr conv={c2} ({s2}, {h2})", flush=True)

    at = read(SDIR / "CrN_sheared_conv.extxyz"); at.calc = calc
    conv, nsteps, how = relax_hard(at, cell=True)
    write(rel_dir / "CrN_sheared_fullrelax.extxyz", at)
    out["phase_fullrelax"] = {"angles_deg": [float(x) for x in np.round(at.get_cell().angles(), 3)],
                              "converged": conv, "steps": nsteps, "opt": how}
    print(f"  phase fullrelax: {out['phase_fullrelax']}", flush=True)

    out["defects"] = {}
    for tag in ["cubic", "sheared"]:
        e_prist, cp, sp, hp = rx(f"CrN_{tag}_2x2x2_pristine")
        blk = {"E_pristine": e_prist, "pristine_converged": cp,
               "pristine_steps": sp, "pristine_opt": hp}
        for d in DEFECTS:
            e_def, cd, sd, hd = rx(f"CrN_{tag}_2x2x2_{d}")
            n_n, n_cr = DEFECT_MU[d]
            blk[d] = {"E": e_def, "E_f_eV": e_def - e_prist + n_n * mu_n + n_cr * mu_cr,
                      "converged": cd, "steps": sd, "opt": hd}
            print(f"  {tag} {d}: E_f={blk[d]['E_f_eV']:.3f} conv={cd} ({sd} steps, {hd})", flush=True)
        out["defects"][tag] = blk
    return out


# ---------------------------------------------------------------- (B)
SENSITIVE = [
    ("SevenNet-0", "sheared", "V_Cr"),   # 0.02 eV, the smallest defect energy in the set
    ("SevenNet-0", "cubic", "V_N"),      # 0.31 eV
    ("MACE-MP-0", "cubic", "V_N"),       # 0.21 eV, drives the Cr-rich negative value
    ("MACE-MP-0", "cubic", "V_Cr"),      # 0.46 eV
    ("CHGNet", "cubic", "V_N"),          # 1.10 eV, Cr-rich −0.11 eV
    ("MatterSim", "cubic", "V_Cr"),      # 0.22 eV — the 30x headline
]


def fmax_sensitivity():
    tight = 0.003
    res = {"fmax_production": 0.01, "fmax_tight": tight, "cases": {}}
    by_model = {}
    for m, tag, d in SENSITIVE:
        by_model.setdefault(m, []).append((tag, d))
    for m, cases in by_model.items():
        calc = make_calc(m)
        for tag, d in cases:
            at = read(SDIR / f"CrN_{tag}_2x2x2_pristine.extxyz"); at.calc = calc
            conv_p, _, _ = relax_hard(at, fmax=tight, steps=3000)
            e_p = at.get_potential_energy()
            ad = read(SDIR / f"CrN_{tag}_2x2x2_{d}.extxyz"); ad.calc = calc
            conv_d, nst, how = relax_hard(ad, fmax=tight, steps=3000)
            e_d = ad.get_potential_energy()
            n2 = read(SDIR / "N2_molecule.extxyz"); n2.calc = calc
            relax_hard(n2, fmax=tight, steps=3000)
            mu_n = n2.get_potential_energy() / 2
            cr = read(SDIR / "Cr_bcc.extxyz"); cr.calc = calc
            relax_hard(cr, cell=True, fmax=tight, steps=3000)
            mu_cr = cr.get_potential_energy() / 2
            n_n, n_cr = DEFECT_MU[d]
            ef = e_d - e_p + n_n * mu_n + n_cr * mu_cr
            key = f"{m}|{tag}|{d}"
            res["cases"][key] = {"E_f_tight": float(ef), "converged": bool(conv_d and conv_p),
                                 "steps": nst, "opt": how}
            print(f"  {key}: E_f(tight)={ef:.4f} conv={conv_d and conv_p}", flush=True)
    return res


# ---------------------------------------------------------------- (C)
def numerical_precision():
    rng = np.random.default_rng(0)
    base = read(SDIR / "CrN_cubic_conv.extxyz")
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")
    out = {}
    for name in ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]:
        calc = make_calc(name)

        def E(at):
            a = at.copy(); a.calc = calc
            return a.get_potential_energy()

        e0 = E(base)
        repeat = [E(base) for _ in range(5)]
        rots = []
        for _ in range(5):
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            ang = float(rng.uniform(0, 360))
            a = base.copy(); a.rotate(ang, v, rotate_cell=True)
            rots.append(E(a))
        trans = []
        for _ in range(5):
            a = base.copy()
            a.translate(rng.uniform(-3, 3, size=3))
            a.wrap()
            trans.append(E(a))
        mir = base.copy()
        cell = np.array(mir.get_cell()); pos = mir.get_positions()
        M = np.diag([-1.0, 1.0, 1.0])
        mir.set_cell(cell @ M, scale_atoms=False); mir.set_positions(pos @ M)
        mir.wrap()
        e_mir = E(mir)

        def spread(vals):
            return (max(vals) - min(vals)) / n_fu * 1000  # meV/f.u.

        out[name] = {
            "E0_eV": e0,
            "repeat_spread_meV_per_fu": spread(repeat),
            "rotation_spread_meV_per_fu": spread(rots + [e0]),
            "translation_spread_meV_per_fu": spread(trans + [e0]),
            "mirror_diff_meV_per_fu": abs(e_mir - e0) / n_fu * 1000,
        }
        out[name]["noise_floor_meV_per_fu"] = max(
            out[name]["repeat_spread_meV_per_fu"],
            out[name]["rotation_spread_meV_per_fu"],
            out[name]["translation_spread_meV_per_fu"],
            out[name]["mirror_diff_meV_per_fu"])
        print(f"  {name}: floor = {out[name]['noise_floor_meV_per_fu']:.4f} meV/f.u. "
              f"(rep {out[name]['repeat_spread_meV_per_fu']:.4f}, "
              f"rot {out[name]['rotation_spread_meV_per_fu']:.4f}, "
              f"trans {out[name]['translation_spread_meV_per_fu']:.4f}, "
              f"mir {out[name]['mirror_diff_meV_per_fu']:.4f})", flush=True)
    return out


# ---------------------------------------------------------------- (D)
def gamma_scan():
    base = read(SDIR / "CrN_cubic_conv.extxyz")
    a = A_C
    n_fu = sum(1 for s in base.get_chemical_symbols() if s == "Cr")
    angles = list(np.arange(82, 99, 2.0))
    out = {"angles_deg": [float(g) for g in angles], "a_c_A": a,
           "cell_construction": "a1=a(1,0,0), a2=a(cos g, sin g, 0), a3=a(0,0,1); "
                                "fixed fractional coordinates; edges gamma-independent; "
                                "V ∝ sin(gamma); gamma and 180-gamma are exact mirror pairs",
           "single_point": True, "dE_meV_per_fu_vs_90": {}, "mirror_asymmetry_meV_per_fu": {}}
    for name in ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]:
        calc = make_calc(name)
        es = []
        for g in angles:
            at = base.copy()
            rad = np.radians(g)
            at.set_cell(np.array([[a, 0, 0], [a * np.cos(rad), a * np.sin(rad), 0], [0, 0, a]]),
                        scale_atoms=True)
            at.calc = calc
            es.append(at.get_potential_energy())
        es = np.array(es)
        i90 = angles.index(90.0)
        rel = (es - es[i90]) / n_fu * 1000
        out["dE_meV_per_fu_vs_90"][name] = [float(round(x, 4)) for x in rel]
        asym = max(abs(rel[i] - rel[len(rel) - 1 - i]) for i in range(len(rel) // 2))
        out["mirror_asymmetry_meV_per_fu"][name] = float(round(asym, 4))
        print(f"  {name}: mirror asymmetry = {asym:.4f} meV/f.u.", flush=True)
    return out


SECTIONS = {"m3gnet_rerun": rerun_m3gnet, "fmax_sensitivity": fmax_sensitivity,
            "numerical_precision": numerical_precision, "gamma_scan": gamma_scan}


def main():
    todo = sys.argv[1:] or list(SECTIONS)
    for key in todo:
        path = RDIR / f"batch1_{key}.json"
        print(f"=== {key} ===", flush=True)
        try:
            res = SECTIONS[key]()
            path.write_text(json.dumps(res, indent=1, default=float))
            print(f"OK {key} -> {path.name}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {key}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
