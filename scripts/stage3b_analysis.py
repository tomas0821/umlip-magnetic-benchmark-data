#!/usr/bin/env python3
"""Stage 3b — the four remaining analyses on top of stage3_run.py output.

1. CHGNet magmoms on the relaxed geometries (pristine + defects, both phases).
   NOTE: CHGNet emits UNSIGNED site-moment magnitudes — it cannot produce the signed
   net supercell moments of the 2018 paper. We compare (a) pristine Cr |m| vs 2.90 mu_B,
   (b) defect-induced changes in the neighbor-shell moments. Protocol limitation -> paper.
2. Displacement analysis vs 2018 Table I. uMLIPs cannot distinguish Cr(up)/Cr(down),
   so the reference is the spin-channel average; we report the mean radial displacement
   (positive = away from defect) of the first Cr and first N shells.
3. Elastic constants of the cubic cell per model: B from E(V) fit; C' and C44 from
   volume-conserving tetragonal/trigonal strains (no internal relaxation; rocksalt
   symmetry forbids it for these strains). DFT ref: C11=677, C12=87.7, C44=155, B=289 GPa.
4. Reservoir cross-check: defect E_f under Cr-rich condition
   (mu_Cr = E(bcc)/atom, mu_N = E_fu(CrN) - mu_Cr) vs the N2-based values.

Output: results/stage3b_analysis.json
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"
RDIR = ROOT / "results"
EV_A3_TO_GPA = 160.21766
DEFECTS = ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]
MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim"]


def make_calc(name):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "MACE-MP-0":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device=device, default_dtype="float64")
    if name == "CHGNet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=device)
    if name == "SevenNet-0":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model="7net-0", device=device)
    if name == "M3GNet-MatPES":
        import matgl
        from matgl.ext.ase import PESCalculator
        return PESCalculator(matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2", revision="8414b23e68b173d6e36271e4366ee3717364051c"))
    if name == "MatterSim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=device)
    raise ValueError(name)


def defect_site(pristine, kind):
    """Ideal defect-site position + reference structure, mirroring build_structures.py."""
    syms = pristine.get_chemical_symbols()
    n0 = next(i for i, s in enumerate(syms) if s == "N")
    cr0 = next(i for i, s in enumerate(syms) if s == "Cr")
    ref = pristine.copy()
    if kind == "V_N":
        site = ref.positions[n0].copy(); del ref[n0]
    elif kind == "V_Cr":
        site = ref.positions[cr0].copy(); del ref[cr0]
    elif kind == "A_N":
        site = ref.positions[cr0].copy(); ref[cr0].symbol = "N"
    elif kind == "A_Cr":
        site = ref.positions[n0].copy(); ref[n0].symbol = "Cr"
    elif kind == "I_N":
        site = (np.array([0.125, 0.125, 0.125]) @ ref.get_cell())
        from ase import Atoms
        ref += Atoms("N", positions=[site])
    return site, ref


def shell_displacements(ideal, relaxed, site):
    """Mean radial displacement (away from site > 0) of first Cr and first N shells."""
    cell = ideal.get_cell()
    out = {}
    d_ideal = ideal.get_positions() - site
    # minimal image
    frac = d_ideal @ np.linalg.inv(cell)
    frac -= np.round(frac)
    d_ideal = frac @ cell
    r_ideal = np.linalg.norm(d_ideal, axis=1)
    disp = relaxed.get_positions() - ideal.get_positions()
    fd = disp @ np.linalg.inv(cell); fd -= np.round(fd); disp = fd @ cell
    for sp in ["Cr", "N"]:
        idx = [i for i, s in enumerate(ideal.get_chemical_symbols())
               if s == sp and r_ideal[i] > 0.1]
        if not idx:
            continue
        rmin = min(r_ideal[i] for i in idx)
        shell = [i for i in idx if r_ideal[i] < rmin + 0.3]
        radial = [float(np.dot(disp[i], d_ideal[i] / r_ideal[i])) for i in shell]
        out[sp] = {"n_shell": len(shell), "r_shell_A": round(float(rmin), 3),
                   "mean_radial_disp_A": round(float(np.mean(radial)), 4),
                   "spread_A": round(float(np.ptp(radial)), 4)}
    return out


def elastic_constants(name):
    calc = make_calc(name)
    # relax cubic cell isotropically first: scan E(a)
    base = read(SDIR / "CrN_cubic_conv.extxyz")
    def energy_at(cellmat, at=base):
        a = at.copy(); a.set_cell(cellmat, scale_atoms=True); a.calc = calc
        return a.get_potential_energy()
    a0 = base.cell[0, 0]
    scales = np.linspace(0.97, 1.03, 7)
    es = [energy_at(np.eye(3) * a0 * s) for s in scales]
    p = np.polyfit(scales, es, 2)
    s_min = -p[1] / (2 * p[0])
    a_eq = a0 * s_min
    cell0 = np.eye(3) * a_eq
    e0 = energy_at(cell0)
    v0 = a_eq ** 3
    # bulk modulus from E(V)
    vols, ev = [], []
    for s in np.linspace(0.98, 1.02, 5):
        c = cell0 * s
        vols.append(v0 * s ** 3); ev.append(energy_at(c))
    pv = np.polyfit(vols, ev, 2)
    B = 2 * pv[0] * v0 * EV_A3_TO_GPA
    # C' (tetragonal, volume conserving)
    ds = [-0.02, -0.01, 0.01, 0.02]
    de = []
    for d in ds:
        D = np.diag([1 + d, 1 - d, 1 / (1 - d * d)])
        de.append(energy_at(cell0 @ D) - e0)
    cp = np.polyfit(ds, de, 2)[0] / v0 * EV_A3_TO_GPA  # = C11 - C12
    # C44 (trigonal, volume conserving)
    de = []
    for d in ds:
        D = np.array([[1, d / 2, 0], [d / 2, 1, 0], [0, 0, 4 / (4 - d * d)]])
        de.append(energy_at(cell0 @ D) - e0)
    c44 = 2 * np.polyfit(ds, de, 2)[0] / v0 * EV_A3_TO_GPA
    c11 = B + 2 * cp / 3
    c12 = B - cp / 3
    return {"a_eq_A": round(float(a_eq), 4), "B_GPa": round(float(B), 1),
            "C11_GPa": round(float(c11), 1), "C12_GPa": round(float(c12), 1),
            "C44_GPa": round(float(c44), 1)}


def chgnet_magmoms():
    calc = make_calc("CHGNet")
    out = {}
    for tag in ["cubic", "sheared"]:
        prist = read(RDIR / "relaxed" / "CHGNet" / f"CrN_{tag}_2x2x2_pristine.extxyz")
        prist.calc = calc
        prist.get_potential_energy()
        mm = np.array(prist.calc.results["magmoms"])
        syms = prist.get_chemical_symbols()
        cr_m = mm[[i for i, s in enumerate(syms) if s == "Cr"]]
        block = {"pristine_Cr_mean_absmoment": round(float(cr_m.mean()), 3),
                 "pristine_Cr_spread": round(float(np.ptp(cr_m)), 3)}
        prist_plain = read(SDIR / f"CrN_{tag}_2x2x2_pristine.extxyz")
        for d in DEFECTS:
            f = RDIR / "relaxed" / "CHGNet" / f"CrN_{tag}_2x2x2_{d}.extxyz"
            if not f.exists():
                continue
            at = read(f); at.calc = calc
            at.get_potential_energy()
            mm = np.array(at.calc.results["magmoms"])
            syms = at.get_chemical_symbols()
            cr = mm[[i for i, s in enumerate(syms) if s == "Cr"]]
            site, ideal = defect_site(prist_plain, d)
            dvec = at.get_positions() - site
            frac = dvec @ np.linalg.inv(at.get_cell()); frac -= np.round(frac)
            r = np.linalg.norm(frac @ at.get_cell(), axis=1)
            near_cr = [i for i, s in enumerate(syms) if s == "Cr" and 0.1 < r[i] < 3.0]
            block[d] = {"Cr_mean_absmoment": round(float(cr.mean()), 3),
                        "defect_NN_Cr_moments": [round(float(mm[i]), 3) for i in near_cr[:8]],
                        "sum_absmoments": round(float(np.abs(mm).sum()), 2)}
        out[tag] = block
    return out


def main():
    results = {"note": "see script docstring for protocols and caveats"}

    # 4. reservoir cross-check (pure post-processing)
    st3 = json.loads((RDIR / "stage3_results.json").read_text())
    resv = {}
    for m in st3:
        e_prist = m["defects"]["cubic"]["E_pristine"]  # 64 atoms = 32 f.u.
        e_fu = e_prist / 32
        mu_cr = m["reservoirs"]["mu_Cr"]
        mu_n_crrich = e_fu - mu_cr
        block = {"mu_N_N2": round(m["reservoirs"]["mu_N"], 3),
                 "mu_N_CrRich": round(mu_n_crrich, 3)}
        for tag in ["cubic", "sheared"]:
            for d in DEFECTS:
                v = m["defects"][tag].get(d)
                if not v:
                    continue
                nn = {"V_N": 1, "V_Cr": 0, "A_N": -1, "A_Cr": 1, "I_N": -1}[d]
                block[f"{tag}_{d}_Ef_CrRich"] = round(v["E_f_eV"] + nn * (mu_n_crrich - m["reservoirs"]["mu_N"]), 3)
        resv[m["model"]] = block
    results["reservoir_crosscheck"] = resv

    # 2. displacements per model (post-processing on saved geometries)
    disp = {}
    for name in MODELS:
        mdl = {}
        for tag in ["cubic", "sheared"]:
            prist_plain = read(SDIR / f"CrN_{tag}_2x2x2_pristine.extxyz")
            for d in DEFECTS:
                f = RDIR / "relaxed" / name / f"CrN_{tag}_2x2x2_{d}.extxyz"
                if not f.exists():
                    continue
                site, ideal = defect_site(prist_plain, d)
                relaxed = read(f)
                mdl[f"{tag}_{d}"] = shell_displacements(ideal, relaxed, site)
        disp[name] = mdl
    results["displacements"] = disp
    (RDIR / "stage3b_analysis.json").write_text(json.dumps(results, indent=2))
    print("displacements + reservoirs done", flush=True)

    # 1. CHGNet magmoms (GPU)
    results["chgnet_magmoms"] = chgnet_magmoms()
    (RDIR / "stage3b_analysis.json").write_text(json.dumps(results, indent=2))
    print("magmoms done", flush=True)

    # 3. elastic constants (GPU, per model)
    el = {}
    for name in MODELS:
        el[name] = elastic_constants(name)
        print(f"elastic {name}: {el[name]}", flush=True)
        results["elastic"] = el
        (RDIR / "stage3b_analysis.json").write_text(json.dumps(results, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
