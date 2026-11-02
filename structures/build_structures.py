#!/usr/bin/env python3
"""Stage 2 — build CrN benchmark structures from the published reference geometry.

Geometry sources (see reference_data.csv):
  a_c = 4.14 A  cubic rocksalt lattice constant   [rojasStrainFieldsElectronic2017 Sec. II]
  alpha = 88 deg orthorhombic AFM2[110] shear     [rojasStrainFieldsElectronic2017 Sec. II]
  Defect supercells: 2x2x2 conventional (64 atoms) [rojasEnergeticsElectronicStructure2018 Sec. II]

Protocol notes (also in reference_data_README.md):
  * uMLIPs are spin-blind: magnetic phases are represented ONLY by their structural
    signature. Cubic AFM1[010] and cubic AFM2[110] are geometrically identical, so the
    only uMLIP-observable ordering probe in CrN is cubic-vs-sheared (the 8.0 meV/f.u.
    magnetostructural distortion of AFM2[110]).
  * I_N is placed at the tetrahedral hollow (1/4,1/4,1/4) of the conventional cell.
    The 2018 paper says only "previously hollow location" — ASSUMPTION, flagged.
  * Reservoirs for formation energies: N2 molecule (d = 1.10 A) and bcc Cr (a = 2.88 A),
    matching the reference's choice of N2 + bulk Cr (its Cr reservoir is AFM bulk Cr;
    spin-blind models see plain bcc Cr).
"""
import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import write
from pathlib import Path

A_C = 4.14
ALPHA_DEG = 88.0
OUT = Path(__file__).parent

def rocksalt_conventional(a=A_C):
    return bulk("CrN", crystalstructure="rocksalt", a=a, cubic=True)  # 8 atoms

def sheared_conventional(a=A_C, alpha_deg=ALPHA_DEG):
    """Cubic cell sheared so the angle between the two in-plane cubic axes is alpha.
    This turns the [110]/[1-10] frame into the orthorhombic AFM2[110] cell of the reference papers
    (a_o = 2a sin(alpha/2), b_o = a cos(alpha/2))."""
    at = rocksalt_conventional(a)
    alpha = np.radians(alpha_deg)
    cell = np.array([[a, 0, 0],
                     [a * np.cos(alpha), a * np.sin(alpha), 0],
                     [0, 0, a]])
    at.set_cell(cell, scale_atoms=True)
    return at

def supercell(at, n=(2, 2, 2)):
    return make_supercell(at, np.diag(n))

def make_defect(sc, kind):
    at = sc.copy()
    syms = at.get_chemical_symbols()
    n_idx = [i for i, s in enumerate(syms) if s == "N"]
    cr_idx = [i for i, s in enumerate(syms) if s == "Cr"]
    if kind == "V_N":
        del at[n_idx[0]]
    elif kind == "V_Cr":
        del at[cr_idx[0]]
    elif kind == "A_N":            # N on a Cr site
        at[cr_idx[0]].symbol = "N"
    elif kind == "A_Cr":           # Cr on an N site
        at[n_idx[0]].symbol = "Cr"
    elif kind == "I_N":            # tetrahedral hollow of the first conventional cube
        frac = np.array([0.125, 0.125, 0.125])  # (1/4,1/4,1/4) of one cube in a 2x2x2 cell
        pos = frac @ at.get_cell()
        at += Atoms("N", positions=[pos])
    else:
        raise ValueError(kind)
    return at

def main():
    structures = {}
    # Phase cells
    structures["CrN_cubic_conv"] = rocksalt_conventional()
    structures["CrN_sheared_conv"] = sheared_conventional()
    # Defect supercells in both phase geometries
    for tag, base in [("cubic", rocksalt_conventional()), ("sheared", sheared_conventional())]:
        sc = supercell(base)
        structures[f"CrN_{tag}_2x2x2_pristine"] = sc
        for kind in ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]:
            structures[f"CrN_{tag}_2x2x2_{kind}"] = make_defect(sc, kind)
    # Reservoirs
    structures["N2_molecule"] = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.10]],
                                      cell=[12, 12, 12], pbc=True)
    structures["Cr_bcc"] = bulk("Cr", crystalstructure="bcc", a=2.88, cubic=True)

    for name, at in structures.items():
        write(OUT / f"{name}.extxyz", at)
        print(f"{name}: {len(at)} atoms")

if __name__ == "__main__":
    main()
