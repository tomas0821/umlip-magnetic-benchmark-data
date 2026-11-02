#!/usr/bin/env python3
"""Build the MnO test set: the CrN protocol transferred to a second magnetic material.

WHY MnO. The CrN result rests on one material, one distortion mode (a [110] shear) and one
energy scale (8.0 meV/f.u.). MnO tests whether the three preflight diagnostics
generalise or are CrN-specific, with no new DFT: Eckhoff & Behler,
npj Comput. Mater. 7, 170 (2021) publish HSE06 reference values for three magnetic orders,
each with its OWN lattice symmetry -- and the distortion mode (trigonal) and energy scale
(2.1 meV/atom) are both different from CrN's.

Rocksalt MnO under each magnetic order (all values HSE06 unless noted):
  FM      -> cubic         a = 4.462 A
  AFM-I   -> tetragonal    a = 4.459 A, c = 4.420 A
  AFM-II  -> rhombohedral  a = 4.434 A, alpha = 90.89 deg   (ground state)
                           exp: a = 4.430 A, alpha = 90.62 deg at 8 K
alpha > 90 deg is a contraction along [111] (|a1+a2+a3|^2 = 3a^2(1 + 2 cos alpha)).

Rocksalt has no free internal coordinate under either trigonal or tetragonal strain -- the
ions sit on fixed Wyckoff positions -- so the lattice parameters specify the geometry
completely. That is why Eckhoff quotes only lattice parameters, and why we can rebuild their
structures exactly without their coordinate files.

THE SHARP TEST. Eckhoff report that on a CUBIC lattice, AFM-II is exactly degenerate with a
second, different AFM order (their Fig. 2b), and that the second one stays cubic when relaxed.
The rhombohedral distortion is therefore the entire origin of AFM-II's energetic preference.
Those two orders share one geometry, so a spin-blind potential must assign them identical
energies -- not approximately, but exactly, by symmetry. It is the cleanest available
statement of the spin-blindness limit and it needs no calculation to establish.

Writes structures/MnO_*.extxyz and reference_data_MnO.csv.
"""
import csv
from pathlib import Path

from ase import Atoms
from ase.cell import Cell

ROOT = Path(__file__).parent.parent
SDIR = ROOT / "structures"

# conventional rocksalt cell, 4 f.u.; fractional coords are strain-independent
FRAC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]
SYMBOLS = ["Mn"] * 4 + ["O"] * 4

# name -> (cellpar, description)
PHASES = {
    "MnO_FM_cubic":       ([4.462, 4.462, 4.462, 90.0, 90.0, 90.0],
                           "FM, cubic, HSE06 a=4.462"),
    "MnO_AFM1_tetra":     ([4.459, 4.459, 4.420, 90.0, 90.0, 90.0],
                           "AFM-I, tetragonal, HSE06 a=4.459 c=4.420"),
    "MnO_AFM2_rhomb":     ([4.434, 4.434, 4.434, 90.89, 90.89, 90.89],
                           "AFM-II ground state, rhombohedral, HSE06 a=4.434 alpha=90.89"),
    # cubic-constrained AFM-II: the reference for the distortion energy. Eckhoff do not quote
    # its lattice constant, so we seed it at the AFM-II volume and let each model relax `a`
    # under a fixed cubic cell -- the same construction used for CrN.
    "MnO_AFM2_cubic_seed": ([4.434, 4.434, 4.434, 90.0, 90.0, 90.0],
                            "AFM-II restricted to cubic; seed for constrained relaxation"),
    "MnO_AFM2_rhomb_exp": ([4.430, 4.430, 4.430, 90.62, 90.62, 90.62],
                           "AFM-II experimental geometry at 8 K"),
}

# HSE06 reference energetics, meV/atom, relative to the rhombohedral AFM-II ground state
REFERENCE = [
    ("ordering_energy", "FM minus AFM-II(rhomb)", 45.5, "meV/atom", "HSE06",
     "mHDNNP 45.8; both from Eckhoff Fig. 2 discussion"),
    ("distortion_energy", "cubic AFM-II minus rhombohedral AFM-II", 2.1, "meV/atom", "HSE06",
     "mHDNNP 1.9; the analogue of CrN's 8.0 meV/f.u. shear"),
    ("distortion_angle", "AFM-II alpha", 90.89, "degree", "HSE06",
     "mHDNNP 90.77; exp 90.62 at 8 K"),
    ("lattice_constant", "AFM-II a", 4.434, "Angstrom", "HSE06", "mHDNNP 4.433; exp 4.430"),
    ("lattice_constant", "FM a (cubic)", 4.462, "Angstrom", "HSE06", "mHDNNP 4.461"),
    ("lattice_constant", "AFM-I a (tetragonal)", 4.459, "Angstrom", "HSE06", "mHDNNP 4.461"),
    ("lattice_constant", "AFM-I c (tetragonal)", 4.420, "Angstrom", "HSE06", "mHDNNP 4.414"),
    ("spinflip_energy", "one spin flip per 2x2x2 supercell", 1.5, "meV/atom", "HSE06",
     "mHDNNP 1.5; reduces the distortion by 0.11 deg (mHDNNP 0.09)"),
    ("degeneracy", "AFM-II vs second AFM order on a cubic lattice", 0.0, "meV/atom", "HSE06",
     "exactly degenerate; same geometry, so spin-blind models cannot split them even in "
     "principle"),
]


def main():
    SDIR.mkdir(exist_ok=True)
    for name, (cellpar, desc) in PHASES.items():
        at = Atoms(SYMBOLS, cell=Cell.fromcellpar(cellpar), pbc=True)
        at.set_scaled_positions(FRAC)
        at.info["description"] = desc
        at.write(SDIR / f"{name}.extxyz")
        a, b, c, al, be, ga = at.cell.cellpar()
        print(f"{name:22s} a={a:.3f} b={b:.3f} c={c:.3f} "
              f"alpha={al:.2f} V={at.get_volume():.2f} A^3  {desc}")

    out = ROOT / "reference_data_MnO.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "detail", "value", "unit", "method", "note",
                    "material", "source", "uMLIP_observable"])
        for prop, detail, val, unit, method, note in REFERENCE:
            w.writerow([prop, detail, val, unit, method, note, "MnO",
                        "eckhoffHighdimensionalNeuralNetwork2021", "yes"])
    print(f"\nwrote {out.relative_to(ROOT)} ({len(REFERENCE)} rows)")


if __name__ == "__main__":
    main()
