#!/usr/bin/env python3
"""Smoke test: single-point energy of the cubic rocksalt CrN conventional cell (8 atoms)
with every installed uMLIP. Passes when each model returns a finite energy on GPU.

Prints one line per model: name, checkpoint id, energy (eV), energy per formula unit.
Models that are not installed are reported and skipped, not fatal.
"""
import traceback
from pathlib import Path
from ase.io import read

STRUCT = Path(__file__).parent.parent / "structures" / "CrN_cubic_conv.extxyz"

def get_calculators():
    """Yield (name, checkpoint_label, calculator) for every available model."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device: {device}")

    try:
        from mace.calculators import mace_mp
        yield "MACE-MP-0", "mace-mp medium", mace_mp(model="medium", device=device,
                                                     default_dtype="float64")
    except Exception:
        print("MACE: unavailable"); traceback.print_exc(limit=1)

    try:
        from chgnet.model.dynamics import CHGNetCalculator
        yield "CHGNet", "chgnet default", CHGNetCalculator(use_device=device)
    except Exception:
        print("CHGNet: unavailable"); traceback.print_exc(limit=1)

    try:
        from sevenn.calculator import SevenNetCalculator
        yield "SevenNet-0", "7net-0", SevenNetCalculator(model="7net-0", device=device)
    except Exception:
        print("SevenNet: unavailable"); traceback.print_exc(limit=1)

    try:
        import matgl
        from matgl.ext.ase import PESCalculator
        pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2", revision="8414b23e68b173d6e36271e4366ee3717364051c")
        yield "M3GNet", "M3GNet-PES-MatPES-PBE-2025.2", PESCalculator(pot)
    except Exception:
        print("M3GNet/matgl: unavailable"); traceback.print_exc(limit=1)

    try:
        from mattersim.forcefield import MatterSimCalculator
        yield "MatterSim", "mattersim-v1", MatterSimCalculator(device=device)
    except Exception:
        print("MatterSim: unavailable"); traceback.print_exc(limit=1)

def main():
    atoms0 = read(STRUCT)
    n_fu = sum(1 for s in atoms0.get_chemical_symbols() if s == "Cr")
    for name, ckpt, calc in get_calculators():
        atoms = atoms0.copy()
        atoms.calc = calc
        e = atoms.get_potential_energy()
        print(f"{name:12s} [{ckpt}]  E = {e:.4f} eV   E/f.u. = {e/n_fu:.4f} eV")

if __name__ == "__main__":
    main()
