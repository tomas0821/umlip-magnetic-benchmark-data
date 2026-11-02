#!/usr/bin/env python3
"""Systematic reconstruction audit of every relaxed defect cell.

A "reconstruction" = the relaxation left the intended defect configuration (an atom moved
far enough to change coordination), which makes the resulting formation energy a different
physical quantity from the reference. Flagged by max |displacement| from the ideal defect
geometry. Over all 70 relaxations (seven models) the distribution is bimodal: 59 cases that
stay in configuration span 0.06-0.70 A (interstitials at the top end, large but physical; the
largest shell-averaged DFT reference displacement is 0.31 A, orthorhombic A_Cr Cr shell), then
an empty interval, then 11 antisite cases at 1.10-2.17 A. The 0.9 A threshold was placed in the
gap of the first 50 relaxations (0.69 -> 1.11 A) and holds for the 20 added later (0.70 -> 1.10 A).

Writes results/relaxation_audit.json.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read

ROOT = Path(__file__).parent.parent
MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim",
          "Orb-v3", "GRACE-2L-OMAT"]
# relaxed geometries live in three places: the original run, the M3GNet batch-1 rerun that
# supersedes it, and the second-environment models.
REL_DIRS = {"M3GNet-MatPES": "relaxed_batch1", "Orb-v3": "relaxed_extra",
            "GRACE-2L-OMAT": "relaxed_extra"}
DEFECTS = ["V_N", "V_Cr", "A_N", "A_Cr", "I_N"]
THRESH = 0.9  # Angstrom - in the empirical gap (0.69 -> 1.11 A over 50; 0.70 -> 1.10 A over 70)


def audit_one(model, tag, defect):
    ideal = read(ROOT / "structures" / f"CrN_{tag}_2x2x2_{defect}.extxyz")
    f = (ROOT / "results" / REL_DIRS.get(model, "relaxed") / model /
         f"CrN_{tag}_2x2x2_{defect}.extxyz")
    if not f.exists():
        return None
    rel = read(f)
    cell = rel.get_cell()
    d = rel.get_positions() - ideal.get_positions()
    fr = d @ np.linalg.inv(cell)
    fr -= np.round(fr)
    d = fr @ cell
    disp = np.linalg.norm(d, axis=1)
    dmin = min(rel.get_distance(i, j, mic=True)
               for i in range(len(rel)) for j in range(i + 1, len(rel)))
    return {"max_disp_A": round(float(disp.max()), 4),
            "mean_disp_A": round(float(disp.mean()), 4),
            "min_interatomic_A": round(float(dmin), 4),
            "reconstructed": bool(disp.max() > THRESH)}


def main():
    out = {"threshold_A": THRESH, "cases": {}}
    flagged = []
    for m in MODELS:
        for tag in ["cubic", "sheared"]:
            for d in DEFECTS:
                r = audit_one(m, tag, d)
                if r is None:
                    continue
                key = f"{m}|{tag}|{d}"
                out["cases"][key] = r
                if r["reconstructed"]:
                    flagged.append((key, r["max_disp_A"]))
    out["flagged"] = [{"case": k, "max_disp_A": v} for k, v in flagged]
    vals = [c["max_disp_A"] for c in out["cases"].values() if not c["reconstructed"]]
    out["clean_population_max_disp_A"] = {"min": min(vals), "max": max(vals),
                                          "n": len(vals)}
    (ROOT / "results" / "relaxation_audit.json").write_text(json.dumps(out, indent=1))
    print(f"audited {len(out['cases'])} cases; clean max-disp range "
          f"{min(vals):.3f}-{max(vals):.3f} A over {len(vals)} cases")
    for k, v in flagged:
        print(f"  RECONSTRUCTED: {k}  max|d| = {v} A")


if __name__ == "__main__":
    main()
