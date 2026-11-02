#!/usr/bin/env python3
"""Authoritative per-model defect and phase statistics.

Single source of truth for the aggregate statistics. Supersedes ad-hoc counts. Applies three documented exclusion rules, each with a physical justification, and
reports every statistic both with and without exclusions so nothing is hidden.

EXCLUSION RULES
  E1 reconstructed   : relaxation left the intended defect configuration (max|d| > 0.9 A,
                       results/relaxation_audit.json). The resulting energy describes a
                       different defect than the reference, so a model-vs-DFT ratio is a
                       category error. 11 antisite cases over the 70 relaxations.
  E2 unstable host   : SevenNet-0, BOTH phases (extended 2026-11-01). Its
                       C44 < 0 makes the parent lattice mechanically unstable. Cubic host:
                       rattle-and-relax of the pristine fixed-cell 64-atom cell descends
                       0.64 eV/cell (20 meV/f.u.) below the symmetric reference used for the
                       formation energies (results/diagnostic_validation.json), so cubic
                       formation energies are measured from a saddle point. Sheared host: the
                       V_Cr relaxation runs away along the unstable shear (E_f +0.017 ->
                       -0.867 eV between fmax 1e-2 and 3e-3, stable at 1e-3).
  E3 below noise     : M3GNet-MatPES meV-scale phase-probe values. Its numerical noise floor
                       is 2.0 meV/f.u. (results/batch1_numerical_precision.json), a quarter of
                       the 8 meV/f.u. signal being measured.

M3GNet defect/reservoir values are taken from the batch-1 rerun (four-fold step budget for
defects, eight-fold for the cell, plus an LBFGS fallback), which supersedes the original run.

Writes results/final_statistics.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
R = ROOT / "results"

DFT_EF = {("cubic", "V_N"): 2.48, ("cubic", "V_Cr"): 6.90, ("cubic", "A_N"): 10.05,
          ("cubic", "I_N"): 12.01, ("sheared", "V_N"): 2.13, ("sheared", "V_Cr"): 5.72,
          ("sheared", "A_Cr"): 9.70, ("sheared", "A_N"): 10.14}
MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim",
          "Orb-v3", "GRACE-2L-OMAT"]


def load():
    st3 = {m["model"]: m for m in json.loads((R / "stage3_results.json").read_text())}
    rerun = json.loads((R / "batch1_m3gnet_rerun.json").read_text())
    st3["M3GNet-MatPES"]["defects"] = rerun["defects"]
    st3["M3GNet-MatPES"]["reservoirs"].update(
        {k: rerun["reservoirs"][k] for k in ("E_N2", "mu_N", "mu_Cr")})
    audit = json.loads((R / "relaxation_audit.json").read_text())
    noise = json.loads((R / "batch1_numerical_precision.json").read_text())
    for extra in ("Orb-v3", "GRACE-2L-OMAT"):
        f = R / f"extra_{extra}.json"
        if f.exists():
            e = json.loads(f.read_text())
            st3[extra] = {"defects": e["defects"], "reservoirs": e["reservoirs"],
                          "phase": e["phase"]}
            noise[extra] = {"noise_floor_meV_per_fu":
                            e["numerical_precision"]["noise_floor_meV_per_fu"]}
    return st3, audit, noise


def excluded(model, tag, defect, audit):
    """Return (bool, reason) for the exclusion rules."""
    if audit["cases"].get(f"{model}|{tag}|{defect}", {}).get("reconstructed"):
        return True, "E1 reconstructed"
    if model == "SevenNet-0":
        return True, "E2 unstable host"
    return False, ""


def main():
    st3, audit, noise = load()
    out = {"rules": {"E1": "reconstructed (max|d| > 0.9 A)",
                     "E2": "SevenNet-0, both phases: mechanically unstable host (C44 < 0)",
                     "E3": "M3GNet: below 2.0 meV/f.u. numerical noise floor"},
           "per_model": {}, "aggregate": {}}

    ratios_all, ratios_clean = [], []
    for m in MODELS:
        blk = {"points": {}, "excluded": {}}
        for (tag, d), ref in DFT_EF.items():
            v = st3[m]["defects"][tag].get(d)
            if not v:
                continue
            ef = v["E_f_eV"]
            ex, why = excluded(m, tag, d, audit)
            rec = {"E_f": round(ef, 4), "DFT": ref,
                   "ratio_DFT_over_model": (round(ref / ef, 2) if ef > 0 else None),
                   "negative": ef < 0}
            if ex:
                rec["reason"] = why
                blk["excluded"][f"{tag}|{d}"] = rec
            else:
                blk["points"][f"{tag}|{d}"] = rec
                if ef > 0:
                    ratios_clean.append((ref / ef, m, tag, d))
            if ef > 0:
                ratios_all.append((ref / ef, m, tag, d))

        # hierarchy: is V_N the lowest-energy defect, per phase? (DFT: yes in both)
        hier = {}
        for tag in ("cubic", "sheared"):
            ds = {d: st3[m]["defects"][tag][d]["E_f_eV"]
                  for d in ("V_N", "V_Cr", "A_N", "A_Cr", "I_N")
                  if d in st3[m]["defects"][tag]}
            lowest = min(ds, key=ds.get)
            ex, why = excluded(m, tag, "V_Cr", audit)
            hier[tag] = {"lowest_defect": lowest, "V_N_is_lowest": lowest == "V_N",
                         "E_V_N": round(ds.get("V_N", float("nan")), 3),
                         "E_V_Cr": round(ds.get("V_Cr", float("nan")), 3),
                         "phase_excluded": ex, "exclusion_reason": why}
        blk["hierarchy"] = hier
        blk["noise_floor_meV_per_fu"] = noise[m]["noise_floor_meV_per_fu"]

        # parity slope on the clean subset
        xs = [DFT_EF[tuple(k.split("|"))] for k in blk["points"]]
        ys = [blk["points"][k]["E_f"] for k in blk["points"]]
        if len(xs) >= 3:
            x, y = np.array(xs), np.array(ys)
            s0 = float((x @ y) / (x @ x))
            A = np.vstack([x, np.ones_like(x)]).T
            (s1, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = float(1 - res[0] / ((y - y.mean()) ** 2).sum()) if len(res) else None
            blk["parity_clean"] = {"n": len(xs), "slope_through_origin": round(s0, 3),
                                   "slope": round(float(s1), 3), "intercept": round(float(b), 3),
                                   "R2": (round(r2, 2) if r2 is not None else None)}
        out["per_model"][m] = blk

    # aggregates
    ra = sorted(ratios_all); rc = sorted(ratios_clean)
    out["aggregate"]["underestimation_factor_all_positive"] = {
        "min": {"factor": round(ra[0][0], 1), "case": f"{ra[0][1]} {ra[0][2]} {ra[0][3]}"},
        "max": {"factor": round(ra[-1][0], 1), "case": f"{ra[-1][1]} {ra[-1][2]} {ra[-1][3]}"},
        "n": len(ra)}
    out["aggregate"]["underestimation_factor_after_exclusions"] = {
        "min": {"factor": round(rc[0][0], 1), "case": f"{rc[0][1]} {rc[0][2]} {rc[0][3]}"},
        "max": {"factor": round(rc[-1][0], 1), "case": f"{rc[-1][1]} {rc[-1][2]} {rc[-1][3]}"},
        "n": len(rc)}
    inv_any = [m for m in MODELS
               if not all(out["per_model"][m]["hierarchy"][t]["V_N_is_lowest"]
                          for t in ("cubic", "sheared"))]
    inv_clean = [m for m in MODELS
                 if not all(out["per_model"][m]["hierarchy"][t]["V_N_is_lowest"]
                            for t in ("cubic", "sheared")
                            if not out["per_model"][m]["hierarchy"][t]["phase_excluded"])]
    out["aggregate"]["hierarchy_inverted_any_phase"] = {"models": inv_any, "count": len(inv_any)}
    out["aggregate"]["hierarchy_inverted_after_exclusions"] = {"models": inv_clean,
                                                               "count": len(inv_clean)}
    out["aggregate"]["n_reconstructed"] = len(audit["flagged"])
    out["aggregate"]["negative_Ef_standard_reservoirs"] = [
        {"model": m, "case": k}
        for m in MODELS
        for k, v in {**out["per_model"][m]["points"], **out["per_model"][m]["excluded"]}.items()
        if v["negative"]]
    (R / "final_statistics.json").write_text(json.dumps(out, indent=1))

    # human-readable
    print("UNDERESTIMATION FACTORS (DFT / model, positive values only)")
    a = out["aggregate"]["underestimation_factor_all_positive"]
    c = out["aggregate"]["underestimation_factor_after_exclusions"]
    print(f"  all points   : {a['min']['factor']}x ({a['min']['case']}) .. "
          f"{a['max']['factor']}x ({a['max']['case']})  [n={a['n']}]")
    print(f"  after E1/E2  : {c['min']['factor']}x ({c['min']['case']}) .. "
          f"{c['max']['factor']}x ({c['max']['case']})  [n={c['n']}]")
    print("\nHIERARCHY (is V_N the lowest-energy defect? DFT says yes in both phases)")
    for m in MODELS:
        h = out["per_model"][m]["hierarchy"]
        bits = []
        for t in ("cubic", "sheared"):
            mark = "OK " if h[t]["V_N_is_lowest"] else "INV"
            ex = " (excl)" if h[t]["phase_excluded"] else ""
            bits.append(f"{t}={mark}[{h[t]['lowest_defect']}]{ex}")
        print(f"  {m:<15} {'  '.join(bits)}")
    print(f"  inverted in any phase        : {len(inv_any)}/{len(MODELS)} {inv_any}")
    print(f"  inverted after exclusions    : {len(inv_clean)}/{len(MODELS)} {inv_clean}")
    neg = [(m, k) for m in MODELS
           for k, v in {**out["per_model"][m]["points"], **out["per_model"][m]["excluded"]}.items()
           if v["negative"]]
    print(f"\nNEGATIVE formation energies at standard reservoirs: {neg if neg else 'none'}")
    print("\nPARITY SLOPES on clean subset")
    for m in MODELS:
        p = out["per_model"][m].get("parity_clean")
        if p:
            print(f"  {m:<15} n={p['n']}  slope0={p['slope_through_origin']:.3f}  "
                  f"slope={p['slope']:.3f} (R2={p['R2']})")
    print(f"\nreconstructed cases: {out['aggregate']['n_reconstructed']}")


if __name__ == "__main__":
    main()
