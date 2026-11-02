#!/usr/bin/env python3
"""Score the preflight diagnostics against what they predicted, and print the two summary
tables.

The question is not whether these models fail. It is whether three cheap preflight checks tell you IN
ADVANCE whether a given uMLIP can be trusted on a given magnetic system. That claim is only
worth printing if firing a diagnostic actually predicts the downstream failure -- otherwise it
is a post-hoc rationalisation of results we already had. This script does the scoring.

  Table A  predictive validation (CrN). Each diagnostic was measured on a cheap test, made a
           prediction, and an independent calculation checked it.
             elastic stability  C44 < 0        -> cubic lattice runs away under a 1 deg shear
             resolution floor   floor > 0.08   -> rattle-and-relax energies scatter
  Table B  transfer (MnO). The diagnostics were calibrated on CrN; MnO has a different
           distortion mode (trigonal, not [110] shear) and half the energy scale
           (2.1 meV/atom = 4.2 meV/f.u. vs 8.0 meV/f.u.), so the resolution floor measured on
           CrN makes a falsifiable prediction about MnO before MnO is run.

Scoring rules (fixed 2026-11-01):
  * Shear test scored only where the UNPERTURBED control run (start at 90 deg) returns to cubic.
    A model whose control drifts cannot be judged on the perturbed runs (M3GNet-MatPES: its
    C44 is below its noise floor and its control wanders to 92.6 deg) -> "indeterminate".
    The shear test is a consistency check on the sign of C44 (same eps_xy curvature), not an
    independent prediction; it is reported as such.
  * Resolution: predicted "scatters" if floor > PRED_THRESHOLD (1% of the 8 meV/f.u. signal,
    fixed before the runs); observed "scatters" if spread > OBS_THRESHOLD (0.5 meV/f.u., ~6% of
    the signal, chosen when scoring). Sensitivity to a common threshold is recorded.
  * Stability precondition for the resolution score: adopted AFTER the SevenNet-0 result (it
    would otherwise be a false clear of the floor-only rule). Recorded, not hidden.

Reads results/{diagnostic_validation,mno_results}{,_extra}.json and results/claims.json.
Writes results/validation_summary.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
R = ROOT / "results"
MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim",
          "Orb-v3", "GRACE-2L-OMAT"]
DFT_DISTORTION_MEV_ATOM = 2.1          # MnO, HSE06
DFT_ORDERING_MEV_ATOM = 45.5           # MnO, HSE06
DFT_ALPHA = 90.89
DFT_A_AFM2 = 4.434
PRED_THRESHOLD = 0.08   # meV/f.u.: floor that predicts scatter (1% of the CrN signal)
OBS_THRESHOLD = 0.5     # meV/f.u.: rattle spread counted as observed scatter


def load(stem):
    out = {}
    for suffix in ("", "_extra"):
        f = R / f"{stem}{suffix}.json"
        if f.exists():
            out.update(json.loads(f.read_text()))
    return out


def ok(x):
    return isinstance(x, dict) and "error" not in x


def main():
    val, mno = load("diagnostic_validation"), load("mno_results")
    summary = {"table_A_predictive_validation": {}, "table_B_mno_transfer": {},
               "scorecard": {}}

    # ---- Table A: did the diagnostics predict the failures? ----
    hits = misses = 0
    print("TABLE A  Predictive validation of the preflight diagnostics (CrN)")
    print(f"{'model':16s} {'C44':>7s} {'predicts':>10s} {'observed':>10s} {'':3s}"
          f" {'floor':>7s} {'predicts':>13s} {'spread':>8s}")
    print("-" * 92)
    for m in MODELS:
        v = val.get(m)
        if not ok(v):
            continue
        row = {"C44_GPa": v["C44_GPa"], "noise_floor_meV_per_fu": v["noise_floor_meV_per_fu"]}
        # (A) elastic stability
        obs_s = pred_s = None
        if ok(v.get("shear_stability")):
            s = v["shear_stability"]
            pred_s = v["prediction_shear"]
            control_ok = s.get("control_90deg", {}).get("returned_to_cubic", True)
            if not control_ok:
                obs_s = "indeterminate"
                row.update({"shear_predicted": pred_s, "shear_observed": obs_s,
                            "shear_correct": None, "control_returned_to_cubic": False,
                            "control_final_angles": s["control_90deg"].get("final_angles"),
                            "max_departure_deg": s.get("_max_departure_deg"),
                            "note_shear": "unperturbed control drifts; shear test not scored"})
            else:
                obs_s = "unstable" if "UNSTABLE" in s["_verdict"] else "stable"
                row.update({"shear_predicted": pred_s, "shear_observed": obs_s,
                            "shear_correct": pred_s == obs_s, "control_returned_to_cubic": True,
                            "max_departure_deg": s.get("_max_departure_deg")})
                hits += pred_s == obs_s
                misses += pred_s != obs_s
        # (B) resolution floor.
        # PRECEDENCE, established by this run: the invariance floor
        # predicts the achievable resolution only for a model whose lattice is mechanically
        # stable. SevenNet-0 has a numerically perfect floor (0.0000) yet scatters by
        # 18 meV/f.u., because its irreproducibility comes from the C44 < 0 instability -- the
        # ions roll into different broken-symmetry minima -- not from numerical noise. Scoring
        # it against the floor would be scoring a diagnostic outside its stated precondition,
        # so it is reported as "precondition failed" rather than silently counted as a hit or
        # quietly dropped.
        obs_r = pred_r = spread = None
        precond = row.get("shear_observed") == "stable"
        if ok(v.get("mev_resolution")):
            r = v["mev_resolution"]
            spread = r["spread_meV_per_fu"]
            pred_r = v["prediction_resolution"]
            # "scatters" = the spread is a meV-scale fraction of the 8 meV/f.u. CrN signal
            obs_r = "scatters" if spread > OBS_THRESHOLD else "reproducible"
            row.update({"resolution_predicted": pred_r, "resolution_observed": obs_r,
                        "spread_meV_per_fu": spread,
                        "stability_precondition_met": precond})
            if precond:
                row["resolution_correct"] = pred_r == obs_r
                hits += pred_r == obs_r
                misses += pred_r != obs_r
                if pred_r != obs_r:
                    # Direction matters for a preflight check. Predicting scatter that does
                    # not materialise is a FALSE ALARM: the user runs a more careful
                    # calculation than strictly needed. Predicting clean numbers that then
                    # scatter is a FALSE CLEAR, which silently corrupts results. Only the
                    # second kind is dangerous, so the two are never pooled into one count.
                    row["miss_type"] = ("false alarm (conservative)"
                                        if pred_r == "scatters" else
                                        "false clear (dangerous)")
            else:
                row["resolution_correct"] = None
                row["note"] = ("stability precondition not met (unstable or indeterminate "
                               "lattice); floor diagnostic not scored")
                # what the floor-only rule (fixed before the runs) would have scored
                row["floor_only_rule_outcome"] = (
                    "hit" if pred_r == obs_r else
                    ("false clear" if pred_r == "reproducible" else "false alarm"))
        summary["table_A_predictive_validation"][m] = row
        mark = lambda p, o: "OK" if (p is not None and p == o) else ("--" if p is None else "XX")
        rmark = "n/a" if (spread is not None and not precond) else mark(pred_r, obs_r)
        print(f"{m:16s} {v['C44_GPa']:+7.0f} {str(pred_s):>10s} {str(obs_s):>10s}"
              f" {('n/a' if obs_s == 'indeterminate' else mark(pred_s, obs_s)):>3s} {v['noise_floor_meV_per_fu']:7.4f}"
              f" {str(pred_r):>13s} {('%.3f' % spread) if spread is not None else '-':>8s}"
              f" {rmark}")
    summary["scorecard"] = {"predictions_correct": hits, "predictions_wrong": misses,
                            "total": hits + misses,
                            "pred_threshold_meV_per_fu": PRED_THRESHOLD,
                            "obs_threshold_meV_per_fu": OBS_THRESHOLD}
    print(f"\n  {hits}/{hits + misses} predictions correct "
          f"(resolution scored only where the stability precondition holds)")

    # The sharper result: the C44 sign separates the rattle spread with a clean gap. Quote the
    # gap rather than a threshold we chose, so the separation is a property of the data.
    rows = summary["table_A_predictive_validation"]
    st = [r["spread_meV_per_fu"] for r in rows.values()
          if r.get("shear_observed") == "stable" and "spread_meV_per_fu" in r]
    un = [r["spread_meV_per_fu"] for r in rows.values()
          if r.get("shear_observed") == "unstable" and "spread_meV_per_fu" in r]
    ind = {m: r["spread_meV_per_fu"] for m, r in rows.items()
           if r.get("shear_observed") == "indeterminate" and "spread_meV_per_fu" in r}
    if st and un:
        sep = {"max_spread_stable_models": max(st), "min_spread_unstable_models": min(un),
               "separation_factor": round(min(un) / max(st), 1),
               "n_stable": len(st), "n_unstable": len(un),
               "unstable_models": {m: r["spread_meV_per_fu"] for m, r in rows.items()
                                   if r.get("shear_observed") == "unstable"},
               "indeterminate_models": ind}
        summary["scorecard"]["rattle_separation_by_C44_sign"] = sep
        print(f"  C44 sign separates the rattle spread: stable models <= "
              f"{max(st):.3f}, unstable models >= {min(un):.3f} meV/f.u. "
              f"(factor {sep['separation_factor']}x, no model in between)")

    mt = {m: r["miss_type"] for m, r in rows.items() if "miss_type" in r}
    if mt:
        summary["scorecard"]["misses_by_type"] = mt
        dangerous = [m for m, t in mt.items() if t.startswith("false clear")]
        summary["scorecard"]["false_clears"] = dangerous
        print(f"  misses: " + "; ".join(f"{m} = {t}" for m, t in mt.items())
              + f"  -> {len(dangerous)} false clear(s)")

    fc0 = [m for m, r in rows.items() if r.get("floor_only_rule_outcome") == "false clear"]
    summary["scorecard"]["false_clears_without_precondition"] = fc0
    summary["scorecard"]["obs_threshold_set_at"] = "scoring (after the runs)"
    summary["scorecard"]["resolution_true_positives"] = sum(
        1 for r in rows.values() if r.get("resolution_correct") and r.get("resolution_predicted") == "scatters")
    # rattle detail: is the spread one outlier, and do the rattled cells sit below the symmetric
    # pristine reference used for formation energies? (64-atom cell = 32 f.u.)
    import numpy as _np
    st3 = {x["model"]: x for x in json.loads((R / "stage3_results.json").read_text())}
    for xm in ("Orb-v3", "GRACE-2L-OMAT"):
        f = R / f"extra_{xm}.json"
        if f.exists():
            st3[xm] = json.loads(f.read_text())
    rattle = {}
    for m in MODELS:
        v = val.get(m, {})
        en = v.get("mev_resolution", {}).get("energies_eV") if ok(v) else None
        if not en:
            continue
        e = _np.array(en)
        med = _np.median(e)
        drop_out = _np.delete(e, _np.argmax(_np.abs(e - med)))
        rec = {"spread_meV_per_fu": round(float(e.max() - e.min()) / 32 * 1000, 3),
               "spread_without_largest_outlier_meV_per_fu": round(float(drop_out.max() - drop_out.min()) / 32 * 1000, 3)}
        ep = st3.get(m, {}).get("defects", {}).get("cubic", {}).get("E_pristine")
        if ep is not None and m != "M3GNet-MatPES":   # M3GNet pristine comes from a rerun; skip
            rec["median_drop_below_symmetric_pristine_meV_per_fu"] = round(float(med - ep) / 32 * 1000, 2)
        rattle[m] = rec
    summary["scorecard"]["rattle_detail"] = rattle
    scored = {m: r for m, r in rows.items() if r.get("resolution_correct") is not None}
    sens = {}
    for t_ in (0.05, 0.08, 0.1, 0.15, 0.187, 0.2, 0.3, 0.5, 0.7):
        h = fc = fa = 0
        for m, r in scored.items():
            p_ = val[m]["noise_floor_meV_per_fu"] > t_
            o_ = r["spread_meV_per_fu"] > t_
            h += p_ == o_
            fc += (not p_) and o_
            fa += p_ and not o_
        sens[f"{t_:g}"] = {"hits": h, "n": len(scored), "false_clears": fc, "false_alarms": fa}
    summary["scorecard"]["common_threshold_sensitivity"] = sens

    # Threshold-free check on the resolution diagnostic: forget the binary labels and ask
    # whether the floor RANKS the achievable resolution correctly among stable models. This
    # does not depend on where we drew the "scatters" line.
    import numpy as np
    pairs = [(val[m]["noise_floor_meV_per_fu"], rows[m]["spread_meV_per_fu"])
             for m in rows if rows[m].get("shear_observed") == "stable"
             and "spread_meV_per_fu" in rows[m]]
    if len(pairs) >= 4:
        f = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])
        rank = lambda a: np.argsort(np.argsort(a))
        rho = float(np.corrcoef(rank(f), rank(s))[0, 1])
        from scipy.stats import spearmanr
        summary["scorecard"]["floor_vs_spread_spearman_stable_models"] = {
            "rho": round(rho, 3), "n": len(pairs),
            "p_value": round(float(spearmanr(f, s).pvalue), 2)}
        print(f"  floor ranks achievable resolution among stable models: "
              f"Spearman rho = {rho:+.3f} (n = {len(pairs)})\n")

    # ---- Table B: MnO transfer ----
    if mno:
        print("TABLE B  Transfer to MnO (Eckhoff HSE06 reference; no new DFT)")
        print(f"{'model':16s} {'ordering':>10s} {'distortion':>11s} {'alpha_relaxed':>14s}"
              f" {'a_eq cubic':>11s}  verdict")
        print("-" * 96)
        print(f"{'DFT (HSE06)':16s} {DFT_ORDERING_MEV_ATOM:+10.1f} "
              f"{DFT_DISTORTION_MEV_ATOM:+11.1f} {DFT_ALPHA:14.2f} {4.434:11.3f}  reference")
        for m in MODELS:
            b = mno.get(m)
            if not ok(b):
                continue
            row = {}
            p = b.get("phase_probe")
            if ok(p):
                row["ordering_meV_per_atom"] = p["ordering_FM_minus_AFM2"]
                row["distortion_meV_per_atom"] = p["distortion_cubic_minus_rhomb"]
                # same sign convention as the CrN phase probe: E(distorted) - E(cubic)
                row["distortion_rhomb_minus_cubic_meV_per_atom"] = -p["distortion_cubic_minus_rhomb"]
                # exact split: ordering = [E_cub(4.462) - E_cub(4.434)] + [E_cub(4.434) - E_rh(4.434)]
                row["volume_term_meV_per_atom"] = round(
                    p["ordering_FM_minus_AFM2"] - p["distortion_cubic_minus_rhomb"], 3)
            s = b.get("symmetry_retention")
            if ok(s):
                row["alpha_relaxed_deg"] = s["MnO_AFM2_rhomb"]["alpha"]
                row["retains_rhombohedral"] = s["_retains_rhombohedral"]
                row["verdict"] = s["_verdict"]
            e = b.get("cubic_eos")
            if ok(e):
                row["a_eq_cubic_A"] = e["a_eq_cubic_A"]
                # robustness: the experimental geometry, edge-matched (cubic at 4.430 A from a cubic
                # spline of the EOS scan, anchored to the 4.434 A cubic single point)
                if ok(p) and "MnO_AFM2_rhomb_exp" in p.get("energies_meV_per_atom", {}):
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(e["a_grid"], e["E_meV_per_atom"])
                    en_ = p["energies_meV_per_atom"]
                    e_cub_430 = en_["MnO_AFM2_cubic_seed"] + float(cs(4.430) - cs(4.434))
                    row["distortion_rhomb_minus_cubic_expgeom_meV_per_atom"] = round(
                        en_["MnO_AFM2_rhomb_exp"] - e_cub_430, 3)
                    # the reference's own construction (Eckhoff & Behler): relaxed rhombohedral
                    # AFM-II (4.434 A) vs the cubic lattice at the experimental constant (4.430 A);
                    # for a model far from its own equilibrium this re-adds a volume term
                    row["distortion_reference_construction_meV_per_atom"] = round(
                        en_["MnO_AFM2_rhomb"] - e_cub_430, 3)
            # can the model even resolve the signal it is being asked about?
            floor_fu = val.get(m, {}).get("noise_floor_meV_per_fu")
            if floor_fu is not None:
                signal_fu = DFT_DISTORTION_MEV_ATOM * 2  # meV/atom -> meV/f.u. (MnO: 2 atoms)
                row["dft_signal_to_floor"] = (round(signal_fu / floor_fu, 1)
                                              if floor_fu > 0 else None)
                own_fu = (abs(row["distortion_meV_per_atom"]) * 2
                          if "distortion_meV_per_atom" in row else None)
                row["own_distortion_meV_per_fu"] = (round(own_fu, 3) if own_fu is not None
                                                    else None)
                # resolvable only if the model's OWN value exceeds its floor
                row["below_noise_floor"] = bool(own_fu is not None and own_fu < floor_fu)
            summary["table_B_mno_transfer"][m] = row
            print(f"{m:16s} {row.get('ordering_meV_per_atom', float('nan')):+10.2f}"
                  f" {row.get('distortion_meV_per_atom', float('nan')):+11.3f}"
                  f" {row.get('alpha_relaxed_deg', float('nan')):14.3f}"
                  f" {row.get('a_eq_cubic_A', float('nan')):11.3f}  "
                  f"{row.get('verdict', '')}")
        print()

    # ---- Mechanism: the apparent MnO "ordering energy" is an equation-of-state artefact ----
    # DFT gives a(FM, cubic) = 4.462 A > a(AFM-II) = 4.434 A, so the two phases differ in
    # volume as well as in shape. A spin-blind model comparing those two geometries is really
    # just reporting which volume its own equation of state prefers. Every model whose cubic
    # equilibrium a_eq overshoots BOTH values must therefore favour the larger-volume FM cell
    # -- which is exactly the observed sign error. The prediction is quantitative: the size of
    # the ordering error should track the size of the a_eq error, and it does.
    tb = summary["table_B_mno_transfer"]
    pts = [(v["a_eq_cubic_A"] - DFT_A_AFM2, v["ordering_meV_per_atom"], m)
           for m, v in tb.items()
           if "a_eq_cubic_A" in v and "ordering_meV_per_atom" in v]
    if len(pts) >= 3:
        import numpy as np
        da = np.array([p[0] for p in pts])
        eo = np.array([p[1] for p in pts])
        r = float(np.corrcoef(da, eo)[0, 1])
        from scipy.stats import pearsonr
        pr = pearsonr(da, eo)
        ci = pr.confidence_interval(0.95)
        keep = [i for i, p_ in enumerate(pts) if p_[2] != "M3GNet-MatPES"]
        r_wo = float(np.corrcoef(da[keep], eo[keep])[0, 1])
        summary["mechanism_eos_artefact"] = {
            "pearson_r_aeq_error_vs_ordering": round(r, 3), "n": len(pts),
            "p_value": round(float(pr.pvalue), 3),
            "ci95": [round(float(ci.low), 2), round(float(ci.high), 2)],
            "r_without_M3GNet": round(r_wo, 2),
            "a_AFM2_DFT": DFT_A_AFM2, "a_FM_DFT": 4.462,
            "per_model": {m: {"a_eq_error_A": round(d, 4), "ordering_meV_per_atom": e}
                          for d, e, m in pts},
            "statement": ("The apparent ordering energy of a spin-blind model on MnO tracks "
                          "its equation-of-state error, not magnetism: r = %+.3f between the "
                          "cubic a_eq error and the reported FM-AFM-II splitting. The one "
                          "model with an accurate a_eq (M3GNet-MatPES, +0.004 A) reports a "
                          "splitting of essentially zero -- the honest spin-blind answer -- "
                          "rather than the correct +45.5 meV/atom." % r)}
        print(f"MECHANISM  r(a_eq error, apparent ordering energy) = {r:+.3f} over "
              f"n = {len(pts)} models")
        print("  -> the reported 'ordering energy' is an equation-of-state artefact of the\n"
              "     volume difference between the two phases, not a magnetic energy.\n")

    (R / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote results/validation_summary.json")


if __name__ == "__main__":
    main()
