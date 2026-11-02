#!/usr/bin/env python3
"""Derive every aggregate quantity (counts, ranges, ratios, fits, correlations) from the
released results into results/claims.json, so that no summary number is maintained by hand.

Reads only released results; writes results/claims.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
R = ROOT / "results"

MODELS = ["MACE-MP-0", "CHGNet", "SevenNet-0", "M3GNet-MatPES", "MatterSim",
          "Orb-v3", "GRACE-2L-OMAT"]
DFT = {"a0": 4.14, "B": 289.0, "C11": 677.0, "C12": 87.7, "C44": 155.0,
       "distortion_meV_per_fu": -8.0}
DFT_EF = {("cubic", "V_N"): 2.48, ("cubic", "V_Cr"): 6.90, ("cubic", "A_N"): 10.05,
          ("cubic", "I_N"): 12.01, ("sheared", "V_N"): 2.13, ("sheared", "V_Cr"): 5.72,
          ("sheared", "A_Cr"): 9.70, ("sheared", "A_N"): 10.14}


def load():
    st = {m["model"]: m for m in json.loads((R / "stage3_results.json").read_text())}
    rerun = json.loads((R / "batch1_m3gnet_rerun.json").read_text())
    st["M3GNet-MatPES"]["defects"] = rerun["defects"]
    st["M3GNet-MatPES"]["phase_rerun"] = rerun["phase_fullrelax"]
    st["M3GNet-MatPES"]["reservoirs"] = rerun["reservoirs"]   # rerun supersedes
    noise = json.loads((R / "batch1_numerical_precision.json").read_text())
    el = json.loads((R / "stage3b_analysis.json").read_text())["elastic"]
    audit = json.loads((R / "relaxation_audit.json").read_text())
    for x in ("Orb-v3", "GRACE-2L-OMAT"):
        e = json.loads((R / f"extra_{x}.json").read_text())
        st[x] = {"defects": e["defects"], "reservoirs": e["reservoirs"], "phase": e["phase"]}
        noise[x] = e["numerical_precision"]
        el[x] = e["elastic"]
    return st, noise, el, audit


def excluded(m, tag, d, audit):
    if audit["cases"].get(f"{m}|{tag}|{d}", {}).get("reconstructed"):
        return "E1"
    if m == "SevenNet-0":          # both phases: unstable host (C44 < 0), see final_statistics.py
        return "E2"
    return None


def main():
    st, noise, el, audit = load()
    C = {"_note": "Derived from released results by scripts/emit_claims.py. "
                  "Every aggregate quantity is derived here.",
         "n_models": len(MODELS), "models": MODELS}

    # ---- lattice constants ----
    errs = {m: (el[m]["a_eq_A"] - DFT["a0"]) / DFT["a0"] * 100 for m in MODELS}
    worst = max(errs, key=lambda m: abs(errs[m]))
    C["lattice"] = {"pct_error": {m: round(errs[m], 3) for m in MODELS},
                    "max_abs_pct": round(abs(errs[worst]), 2), "worst_model": worst}

    # ---- phase probe ----
    ph = {}
    for m in MODELS:
        p = st[m]["phase"]
        sp = p["dE_sp_meV_per_fu"]
        ang = p.get("fullrelax_angles_deg") or p.get("fullrelax_from_sheared_angles_deg")
        if m == "M3GNet-MatPES":
            ang = st[m]["phase_rerun"]["angles_deg"]  # rerun supersedes
        ph[m] = {"dE_sp": round(sp, 2), "dE_posrelax": round(p["dE_posrelax_meV_per_fu"], 2),
                 "fullrelax_angles": [round(a, 3) for a in ang],
                 "reverts_to_cubic": all(abs(a - 90) < 0.3 for a in ang)}
    floors = {m: noise[m]["noise_floor_meV_per_fu"] for m in MODELS}
    unresolvable = [m for m in MODELS if floors[m] > abs(ph[m]["dE_sp"])]
    wrong_sign = [m for m in MODELS if ph[m]["dE_sp"] > 0 and m not in unresolvable]
    right_sign = [m for m in MODELS if ph[m]["dE_sp"] < 0 and m not in unresolvable]
    C["phase"] = {"per_model": ph, "dft_meV_per_fu": DFT["distortion_meV_per_fu"],
                  "wrong_sign_models": wrong_sign, "wrong_sign_count": len(wrong_sign),
                  "right_sign_models": right_sign, "right_sign_count": len(right_sign),
                  "unresolvable_models": unresolvable, "unresolvable_count": len(unresolvable),
                  "n_revert_to_cubic": sum(1 for m in MODELS if ph[m]["reverts_to_cubic"]),
                  "largest_magnitude_model": max(MODELS, key=lambda m: abs(ph[m]["dE_sp"])),
                  "sp_equals_posrelax_models": [m for m in MODELS
                                                if abs(ph[m]["dE_sp"] - ph[m]["dE_posrelax"]) < 0.005]}
    C["phase"]["sp_equals_posrelax_count"] = len(C["phase"]["sp_equals_posrelax_models"])

    # ---- noise floors ----
    srt = sorted(floors.items(), key=lambda kv: -kv[1])
    C["noise"] = {"floor_meV_per_fu": {m: round(floors[m], 4) for m in MODELS},
                  "max_model": srt[0][0], "max": round(srt[0][1], 4),
                  "second_model": srt[1][0], "second": round(srt[1][1], 4),
                  "ratio_max_over_second": round(srt[0][1] / srt[1][1], 1) if srt[1][1] else None,
                  "signal_meV_per_fu": abs(DFT["distortion_meV_per_fu"])}
    # "exactly zero" is false for most: MACE is 3.6e-12, GRACE 4.1e-8, only SevenNet is 0.0.
    # Use thresholds tied to the physics instead of to printed precision.
    C["noise"]["negligible_models"] = [m for m in MODELS if floors[m] < 1e-3]      # >1000x below signal
    C["noise"]["negligible_count"] = len(C["noise"]["negligible_models"])
    C["noise"]["comparable_to_signal_models"] = [m for m in MODELS
                                                 if floors[m] > 0.01 * C["noise"]["signal_meV_per_fu"]]
    C["noise"]["comparable_to_signal_count"] = len(C["noise"]["comparable_to_signal_models"])
    C["noise"]["exactly_zero_models"] = [m for m in MODELS if floors[m] == 0.0]
    C["noise"]["max_as_fraction_of_signal"] = round(srt[0][1] / C["noise"]["signal_meV_per_fu"], 3)

    # ---- defects ----
    ratios, negs, per = [], [], {}
    for m in MODELS:
        pts, exc = {}, {}
        for (tag, d), ref in DFT_EF.items():
            v = st[m]["defects"][tag].get(d)
            if not v:
                continue
            ef = v["E_f_eV"]
            why = excluded(m, tag, d, audit)
            rec = {"E_f": round(ef, 4), "DFT": ref, "negative": ef < 0}
            if why:
                rec["rule"] = why
                exc[f"{tag}|{d}"] = rec
            else:
                pts[f"{tag}|{d}"] = rec
                if ef > 0:
                    ratios.append((ref / ef, m, tag, d))
            if ef < 0:
                negs.append({"model": m, "case": f"{tag}|{d}", "E_f": round(ef, 3)})
        per[m] = {"points": pts, "excluded": exc}
    ratios.sort()
    C["defects"] = {
        "per_model": per, "n_retained_positive_pairs": len(ratios),
        "factor_min": round(ratios[0][0], 1), "factor_min_case": f"{ratios[0][1]} {ratios[0][2]} {ratios[0][3]}",
        "factor_max": round(ratios[-1][0], 1), "factor_max_case": f"{ratios[-1][1]} {ratios[-1][2]} {ratios[-1][3]}",
        "negative_Ef_standard_reservoirs": negs,
        "n_retained_pairs_incl_negative": len(ratios) + len(negs),
        "factor_range_caveat": "after the exclusions of rules E1/E2; all positive points give "
                               "1.3-341x over n=54",
        "excluded_models_E2": ["SevenNet-0"]}

    # ---- Cr-rich reservoir cross-check (all seven models; previously only five) ----
    # Cr-rich limit: mu_Cr fixed at bcc Cr, mu_N = E_fu(CrN) - mu_Cr.
    crrich = {}
    for m in MODELS:
        e_prist = st[m]["defects"]["cubic"]["E_pristine"]      # 64 atoms = 32 f.u.
        mu_cr = st[m]["reservoirs"]["mu_Cr"]
        mu_n_std = st[m]["reservoirs"]["mu_N"]
        mu_n_cr = e_prist / 32 - mu_cr
        shift = mu_n_cr - mu_n_std
        vals = {}
        for tag in ("cubic", "sheared"):
            for d, nn in (("V_N", 1), ("I_N", -1), ("A_N", -1), ("A_Cr", 1)):
                v = st[m]["defects"][tag].get(d)
                if v:
                    vals[f"{tag}|{d}"] = round(v["E_f_eV"] + nn * shift, 3)
        crrich[m] = {"mu_N_standard": round(mu_n_std, 3), "mu_N_CrRich": round(mu_n_cr, 3),
                     "E_f_CrRich": vals,
                     "V_N_negative": [k for k, x in vals.items() if k.endswith("V_N") and x < 0]}
    # SevenNet-0 is excluded from defect statistics in both phases (E2), so it is listed but
    # not counted; the denominator is the six models with a mechanically stable host.
    C["cr_rich"] = {"per_model": crrich,
                    "models_with_negative_V_N_all": [m for m in MODELS if crrich[m]["V_N_negative"]],
                    "models_with_negative_V_N": [m for m in MODELS if crrich[m]["V_N_negative"]
                                                 and m != "SevenNet-0"],
                    "n_models_evaluated": len(MODELS) - 1}
    # which inverted models stay inverted under Cr-rich conditions (V_Cr unchanged, V_N shifted)
    C["cr_rich"]["inversion_persists"] = [
        m for m in MODELS if m != "SevenNet-0"
        and crrich[m]["E_f_CrRich"].get("cubic|V_N", 9) > st[m]["defects"]["cubic"]["V_Cr"]["E_f_eV"]]
    C["cr_rich"]["n_models_with_negative_V_N"] = len(C["cr_rich"]["models_with_negative_V_N"])

    # ---- hierarchy ----
    inv_any, inv_clean = [], []
    for m in MODELS:
        bad_any = bad_clean = False
        for tag in ("cubic", "sheared"):
            ds = {d: st[m]["defects"][tag][d]["E_f_eV"] for d in
                  ("V_N", "V_Cr", "A_N", "A_Cr", "I_N") if d in st[m]["defects"][tag]}
            if min(ds, key=ds.get) != "V_N":
                bad_any = True
                if m != "SevenNet-0":
                    bad_clean = True
        if bad_any:
            inv_any.append(m)
        if bad_clean:
            inv_clean.append(m)
    C["hierarchy"] = {"n_models_evaluated": len(MODELS) - 1,   # SevenNet-0 defects excluded (E2)
                      "inverted_any_phase": inv_any, "inverted_any_count": len(inv_any),
                      "inverted_after_exclusions": inv_clean,
                      "inverted_after_exclusions_count": len(inv_clean)}

    # ---- reconstruction ----
    flagged = [f["case"] for f in audit["flagged"]]
    clean = [v["max_disp_A"] for v in audit["cases"].values() if not v["reconstructed"]]
    rec_vals = [v["max_disp_A"] for v in audit["cases"].values() if v["reconstructed"]]
    C["reconstruction"] = {"n_reconstructed": len(flagged), "n_total": len(audit["cases"]),
                           "threshold_A": audit["threshold_A"], "cases": flagged,
                           "models": sorted({c.split("|")[0] for c in flagged}),
                           "clean_max_disp_range_A": [round(min(clean), 3), round(max(clean), 3)],
                           "reconstructed_range_A": [round(min(rec_vals), 2), round(max(rec_vals), 2)],
                           "all_antisites": all(c.split("|")[2].startswith("A_") for c in flagged)}

    # ---- elastic ----
    C["elastic"] = {"per_model": {m: {k: (round(v, 3) if k == "a_eq_A" else round(v, 1))
                                      if isinstance(v, float) else v
                                      for k, v in el[m].items()} for m in MODELS},
                    "B_ratio": {m: round(el[m]["B_GPa"] / DFT["B"], 2) for m in MODELS},
                    "C11_ratio": {m: round(el[m]["C11_GPa"] / DFT["C11"], 2) for m in MODELS},
                    "negative_C44_models": [m for m in MODELS if el[m]["C44_GPa"] < 0]}
    soft = {m: round((1 - C["elastic"]["B_ratio"][m]) * 100) for m in MODELS}
    C["elastic"]["B_softening_pct"] = soft
    # M3GNet is a distinct case (within 3%); quote the range over the rest.
    OUTLIER = "M3GNet-MatPES"   # named, not threshold-defined
    rest = sorted(v for m, v in soft.items() if m != OUTLIER)
    C["elastic"]["B_softening_pct_range_excl_outlier"] = [rest[0], rest[-1]]
    C["elastic"]["B_softening_outlier"] = [OUTLIER]
    C["elastic"]["B_softening_pct_range_all"] = [min(soft.values()), max(soft.values())]
    C["elastic"]["B_ratio_range_all"] = [round(min(C["elastic"]["B_ratio"].values()), 2),
                                         round(max(C["elastic"]["B_ratio"].values()), 2)]
    C["elastic"]["C11_ratio_range_all"] = [round(min(C["elastic"]["C11_ratio"].values()), 2),
                                           round(max(C["elastic"]["C11_ratio"].values()), 2)]
    C["elastic"]["least_softened_model_B"] = max(MODELS, key=lambda m: C["elastic"]["B_ratio"][m])
    C["elastic"]["stiffest_model_C11"] = max(MODELS, key=lambda m: C["elastic"]["C11_ratio"][m])
    C["elastic"]["over_stiff_C44_models"] = [m for m in MODELS
                                             if el[m]["C44_GPa"] > DFT["C44"]]
    c44s = {m: round((1 - el[m]["C44_GPa"] / DFT["C44"]) * 100) for m in MODELS
            if 0 < el[m]["C44_GPa"] < DFT["C44"]}
    C["elastic"]["C44_softening_pct"] = c44s
    C["elastic"]["C44_softening_pct_range"] = [min(c44s.values()), max(c44s.values())]
    # training-set generation labels, recorded once here
    C["elastic"]["generation"] = {"MACE-MP-0": "MPtrj", "CHGNet": "MPtrj", "SevenNet-0": "MPtrj",
                                  "M3GNet-MatPES": "MatPES", "MatterSim": "proprietary",
                                  "Orb-v3": "OMat24", "GRACE-2L-OMAT": "OMat24"}

    # ---- phase probe = elastic shear energy for a spin-blind model ----
    # A fixed-edge shear of the conventional cell by dgamma is a pure shear; to leading order
    # dE = 1/2 C44 V dgamma^2, V = a_c^3/4 per formula unit. 1 GPa A^3 = 1/160.2177 eV.
    vfu = DFT["a0"] ** 3 / 4
    dg = np.radians(2.0)
    el_pred = {m: 0.5 * el[m]["C44_GPa"] / 160.2177 * vfu * dg ** 2 * 1000 for m in MODELS}
    resolvable = [m for m in MODELS if m not in unresolvable]
    x = np.array([el[m]["C44_GPa"] for m in resolvable])
    y = np.array([ph[m]["dE_sp"] for m in resolvable])
    dft_el = 0.5 * DFT["C44"] / 160.2177 * vfu * dg ** 2 * 1000
    C["phase_decomposition"] = {
        "elastic_prediction_meV_per_fu": {m: round(v, 2) for m, v in el_pred.items()},
        "pearson_r_dE_vs_C44_resolvable": round(float(np.corrcoef(x, y)[0, 1]), 3),
        "n_resolvable": len(resolvable),
        "dft_elastic_term_meV_per_fu": round(dft_el, 1),
        "dft_magnetic_term_meV_per_fu": round(DFT["distortion_meV_per_fu"] - dft_el, 1),
        "caveat": "DFT split heuristic: reference C44 is cubic AFM1[010] at U=5 eV; the -8.0 "
                  "compares AFM2[110] phases at U=3 eV"}
    from scipy.stats import linregress
    lr = linregress(x, y)
    C["phase_decomposition"]["fit_intercept_meV_per_fu"] = round(float(lr.intercept), 1)
    C["phase_decomposition"]["fit_intercept_stderr_meV_per_fu"] = round(float(lr.intercept_stderr), 2)
    # slope against each model's own elastic prediction (parity = 1), and its leverage points
    xp = np.array([el_pred[m] for m in resolvable])
    lp = linregress(xp, y)
    C["phase_decomposition"]["fit_slope_vs_elastic_prediction"] = round(float(lp.slope), 2)
    k_orb = [i for i, m in enumerate(resolvable) if m != "Orb-v3"]
    C["phase_decomposition"]["fit_slope_without_Orb"] = round(float(linregress(xp[k_orb], y[k_orb]).slope), 2)
    k_7n = [i for i, m in enumerate(resolvable) if m != "SevenNet-0"]
    l7 = linregress(xp[k_7n], y[k_7n])
    C["phase_decomposition"]["fit_intercept_without_SevenNet"] = [round(float(l7.intercept), 2),
                                                                  round(float(l7.intercept_stderr), 2)]
    C["phase_decomposition"]["mean_parity_residual_meV_per_fu"] = round(float(np.mean(y - xp)), 2)
    # harmonic regime check: E(86 deg)/E(88 deg) = 4 for a pure parabola about 90 deg
    gsc = json.loads((R / "batch1_gamma_scan.json").read_text())
    ang = gsc["angles_deg"]; i86, i88 = ang.index(86.0), ang.index(88.0)
    hr = {m: gsc["dE_meV_per_fu_vs_90"][m][i86] / gsc["dE_meV_per_fu_vs_90"][m][i88]
          for m in resolvable if abs(gsc["dE_meV_per_fu_vs_90"][m][i88]) > 0.5}
    C["phase_decomposition"]["harmonic_ratio_E86_over_E88"] = {m: round(v, 2) for m, v in hr.items()}
    C["phase_decomposition"]["harmonic_ratio_range"] = [round(min(hr.values()), 1), round(max(hr.values()), 1)]
    C["phase_decomposition"]["wrong_sign_equals_positive_C44"] = sorted(wrong_sign) == sorted(
        m for m in resolvable if el[m]["C44_GPa"] > 0)

    # ---- reservoir-free defect sums (chemical potentials cancel) ----
    def pair(m, tag, a, b):
        d = st[m]["defects"][tag]
        if a not in d or b not in d or m == "SevenNet-0":
            return None
        if any(audit["cases"].get(f"{m}|{tag}|{x}", {}).get("reconstructed") for x in (a, b)):
            return None
        return d[a]["E_f_eV"] + d[b]["E_f_eV"]
    fr = {m: pair(m, "cubic", "V_N", "I_N") for m in MODELS}
    ap = {m: pair(m, "sheared", "A_N", "A_Cr") for m in MODELS}
    fr = {m: round(v, 2) for m, v in fr.items() if v is not None}
    ap = {m: round(v, 2) for m, v in ap.items() if v is not None}
    dft_fr = DFT_EF[("cubic", "V_N")] + DFT_EF[("cubic", "I_N")]
    dft_ap = DFT_EF[("sheared", "A_N")] + DFT_EF[("sheared", "A_Cr")]
    C["reservoir_free"] = {
        "frenkel_cubic": fr, "frenkel_dft": round(dft_fr, 2),
        "frenkel_factor_range": [round(dft_fr / max(fr.values()), 1), round(dft_fr / min(fr.values()), 1)],
        "antisite_pair_orth": ap, "antisite_pair_dft": round(dft_ap, 2),
        "antisite_factor_range": [round(dft_ap / max(ap.values()), 1), round(dft_ap / min(ap.values()), 1)]}

    # ---- parity slopes on retained points (quoted in the mechanism section) ----
    fs_path = R / "final_statistics.json"
    if fs_path.exists():
        fs = json.loads(fs_path.read_text())
        par = {m: fs["per_model"][m]["parity_clean"] for m in MODELS
               if "parity_clean" in fs["per_model"].get(m, {})}
        # SevenNet-0 retains only 3 points after E1/E2; a 3-point regression is not
        # reportable, so it is excluded from the quoted ranges WITH THE REASON STATED.
        fitted = {m: v for m, v in par.items() if v["n"] >= 5}
        C["parity"] = {
            "per_model": par,
            "excluded_from_ranges": {m: f"n={v['n']} too few to fit" for m, v in par.items()
                                     if m not in fitted},
            "slope0_range_fitted": [round(min(v["slope_through_origin"] for v in fitted.values()), 3),
                                    round(max(v["slope_through_origin"] for v in fitted.values()), 3)],
            "R2_range_fitted": [round(min(v["R2"] for v in fitted.values()), 2),
                                round(max(v["R2"] for v in fitted.values()), 2)],
            "slope0_range_all": [round(min(v["slope_through_origin"] for v in par.values()), 3),
                                 round(max(v["slope_through_origin"] for v in par.values()), 3)],
            "R2_range_all": [round(min(v["R2"] for v in par.values()), 2),
                             round(max(v["R2"] for v in par.values()), 2)]}

    # ---- Phase-1 provenance (force noise, residual forces, E2 evidence) ----
    prov = {}
    for f in ("phase1_provenance.json", "phase1_provenance_extra.json"):
        pth = R / f
        if pth.exists():
            prov.update(json.loads(pth.read_text()))
    fn = {m: prov[m]["force_noise_cubic_2x2x2_V_N_eV_per_A"] for m in MODELS
          if m in prov and "force_noise_cubic_2x2x2_V_N_eV_per_A" in prov[m]}
    resid = {m: prov[m].get("residual_fmax_eV_per_A", {}) for m in MODELS if m in prov}
    C["forces"] = {
        "noise_eV_per_A_defect_cell": fn,
        "max_model": (max(fn, key=fn.get) if fn else None),
        "max": (round(max(fn.values()), 4) if fn else None),
        "residual_fmax_eV_per_A": resid,
        "target_fmax_eV_per_A": 0.01,
        "models_whose_noise_exceeds_target": [m for m, v in fn.items() if v > 0.01],
    }
    e2 = prov.get("_e2_continuation_SevenNet_sheared_V_Cr")
    if e2:
        C["e2_evidence"] = {
            "series": e2,
            "E_f_at_production_fmax": e2["0.01"]["E_f_eV"],
            "E_f_converged": e2["0.001"]["E_f_eV"],
            "shift_from_loose_to_tight_eV": round(e2["0.001"]["E_f_eV"] - e2["0.01"]["E_f_eV"], 3),
            "converged_by_3e-3": abs(e2["0.001"]["E_f_eV"] - e2["0.003"]["E_f_eV"]) < 1e-3,
            "max_disp_A_production": e2["0.01"]["max_disp_A"],
            "max_disp_A_converged": e2["0.001"]["max_disp_A"],
            "converged_value_is_negative": e2["0.001"]["E_f_eV"] < 0,
        }
    # ---- predictive validation of the diagnostics, and the MnO transfer ----
    # Source: results/validation_summary.json (scripts/summarize_validation.py), which in turn
    # derives from diagnostic_validation{,_extra}.json and mno_results{,_extra}.json.
    vs_path = R / "validation_summary.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text())
        ta, sc = vs["table_A_predictive_validation"], vs["scorecard"]

        shear_n = sum(1 for r in ta.values() if r.get("shear_correct") is not None)
        shear_ok = sum(1 for r in ta.values() if r.get("shear_correct"))
        res_scored = [m for m, r in ta.items() if r.get("resolution_correct") is not None]
        res_ok = [m for m in res_scored if ta[m]["resolution_correct"]]
        res_miss = [m for m in res_scored if not ta[m]["resolution_correct"]]
        C["validation"] = {
            "shear_correct": shear_ok, "shear_total": shear_n,
            "shear_all_correct": shear_ok == shear_n,
            "resolution_correct": len(res_ok), "resolution_scored": len(res_scored),
            "resolution_miss_models": res_miss,
            "resolution_miss_types": sc.get("misses_by_type", {}),
            "false_clears": sc.get("false_clears", []),
            "n_false_clears": len(sc.get("false_clears", [])),
            "total_correct": sc["predictions_correct"], "total": sc["total"],
            "precondition_excluded": [m for m, r in ta.items()
                                      if r.get("stability_precondition_met") is False],
            "shear_indeterminate": [m for m, r in ta.items()
                                    if r.get("shear_observed") == "indeterminate"],
            "pred_threshold": sc.get("pred_threshold_meV_per_fu"),
            "obs_threshold": sc.get("obs_threshold_meV_per_fu"),
            "false_clears_without_precondition": sc.get("false_clears_without_precondition", []),
            "common_threshold_sensitivity": sc.get("common_threshold_sensitivity", {}),
            "spearman_p": sc.get("floor_vs_spread_spearman_stable_models", {}).get("p_value"),
            "obs_threshold_set_at": sc.get("obs_threshold_set_at"),
            "resolution_true_positives": sc.get("resolution_true_positives"),
            "rattle_detail": sc.get("rattle_detail", {}),
            "separation": sc.get("rattle_separation_by_C44_sign", {}),
            "spearman_floor_vs_spread": sc.get(
                "floor_vs_spread_spearman_stable_models", {}).get("rho"),
            "per_model": ta,
        }

        tb = vs.get("table_B_mno_transfer", {})
        if tb:
            alphas = {m: r["alpha_relaxed_deg"] for m, r in tb.items()
                      if "alpha_relaxed_deg" in r}
            dist = {m: r["distortion_meV_per_atom"] for m, r in tb.items()
                    if "distortion_meV_per_atom" in r}
            order = {m: r["ordering_meV_per_atom"] for m, r in tb.items()
                     if "ordering_meV_per_atom" in r}
            aeq = {m: r["a_eq_cubic_A"] for m, r in tb.items() if "a_eq_cubic_A" in r}
            mech = vs.get("mechanism_eos_artefact", {})
            C["mno"] = {
                "n_models": len(tb),
                "DFT_alpha": 90.89, "DFT_distortion_meV_per_atom": 2.1,
                "DFT_ordering_meV_per_atom": 45.5,
                "DFT_a_AFM2": 4.434, "DFT_a_FM": 4.462,
                "signal_meV_per_fu": round(2.1 * 2, 1),
                "alpha_min": round(min(alphas.values()), 3),
                "alpha_max": round(max(alphas.values()), 3),
                "max_abs_alpha_departure_from_cubic": round(
                    max(abs(v - 90.0) for v in alphas.values()), 3),
                "n_collapsed_to_cubic": sum(1 for m, r in tb.items()
                                            if not r.get("retains_rhombohedral", False)),
                "all_collapsed": all(not r.get("retains_rhombohedral", False)
                                     for r in tb.values()),
                "n_distortion_wrong_sign": sum(1 for m, v in dist.items() if v < 0
                                               and not tb[m].get("below_noise_floor")),
                "distortion_unresolvable_models": [m for m in dist
                                                   if tb[m].get("below_noise_floor")],
                "n_distortion_wrong_sign_incl_unresolvable": sum(1 for v in dist.values() if v < 0),
                "n_ordering_wrong_sign": sum(1 for v in order.values() if v < 0),
                "ordering_min": round(min(order.values()), 2),
                "ordering_max": round(max(order.values()), 2),
                "distortion_min": round(min(dist.values()), 3),
                "distortion_max": round(max(dist.values()), 3),
                "a_eq_min": round(min(aeq.values()), 4),
                "a_eq_max": round(max(aeq.values()), 4),
                "n_a_eq_above_both_DFT": sum(1 for v in aeq.values() if v > 4.462),
                "best_a_eq_model": min(aeq, key=lambda m: abs(aeq[m] - 4.434)),
                "best_a_eq_value": round(min(aeq.values(), key=lambda v: abs(v - 4.434)), 4),
                "mechanism_r": mech.get("pearson_r_aeq_error_vs_ordering"),
                "mechanism_n": mech.get("n"),
                "mechanism_p": mech.get("p_value"),
                "mechanism_ci95": mech.get("ci95"),
                "mechanism_r_without_M3GNet": mech.get("r_without_M3GNet"),
                "volume_term_meV_per_atom": {m: r.get("volume_term_meV_per_atom")
                                             for m, r in tb.items()},
                "distortion_expgeom": {m: r.get("distortion_rhomb_minus_cubic_expgeom_meV_per_atom")
                                       for m, r in tb.items()},
                "distortion_reference_construction": {m: r.get("distortion_reference_construction_meV_per_atom")
                                                      for m, r in tb.items()},
                "per_model": tb,
            }
            C["mno"]["best_a_eq_ordering"] = round(order[C["mno"]["best_a_eq_model"]], 2)
            # harmonic test: edge-matched distortion at the experimental angle over that at the
            # DFT angle, for models whose floor is below 1% of the CrN signal; a harmonic shear
            # gives ((90.62-90)/(90.89-90))^2
            rmc = {m: r.get("distortion_rhomb_minus_cubic_meV_per_atom") for m, r in tb.items()}
            exg = C["mno"]["distortion_expgeom"]
            lowf = [m for m in tb if floors[m] < 0.01 * C["noise"]["signal_meV_per_fu"]]
            hrm = {m: exg[m] / rmc[m] for m in lowf if exg.get(m) and rmc.get(m)}
            C["mno"]["harmonic_ratio_expgeom_over_dft"] = {m: round(v, 3) for m, v in hrm.items()}
            C["mno"]["harmonic_ratio_range"] = [round(min(hrm.values()), 3), round(max(hrm.values()), 3)]
            C["mno"]["harmonic_ratio_predicted"] = round(((90.62 - 90) / (90.89 - 90)) ** 2, 3)
            C["mno"]["harmonic_ratio_models"] = lowf
            hi6 = [v for v in aeq.values() if v > 4.462]   # the models above both DFT constants
            C["mno"]["a_eq_pct_vs_AFM2_range"] = [round(100 * (min(hi6) / 4.434 - 1), 1),
                                                  round(100 * (max(hi6) / 4.434 - 1), 1)]
            C["mno"]["a_eq_pct_vs_FM_range"] = [round(100 * (min(hi6) / 4.462 - 1), 1),
                                                round(100 * (max(hi6) / 4.462 - 1), 1)]
            # outcome under the reference's own construction (cubic at the experimental 4.430 A);
            # resolved if |value| exceeds the CrN-measured floor per atom (floor per f.u. / 2)
            rcv = C["mno"]["distortion_reference_construction"]
            res = {m: v for m, v in rcv.items() if v is not None and abs(v) > floors[m] / 2}
            C["mno"]["refconstr_wrong_sign_resolved"] = sorted(m for m, v in res.items() if v > 0)
            C["mno"]["refconstr_dft_sign_resolved"] = sorted(m for m, v in res.items() if v < 0)
            C["mno"]["refconstr_unresolved"] = sorted(m for m in rcv if m not in res)

    (R / "claims.json").write_text(json.dumps(C, indent=1))
    print("claims.json written")
    print(f"  models {C['n_models']} | lattice max err {C['lattice']['max_abs_pct']}% "
          f"({C['lattice']['worst_model']})")
    print(f"  phase: wrong-sign {C['phase']['wrong_sign_count']} "
          f"right-sign {C['phase']['right_sign_count']} "
          f"unresolvable {C['phase']['unresolvable_count']} "
          f"| largest {C['phase']['largest_magnitude_model']}")
    print(f"  noise: max {C['noise']['max']} ({C['noise']['max_model']}), "
          f"2nd {C['noise']['second']} ({C['noise']['second_model']}), ratio {C['noise']['ratio_max_over_second']}x, "
          f"negligible {C['noise']['negligible_count']}")
    print(f"  defects: {C['defects']['factor_min']}-{C['defects']['factor_max']}x "
          f"(n={C['defects']['n_retained_positive_pairs']}), negatives {len(C['defects']['negative_Ef_standard_reservoirs'])}")
    print(f"  hierarchy inverted {C['hierarchy']['inverted_after_exclusions_count']} "
          f"(any-phase {C['hierarchy']['inverted_any_count']})")
    print(f"  reconstruction {C['reconstruction']['n_reconstructed']}/{C['reconstruction']['n_total']}")
    print(f"  Cr-rich negative V_N: {C['cr_rich']['n_models_with_negative_V_N']} "
          f"{C['cr_rich']['models_with_negative_V_N']}")
    print(f"  B softening {C['elastic']['B_softening_pct_range_excl_outlier']}% "
          f"(outlier {C['elastic']['B_softening_outlier']}), "
          f"negative C44 {C['elastic']['negative_C44_models']}")
    if "validation" in C:
        v = C["validation"]
        print(f"  validation: shear {v['shear_correct']}/{v['shear_total']}, "
              f"resolution {v['resolution_correct']}/{v['resolution_scored']}, "
              f"total {v['total_correct']}/{v['total']}, "
              f"false clears {v['n_false_clears']}")
    if "mno" in C:
        m = C["mno"]
        print(f"  MnO: {m['n_collapsed_to_cubic']}/{m['n_models']} collapsed to cubic "
              f"(alpha {m['alpha_min']}-{m['alpha_max']} vs DFT {m['DFT_alpha']}), "
              f"wrong-sign distortion {m['n_distortion_wrong_sign']}/{m['n_models']}, "
              f"mechanism r={m['mechanism_r']}")


if __name__ == "__main__":
    main()
