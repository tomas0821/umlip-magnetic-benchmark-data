# Universal machine-learning interatomic potentials on antiferromagnetic CrN and MnO: reference data, structures, model outputs and analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23864936.svg)](https://doi.org/10.5281/zenodo.23864936)

Reference data, input structures, raw model outputs, derived quantities and analysis code for a
zero-shot evaluation of seven released, spin-blind universal machine-learning interatomic
potentials (MACE-MP-0, CHGNet, SevenNet-0, M3GNet-MatPES, MatterSim, Orb-v3, GRACE-2L-OMAT) on
the magnetic energetics of antiferromagnetic CrN and MnO. The observables are magnetostructural
distortion and ordering energies, neutral native-defect formation energies and relaxations,
elastic constants, CHGNet magnetic moments, numerical noise floors and three preflight
diagnostics. **No new DFT calculations were performed**: every reference value is taken from
published work, with provenance recorded per row.

## Contents

```
reference_data.csv          CrN reference values (LSDA+U; Rojas & Ulloa, PRB 96, 125203 (2017);
                            PRB 98, 214111 (2018)), one row per value, with provenance
reference_data_MnO.csv      MnO reference values (HSE06; Eckhoff & Behler, npj Comput. Mater. 7,
                            170 (2021)), with provenance
reference_data_README.md    conventions, sign definitions and scope of the reference tables
structures/                 all input cells (extxyz) and the script that builds them
scripts/                    inference, analysis and scoring scripts; pinned environments
results/                    raw model outputs, derived quantities (claims.json), relaxed geometries
```

## Reproducing

Two Python environments are needed because the model packages have incompatible dependencies:

```bash
uv venv ~/.venvs/umlip  --python 3.11 && uv pip install --python ~/.venvs/umlip/bin/python  -r scripts/requirements.lock.txt
uv venv ~/.venvs/umlip2 --python 3.11 && uv pip install --python ~/.venvs/umlip2/bin/python -r scripts/requirements_extra.lock.txt
```

`e3nn` must stay at 0.4.4 in the first environment (0.5.9+ silently breaks MACE checkpoint loading).
M3GNet-MatPES is loaded at Hugging Face revision `8414b23e68b173d6e36271e4366ee3717364051c`. The
checkpoint name `M3GNet-PES-MatPES-PBE-2025.2` was re-uploaded after the production runs, and the
later revision gives different energies (−73.69 vs −74.76 eV for the 8-atom cubic CrN cell), so all
scripts pin the revision; release v1.0.0 loaded the model by name and does not reproduce the
M3GNet results.
Run one model per process (`scripts/run_validation.sh` does this).

```bash
python structures/build_structures.py                 # CrN cells
python scripts/build_mno.py                           # MnO cells
python scripts/stage3_run.py                          # CrN inference, five models (env 1)
python scripts/extra_models.py Orb-v3 GRACE-2L-OMAT   # CrN inference, two models (env 2)
python scripts/stage3b_analysis.py                    # moments, displacements, elastic constants, reservoirs
python scripts/batch1_fixes.py                        # noise floors, M3GNet rerun, tolerance sensitivity, shear scan
python scripts/phase1_provenance.py                   # force noise, residual forces, exclusion evidence
python scripts/audit_relaxations.py                   # reconstruction audit (70 relaxations)
./scripts/run_validation.sh                           # preflight-diagnostic tests (diagnostic_validation.py)
SCRIPT=scripts/run_mno.py ./scripts/run_validation.sh # MnO
python scripts/final_statistics.py                    # per-model statistics with exclusion rules
python scripts/summarize_validation.py                # scores the preflight diagnostics; MnO summary
python scripts/emit_claims.py                         # -> results/claims.json
python scripts/precision_control.py                   # float32/float64 control (GRACE, env 2)
python scripts/precision_control_m3gnet.py            # float32/float64 control (M3GNet, env 1)
```

`results/claims.json` collects every derived aggregate quantity (counts, ranges, ratios, fits and
correlations); it is regenerated deterministically from the raw results by `scripts/emit_claims.py`.

## Exclusion rules applied to the defect statistics

| Rule | Scope | Reason |
|---|---|---|
| E1 | 11 of 70 relaxations, all antisites | the relaxation left the intended defect configuration (maximum displacement > 0.9 Å, in an empty interval of the observed distribution) |
| E2 | SevenNet-0, both phases | its host lattice is not a minimum: C44 < 0, rattled cubic cells relax ~20 meV/f.u. below the symmetric reference, and the sheared-phase vacancy relaxation runs away |

Both rules are applied in `scripts/final_statistics.py` and `scripts/emit_claims.py`; statistics
without them are also recorded.

## Licence

Code (`scripts/`, `structures/build_structures.py`): MIT (see `LICENSE`).
Data, structures and results: CC BY 4.0 (see `LICENSE-DATA`).

## Citation

Please cite the archived release, DOI [10.5281/zenodo.23864936](https://doi.org/10.5281/zenodo.23864936)
(concept DOI, resolving to the latest version); see `CITATION.cff`.
