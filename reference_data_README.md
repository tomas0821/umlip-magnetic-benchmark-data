# Reference tables: conventions and scope

`reference_data.csv` holds the CrN reference values, extracted from T. Rojas and S. E. Ulloa,
Phys. Rev. B 96, 125203 (2017) (`rojasStrainFieldsElectronic2017`) and Phys. Rev. B 98, 214111
(2018) (`rojasEnergeticsElectronicStructure2018`). `reference_data_MnO.csv` holds the MnO values
from Eckhoff and Behler, npj Comput. Mater. 7, 170 (2021). Every row records its source and the
location of the value in it.

## Conventions

- **Ordering energies** are relative to the orthorhombic AFM²[110] ground state, positive =
  higher in energy, per chemical unit (f.u. = one CrN).
- **Formation energies**: neutral unless a charge state `q` is given; E_F = 0 at the VBM;
  chemical-potential reservoirs are the N₂ molecule and AFM bulk Cr (Eq. 1 of the 2018 paper,
  Freysoldt-corrected).
- **Displacements** (2018 Table I): positive = motion away from the defect; entries with two
  semicolon-separated values are two inequivalent neighbours of the same type.
- **`uMLIP_observable`**: `yes` = comparable via total energies, forces or relaxations;
  `no (charged defect)` = requires charged-supercell electrostatics that no uMLIP provides,
  kept for completeness; `magmom (CHGNet-class only)` = compare only against models that emit
  magnetic moments; `context` = methodological, not a comparison target.

## Scope

- **Not extracted**: deformation potentials, effective masses, band gaps and gap-closure
  strains (2017), dielectric constant and carrier densities (2018). These are electronic-
  structure observables that uMLIPs do not produce.
- **Extracted**: elastic constants (2017 Sec. III A), which are second derivatives of the
  energy and therefore directly comparable.
- Literature values quoted inside the source papers (other authors' distortion angles, elastic
  constants of Cr metal and of GaN/AlN, band gaps) are not extracted.

## Notes on individual rows

1. The 2018 Sec. III A text prints "A_Cr at 10.05 eV" for the cubic AFM¹[010] model. This is a
   typographical error for **A_N**, consistent with Sec. II (A_Cr did not converge in AFM¹) and
   with Table I, which lists A_N for that phase, and confirmed from the original calculation
   records by an author of the reference. The row is recorded as A_N.
2. The 2017 paper reports no numeric magnetic moments; the pristine Cr moment (2.90 μ_B) is
   from Sec. III D of the 2018 paper.
3. The 2017 paper uses U = 3 eV for AFM²[110] and U = 5 eV for AFM¹[010] (J = 0.94 eV for both)
   to give the two orders comparable band gaps. The `U_eV` column records these production
   values. The 8.0 meV/f.u. cubic-versus-orthorhombic AFM²[110] value compares two cells at
   the same U.
