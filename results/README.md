# results/

Everything here is produced by `pipeline/compute_all.py` and
`scripts/validation/{self_gravity_effect,him_mass_fraction}.py`, reading
only `src/physics/` and `cache/core/alpha_core.zarr`. No figures, no
paper content. To regenerate:

```
python pipeline/compute_all.py build [VARIANT ...]   # resumable, per-variant
python pipeline/compute_all.py finalize [VARIANT ...]  # resumable, per-variant
python pipeline/compute_all.py merge                   # or let finalize auto-merge
python scripts/validation/self_gravity_effect.py
python scripts/validation/him_mass_fraction.py
```

## The three self-gravity modes

Every pressure/alpha number in this project is computed under one of
three gas self-gravity settings (`self_gravity` column below):

- **`off`** -- hydrostatic balance against Guo+20's external field only
  (stellar disk + dark matter halo). No gas self-gravity.
- **`mean`** (`footprint_mean`, **DEFAULT**) -- gas self-gravity from a
  single, horizontally averaged density profile: at each z, the density
  used as the pressure weight (RAW/HIM_A/HIM_B's `n_model`) is averaged
  over the entire +-500 pc square footprint (NaN cells excluded from the
  mean, not zeroed), then that one 1D profile's enclosed column,
  `g_gas(z) = 2*pi*G*Sigma_bar(|z'| < |z|)`, is applied identically to
  every sightline. This matches the physical picture behind Guo+20's
  external field: a smooth, horizontally uniform disk.
- **`column`** (`per_column`, sensitivity option) -- gas self-gravity
  computed independently for each sightline from its own density profile,
  `g_gas(z) = 2*pi*G*Sigma_gas(|z'| < |z|)` (the original Step 1
  behavior). A single compact dense cloud sources its own large local
  g_gas under this mode; Phase B found this inflates mass-weighted alpha
  by x1.9-2.6 near the midplane, which is why `mean` is now the default
  (see `self_gravity_effect.txt` for the measured ratios).

Both self-gravity modes source their density from `n_model` by default
(`SELF_GRAVITY_DENSITY = "same_as_weight"` in `pipeline/compute_all.py`'s
control block -- the same, per-variant, HIM-substituted density used as
the pressure weight). The alternative `"observed"` setting (always the
unmodified raw density, regardless of variant) exists in the code but was
not swept in this run.

## numbers_table.csv columns

| Column | Meaning |
|---|---|
| `variant` | `RAW` (unmodified), `HIM_A` (HIM cells -> P_th=P_min, n=P_min/T_HIM), `HIM_B` (HIM cells -> P_th=P_max, n=P_max/T_HIM). No `MASKED` row exists -- no MASKED definition is present in any of the four reference scripts this project was ported from (see the Phase A report); this is a deliberate, documented omission, not an oversight. |
| `self_gravity` | `off` / `mean` / `column` (see above), or `n/a` for quantities that don't depend on self-gravity at all (`Sigma_gas`, the phase fractions). |
| `quantity` | See table below. |
| `weighting` | `vol` = volume-weighted (every in-mask voxel counted equally); `mw` = mass-weighted (voxel weighted by that variant's `n_model`). |
| `stat` | Which statistic, and over what spatial selection -- see below. |
| `value` | The number. |
| `units` | `K cm^-3` (P_th, P_tot -- this is P/k_B, matching Eq. 1 of the paper draft), `Msun/pc^2` (Sigma_gas), `km/s` (sigma_eff), `dimensionless` (alpha, Mach, phase fractions). |
| `definition` | One-line plain-language restatement of the row. |

### `quantity` values

| Quantity | Meaning |
|---|---|
| `Sigma_gas` | Full-column (+-750pc) trapezoidal integral of that variant's density, Msun/pc^2. |
| `Pth` | Thermal pressure / k_B = n*T [K cm^-3]. |
| `Ptot` | Total (hydrostatic) pressure / k_B [K cm^-3], integrated per the chosen `self_gravity` setting. |
| `alpha` | P_tot / P_th (dimensionless). |
| `phase_fraction_{CNM,UNM,WNM,HIM}` | Fraction of cells (`vol`) or fraction of gas mass (`mw`) in that phase, over the full +-500pc square x +-750pc column. HIM is included here (unlike every other quantity, where HIM cells are excluded from statistics). |
| `Mach` | Turbulent Mach number = sqrt(3*(alpha-1)), from the volume-weighted mean alpha at the given location. |
| `sigma_eff` | Effective non-thermal velocity dispersion [km/s], from the same volume-weighted mean alpha and the mean temperature of neutral (non-HIM) cells at that location. |

### `stat` values

| `stat` | Meaning |
|---|---|
| `median` / `mean` | Plain statistic (used only for `Sigma_gas`, which has no z-dependence). "mean" is always the plain arithmetic (or weighted-arithmetic) mean -- never a mean of log10. |
| `z0_median`, `z0_mean`, `z0_p16`, `z0_p84` | Statistic within the z=0 +-25pc slab (50pc thick, centered on the midplane). |
| `z150_*`, `z300_*` | Same, centered on z=150pc and z=300pc. |
| `absz500_median`, `absz500_mean`, `absz500_p16`, `absz500_p84` | Bin-average (mean over the 50 10pc-wide |z| bins spanning 0-500pc) of that statistic's vertical profile -- a single-number summary of the |z|<=500pc behavior, not a single flat statistic over the whole range. |
| `full_volume` | Phase fractions only: over the entire +-500pc square x +-750pc column (not slab-restricted). |
| `midplane` | Mach / sigma_eff only: evaluated at the z=0 plane. |

All statistics (except the phase fractions themselves) exclude HIM-flagged
cells -- HIM enters the P_tot integral as defined per variant, but never
enters a median/mean/percentile.

## Other files

- **`self_gravity_effect.txt`** -- for each variant and each self-gravity
  mode (`mean`, `column`), the per-cell ratio alpha_on/alpha_off reduced
  to 4 statistics (vol median, vol mean, mw median, mw mean) at the same
  three z-slabs and over the full |z|<=500pc range. This is the table
  that motivated switching the default from `column` to `mean`: `column`
  mode's mass-weighted mean ratio reaches ~2.3-2.4 near the midplane
  (compact dense clouds sourcing their own large local self-gravity),
  while `mean` mode stays a much more modest ~1.27-1.32 everywhere.
- **`him_mass_fraction.txt`** -- fraction of total gas mass (observed/RAW
  density, no HIM substitution) that sits in HIM-flagged cells, at the
  same three z-slabs and over the full +-750pc column, within the
  +-500pc square footprint. HIM's mass share grows sharply with height
  (a few percent at the midplane, up to ~46% by z=300pc, ~18% over the
  whole column) -- context for how much of the total gas mass the HIM
  treatment (RAW vs HIM_A vs HIM_B) actually touches at each height.

## Not reproduced from the old reference scripts

- 50pc-thick z-slabs (`z0`/`z150`/`z300` above) are a NEW convention, not
  what any of the four reference scripts did -- they all used a single
  nearest z-plane (2pc thick, given the grid spacing) via `nearest_idx()`.
- The nearest-rank nan-median/nan-percentile formula is applied uniformly
  to both `vol` and `mw` weightings here; `fig4_vertical_profiles/
  compute_data.py`'s own `stats_vol()` used a different (linearly
  interpolated `np.percentile`) estimator for its volume-weighted branch
  than its own `stats_mw()`. See the Phase A report for the full list of
  ported-vs-settled differences.
