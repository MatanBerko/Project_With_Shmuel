# results/

**PROVISIONAL: f98 cube, n_H = 1653 A'; final cube pending.**
**P_th = 1.1 n_H k T (physical).**

Everything here is produced by `pipeline/compute_all.py` and
`scripts/validation/{self_gravity_effect,him_mass_fraction,check_orientation}.py`,
reading only `src/physics/` and `cache/core/alpha_core.zarr`. No figures, no
paper content. To regenerate:

```
python pipeline/compute_all.py build [VARIANT ...]     # resumable, per-variant (~2 min each)
python pipeline/compute_all.py finalize [VARIANT ...]  # resumable, per-variant (~5 min each)
python pipeline/compute_all.py merge                   # or let finalize auto-merge
python scripts/validation/self_gravity_effect.py
python scripts/validation/him_mass_fraction.py
python scripts/validation/check_orientation.py
```

## Why "PROVISIONAL"

Two things about the input cube are known to be wrong or temporary, and
both are recorded in `src/conventions.py` rather than anywhere else:

1. **Orientation.** `results/orientation_check.txt` shows the current
   cube's x and y axes are transposed *and* sign-flipped relative to the
   convention the whole project assumes: the cube's x axis carries
   -y_true and its y axis carries -x_true (best mapping `(-y,-x,+z)`,
   correlation 0.94 against the Edenhofer map, versus 0.29 for the
   identity). The **z axis is correct**, independently confirmed by the
   north/south density ratio agreeing between the cube (0.82) and the
   map (0.83).

   This does **not** bias any number in `numbers_table.csv`, for two
   reasons that both have to hold and both do: the analysis footprint is
   a ±500 pc *square*, which any signed permutation of x and y maps
   exactly onto itself, and no quantity computed here depends on x or y
   individually (P_tot integrates along z, the self-gravity footprint
   average collapses x and y away, and every statistic, slab, profile and
   PDF is binned in z only). What *is* affected is anything that resolves
   the XY plane — the `Sigma_gas(x,y)` map and the z-slice images in
   `cache/core/summary.npz` are reflected about the x = -y diagonal — so
   no figure should be made from those maps until the re-oriented cube
   lands.

2. **Conversion factor.** The current cube has n_H = 1653 A' baked in
   (`N_H_PER_EXTINCTION_F98`); the orientation check's fitted ratio is
   1543, i.e. 6.6% below that, with a log-log slope of 0.986 — a
   near-constant factor, as expected for a *smoothed* reconstruction. The
   final Porter-FUV cube will use n_H = 1727 A'
   (`N_H_PER_EXTINCTION_PORTER_FUV`).

Swapping cubes is a one-line change: the cube path is the single
`zarr_filename` entry in `config/local_config.yaml`, read only through
`src/config_loader.py` and opened only by
`src.physics.loading.open_zarr()`.

## The two thermal pressures (Step 1d) — never interchange them

The dust map gives **n_H**: hydrogen *nuclei* per cm³. Helium is present
at n_He = 0.1 n_H, and it enters mass and particle count by **different**
factors:

| | per H nucleus | used for |
|---|---|---|
| **mass** | 1.4 m_H (`MU`) | `rho = 1.4 m_H n_H` — unchanged, still correct |
| **particles, neutral** | 1.1 (`PARTICLES_PER_H_NEUTRAL`) | thermal pressure of neutral gas, c_s |
| **particles, ionized** | 2.3 (`PARTICLES_PER_H_IONIZED`) | the HIM cells' pressure balance |

Thermal pressure counts *particles*, so two distinct pressures exist and
the code names them separately (`src/physics/thermal.py`):

| | definition | used for |
|---|---|---|
| **`p_nT`** | n_H · T | **Classification only.** The HIM flag (`p_nT < 0.5 P_min`) and the phase scheme (dPdn or temperature). The BS19 table — and therefore `P_min`, `P_max` and the `n_W,max`/`n_C,min` turning points — is tabulated in *this* convention, so this is the only pressure that may be compared against it. **Never reported.** |
| **`p_th_phys`** | 1.1 · n_H · T | **Everything physical.** alpha, c_s, sigma_eff, sigma_nt, Mach, and every reported "P_th" (`Pth_phys` in the numbers table). cf. Wolfire et al. 2003 Eq. 36. |

Before Step 1d, `p_nT` was used as the alpha denominator, which
**overestimated alpha by exactly 1.1** in every non-HIM cell. The mass
density was always right; only the particle count was missing.

A useful consequence: the mean mass per particle of neutral gas is
1.4/1.1 = **1.273 m_H**, the familiar ~1.27 for neutral atomic gas with
helium. It lives in the code as that *ratio*, never as a third constant,
so the three numbers cannot drift apart.

### HIM cells: pressure balance with ionized gas

HIM gas is *fully ionized* at T_HIM = 10⁶ K, so HIM_A/HIM_B's substituted
density follows from balancing the physical boundary pressure with 2.3
particles per H nucleus rather than 1.1:

```
  2.3 · n_H · T_HIM = 1.1 · P        =>   n_H = 1.1 P / (2.3 T_HIM)
  p_th_phys = 1.1 · P                     (P = P_min for HIM_A, P_max for HIM_B)
```

That is **0.478×** the pre-Step-1d `n_H = P / T_HIM`. It is a real change
to those cells' *mass*, so it propagates into rho, the self-gravity
source, Sigma_gas and the mass weighting — which is why **HIM_A/HIM_B
P_tot shifts** while **RAW's does not change at all** (RAW has no
substitution, so its density is identical under both conventions).
Which cells are HIM, and which boundary each variant uses, are unchanged.

### alpha, c_s and the two dispersions

```
  alpha      = P_tot / p_th_phys
  c_s²       = 1.1 k_B T / (1.4 m_H)   = k_B T / (1.273 m_H)
  sigma_eff² = alpha · c_s²            = P_tot / rho
  sigma_nt²  = 3 (alpha − 1) · c_s²
  Mach       = sqrt(3 (alpha − 1))     = sigma_nt / c_s
```

**Two** dispersions are reported, because the pre-Step-1d code reported
the second under the first one's name:

- **`sigma_eff`** (new definition) — the **total** effective dispersion,
  `sqrt(alpha)·c_s = sqrt(P_tot/rho)`. Every particle-count factor
  cancels out of it (alpha carries 1/f, c_s² carries f), so **sigma_eff
  is identical under both conventions** — as it should be, since it is a
  statement about P_tot and rho, neither of which this step changed.
  Note `sigma_eff/c_s = sqrt(alpha)`, *not* Mach.
- **`sigma_nt`** — the **non-thermal** (turbulent) 3D dispersion,
  `sqrt(3(alpha−1))·c_s = Mach·c_s`. This is what
  `fig4b_velocity_dispertion_Mach_number/compute_data.py` computed and
  what **Step 1c reported as `sigma_eff`**. Kept under its own accurate
  name rather than dropped. Unlike `sigma_eff` it *is* convention-
  dependent: the "− 1" breaks the cancellation.

`c_s` itself is now reported too, so the three can be checked against
each other without re-deriving anything.

### The `nT` switch

`THERMAL_PRESSURE_CONVENTION = "physical"` (default) | `"nT"`. Setting
`"nT"` makes both particle counts 1, which collapses `p_th_phys` onto
`p_nT` and the HIM substitution onto `n = P / T_HIM`, reproducing the
pre-Step-1d behaviour **element for element**. It is not physically
correct and exists only so the old numbers stay auditable.

`tests/test_thermal_pressure_convention.py` pins this down from both
ends: elementwise against the pre-Step-1d formulas written out longhand
(independent of `src/`, so it cannot drift with the code it tests), and
against the real Step 1c numbers table, committed as
`tests/reference/step1c_numbers_table.csv`. The measured relationships:
every `Pth` row is exactly 1.1× Step 1c's; RAW's `Ptot`, `Sigma_gas` and
phase fractions are bit-identical and its alpha is exactly Step 1c's ÷
1.1; HIM_A/HIM_B `Ptot` moved (median −1.8% / −5.8%, up to −20% / −36% in
the 15th-percentile tail at z = 300 pc, where HIM dominates the column).

Classification is untouched: the HIM fraction, the volume phase
fractions and both phase schemes are identical to Step 1c.

## Conventions (Shelest et al. 2026, arXiv:2607.15352)

Every one of these is a switch in `src/conventions.py`, with the previous
behaviour still reachable by changing that one constant.

### STATS_BOX — the analysis volume

A 1 kpc x 1 kpc x 800 pc box centred on the Sun:
**|x|, |y| ≤ 500 pc and |z| ≤ 400 pc** (`STATS_BOX_XY_HALF_RANGE_PC`,
`STATS_BOX_Z_HALF_RANGE_PC`). All statistics, vertical profiles and PDFs
use this box and nothing else.

Three things deliberately keep the **full z column the cube provides**
(±750 pc for the f98 cube), and must not be "tidied" into the box:

- **The P_tot hydrostatic integral.** P = 0 stays pinned at the cube's top
  and bottom edge. Clipping the integration to |z| ≤ 400 pc would put the
  zero-pressure boundary inside the gas layer and systematically
  understate P_tot everywhere in the box (by more than 50% at the
  midplane — `tests/test_stats_box.py` pins this down).
- **The self-gravity footprint average.** Taken over |x|, |y| ≤ 500 pc at
  every z the cube provides, because g_gas(z) inside the box depends on
  the gas column outside it.
- **Sigma_gas**, which stays a full-column integral over the ±500 pc
  square.

### PHASE_SCHEME — how the neutral phases are classified

- **`dPdn` (NEW DEFAULT)** — by **density**, against the two dP/dn = 0
  turning points of the BS19 thermal-equilibrium S-curve at that cell's
  I_UV:

  | | |
  |---|---|
  | warm (WNM) | n < n_W,max(I_UV) |
  | unstable (UNM) | n_W,max(I_UV) ≤ n ≤ n_C,min(I_UV) |
  | cold (CNM) | n > n_C,min(I_UV) |

  The boundaries are I_UV-dependent, so unlike the temperature cuts they
  move from cell to cell.

  **Method** (`src/physics/thermal.py::build_phase_density_bounds`, which
  also reports it as a string in the cache attrs): the turning points are
  read off the `.mat` file's *other* tabulation of the same curve,
  `T_2d_n(I_UV, n)` on the log-uniform `n_` grid, not off the
  `T_2d_P(I_UV, P)` grid used for P_min/P_max. The equilibrium curve is
  single-valued in n but triple-valued in P across the two-phase range,
  and `T_2d_P` — being tabulated on a P grid — jumps between branches
  unpredictably there (at I_UV = 1 it flips from T ≈ 8400 K to T ≈ 266 K
  between two adjacent P samples, and dn/dP changes sign 18 times inside
  the two-phase range). P_min/P_max survive that because they only need
  the *first* and *last* such sign change; the *density* at a turning
  point cannot be read reliably off a branch-jumping table.
  P(n) = n·T_2d_n has exactly two clean dP/dn sign changes. For each
  I_UV, dP/d(log n) is formed on the log-uniform n grid, the first sign
  change gives n_W,max (warm-branch turning point, the local P maximum)
  and the last gives n_C,min (cold-branch turning point, the local P
  minimum), each refined by linear root-finding in log n; both are then
  interpolated linearly in log10 I_UV vs log10 n with constant edge fill.
  For I_UV ≳ 250 the S-curve flattens out entirely (no thermal
  instability) — those rows are dropped as NaN and the edge fill applies,
  exactly as `build_pmin_pmax()` already treats the same rows. In this
  run **0.0000%** of neutral cells fell back to UNM for want of a finite
  boundary.

  **Consistency:** the turning-point pressures of the n-grid curve
  reproduce the P-grid P_min/P_max to a median 2–3% (at I_UV = 1: 2445 vs
  2570 and 7191 vs 6880 K cm⁻³) — the same two physical points located by
  two different estimators on two tabulations of one curve.

- **`temperature`** — by temperature against two fixed cuts (CNM
  T ≤ 300 K, UNM 300–6000 K, WNM T > 6000 K). The behaviour of all four
  reference scripts, kept available behind the switch.

The **HIM flag is separate from and unchanged by this switch**: always
P_th < 0.5·P_min(I_UV), computed once, and it takes precedence over the
neutral classification under both schemes. The dPdn classification is
applied to the *observed* density, not to a variant's HIM-substituted
density — since HIM cells are relabelled HIM anyway and are the only ones
a variant's density substitution touches, both choices give an identical
phase flag, and using the observed density makes that independence
explicit. Hence one phase flag per scheme serves all three variants, and
the **volume** phase fractions are identical across variants by
construction (only the **mass** fractions differ, through the weight).

### Slabs, profiles, percentiles

- **Slabs** — single-height statistics use **|z − z_c| ≤ 30 pc** (60 pc
  thick) at z_c = 0, 150, 300 pc. Was ±25 pc (50 pc thick). Neither is
  what the four reference scripts did — they all used a single nearest
  2 pc z-plane via `nearest_idx()`.
- **Vertical profiles** — one point **per grid plane** inside the box, no
  z binning at all (401 signed-z points at 2 pc spacing; was 10 pc bins
  over |z| ≤ 500 pc). Both a signed-z profile (shows north/south
  asymmetry) and a folded |z| profile (higher S/N) are stored.
- **Percentiles** — **15th/85th** (`PERCENTILE_SCHEME_15_85`, the
  default). `16_84` stays available behind the switch. 15/85 is also what
  this project's own `fig4_vertical_profiles/compute_data.py` used before
  Phase B moved to 16/84, so the default now agrees with both Shelest+26
  and the original reference script. Every `WeightedStats` carries the
  levels it was computed at, so a stored number cannot be mistaken for
  the other convention.

## The self-gravity modes

Every pressure/alpha number is computed under one of three gas
self-gravity settings (`self_gravity` column):

- **`off`** — hydrostatic balance against Guo+20's external field only
  (stellar disk + dark matter halo). No gas self-gravity.
- **`mean`** (`footprint_mean`, **DEFAULT**) — gas self-gravity from a
  single, horizontally averaged density profile: at each z, the density
  used as the pressure weight (RAW/HIM_A/HIM_B's `n_model`) is averaged
  over the entire ±500 pc square footprint (NaN cells excluded from the
  mean, not zeroed), then that one 1D profile's enclosed column,
  `g_gas(z) = 2πG·Sigma_bar(|z'| < |z|)`, is applied identically to every
  sightline. This matches the physical picture behind Guo+20's external
  field: a smooth, horizontally uniform disk.
- **`column`** (`per_column`, sensitivity option) — gas self-gravity
  computed independently for each sightline from its own density profile.
  A single compact dense cloud sources its own large local g_gas under
  this mode; Phase B found this inflates mass-weighted alpha by x1.9–2.6
  near the midplane, which is why `mean` is the default.

  **`column` was not swept in this run.** Its code path, its
  `src/conventions.py` entry, its tests and its already-cached zarr
  groups are all untouched — adding `"column"` back to
  `SELF_GRAVITY_SETTINGS` in `pipeline/compute_all.py` (and to `ON_MODES`
  in `self_gravity_effect.py`) is the only change needed to sweep it
  again. It was dropped because Phase B already answered the question it
  existed to answer and that answer is not changing, so recomputing it
  would have tripled the finalize cost for nothing.

Both self-gravity modes source their density from `n_model` by default
(`SELF_GRAVITY_DENSITY = "same_as_weight"` in `pipeline/compute_all.py`'s
control block). The alternative `"observed"` setting exists in the code
but was not swept.

## numbers_table.csv columns

The file starts with two `#`-prefixed header lines — the `PROVISIONAL:`
cube caveat and the `P_th = 1.1 n_H k T (physical)` convention — then the
header row. **`phase_scheme` is new in Step 1c.**

| Column | Meaning |
|---|---|
| `variant` | `RAW` (unmodified), `HIM_A` (HIM cells -> P_th=P_min, n=P_min/T_HIM), `HIM_B` (HIM cells -> P_th=P_max, n=P_max/T_HIM). No `MASKED` row exists -- no MASKED definition is present in any of the four reference scripts this project was ported from (see the Phase A report); this is a deliberate, documented omission, not an oversight. |
| `self_gravity` | `off` / `mean` (see above), or `n/a` for quantities that don't depend on self-gravity at all (`Sigma_gas`, the phase fractions). |
| `phase_scheme` | `dPdn` (headline) / `temperature`, for the phase-fraction rows only; `n/a` for every quantity that does not depend on the scheme. |
| `quantity` | See table below. |
| `weighting` | `vol` = volume-weighted (every in-mask voxel counted equally); `mw` = mass-weighted (voxel weighted by that variant's `n_model`). |
| `stat` | Which statistic, and over what spatial selection -- see below. |
| `value` | The number. |
| `units` | `K cm^-3` (P_th, P_tot -- this is P/k_B, matching Eq. 1 of the paper draft), `Msun/pc^2` (Sigma_gas), `km/s` (sigma_eff), `dimensionless` (alpha, Mach, phase fractions). |
| `definition` | One-line plain-language restatement of the row. |

### `quantity` values

| Quantity | Meaning |
|---|---|
| `Sigma_gas` | **Full-column** (±750 pc) trapezoidal integral of that variant's density over the ±500 pc square, Msun/pc^2 — deliberately not box-clipped. |
| `Pth_phys` | **Physical** thermal pressure / k_B = **1.1·n_H·T** [K cm^-3] (renamed from `Pth`, which held n_H·T — the rename is deliberate, so the change shows up in the data and not only in this prose). Identical across variants, because statistics exclude HIM cells, the HIM substitution only changes HIM cells, and the mass weights outside them are unchanged — a useful built-in consistency check. |
| `Ptot` | Total (hydrostatic) pressure / k_B [K cm^-3], integrated over the full column per the chosen `self_gravity` setting. Unchanged in definition by Step 1d; its *value* moves for HIM_A/HIM_B because their HIM cells' mass changed. |
| `alpha` | P_tot / `Pth_phys` (dimensionless). |
| `phase_fraction_{CNM,UNM,WNM,HIM}` | Fraction of cells (`vol`) or of gas mass (`mw`) in that phase, under the `phase_scheme` of that row. HIM is included here (unlike every other quantity, where HIM cells are excluded from statistics). The `vol` fractions are untouched by Step 1d; the `mw` ones move for HIM_A/HIM_B, by exactly the 0.478 density factor in the HIM bin. |
| `Mach` | Turbulent Mach number = sqrt(3*(alpha-1)) = `sigma_nt`/`c_s`, from the volume-weighted mean alpha at the given location. |
| `sigma_eff` | **Total** effective velocity dispersion [km/s] = sqrt(alpha)·c_s = sqrt(P_tot/rho). Convention-independent. **Not** the same quantity Step 1c reported under this name — see `sigma_nt`. |
| `sigma_nt` | **Non-thermal** (turbulent) 3D velocity dispersion [km/s] = sqrt(3*(alpha-1))·c_s = Mach·c_s. **This is Step 1c's `sigma_eff`.** |
| `c_s` | Isothermal sound speed [km/s] = sqrt(1.1·k_B·T/(1.4·m_H)) of neutral gas, from the mean temperature of neutral (non-HIM) cells at that location. |

### `stat` values

| `stat` | Meaning |
|---|---|
| `median` / `mean` | Plain statistic (used only for `Sigma_gas`, which has no z-dependence). "mean" is always the plain arithmetic (or weighted-arithmetic) mean -- never a mean of log10. |
| `z0_median`, `z0_mean`, `z0_p15`, `z0_p85` | Statistic within the z = 0 ± 30 pc slab (60 pc thick, centred on the midplane). |
| `midplane` (Mach / dispersions / `c_s`) | Evaluated at the single z = 0 grid plane, from the volume-weighted mean alpha and the mean T of neutral cells there. |
| `z150_*`, `z300_*` | Same, centred on z = 150 pc and z = 300 pc. |
| `box_median`, `box_mean`, `box_p15`, `box_p85` | A **direct** statistic over every neutral cell in the STATS_BOX — one number, one population, no double reduction. |
| `z0` / `z150` / `z300` / `box` (phase fractions) | The same four spatial selections, for the phase-fraction rows. |
| `midplane` | Mach / sigma_eff only: evaluated at the z = 0 plane. |

**Retired:** `absz500_*`. Those rows were a bin-average over the 50
10 pc-wide |z| bins of the old vertical profile — an average of an
average, and over |z| ≤ 500 pc, which is outside the STATS_BOX. They are
replaced by `box_*`, a single direct statistic over the box.

All statistics (except the phase fractions themselves) exclude
HIM-flagged cells — HIM enters the P_tot integral as defined per variant,
but never enters a median/mean/percentile.

## Other files

- **`numbers_table.txt`** — the same rows as the `.csv`, human-formatted,
  with the conventions restated in the header block.
- **`orientation_check.txt`** — the 48-axis-mapping scan, the fitted
  n_H / A' ratio and the north/south comparison described under "Why
  PROVISIONAL" above. Report only: `scripts/validation/check_orientation.py`
  does not change the loader or any other output.
- **`self_gravity_effect.txt`** — for each variant, the per-cell ratio
  alpha_mean/alpha_off reduced to 4 statistics (vol median, vol mean, mw
  median, mw mean) at the three 60 pc slabs and over the whole
  STATS_BOX. The ratio itself is unaffected by the STATS_BOX change —
  alpha on both sides comes from the unchanged full-column P_tot integral
  and the box only decides which cells enter the four statistics. Step 1d
  likewise leaves **RAW's** ratios bit-identical (the 1.1 factor is on
  both sides of alpha_on/alpha_off and cancels); HIM_A/HIM_B's move by
  ~0.5–1% because their HIM density change alters the self-gravity
  source.
- **`him_mass_fraction.txt`** — fraction of total gas mass
  (observed/RAW density, no HIM substitution) in HIM-flagged cells, at
  the three slabs, over the STATS_BOX, and over the full ±750 pc column.
  Unchanged by Step 1d, since it reports the *observed* density; note
  that the HIM_A/HIM_B *models* now place only 0.478× as much mass in
  those cells as these fractions imply.
  The full-column number is kept alongside the box number because that is
  the column the P_tot integral actually uses. HIM's mass share grows
  sharply with height (3% at the midplane, 46% by z = 300 pc, 10% over
  the box, 18% over the whole column) — context for how much of the total
  gas mass the HIM treatment (RAW vs HIM_A vs HIM_B) actually touches at
  each height.

## Not reproduced from the old reference scripts

- The 60 pc-thick z-slabs and the per-grid-plane vertical profiles are
  Shelest+26 conventions, not what any of the four reference scripts did
  — they all used a single nearest z-plane (2 pc thick) via
  `nearest_idx()`.
- The nearest-rank nan-median/nan-percentile formula is applied uniformly
  to both `vol` and `mw` weightings here;
  `fig4_vertical_profiles/compute_data.py`'s own `stats_vol()` used a
  different (linearly interpolated `np.percentile`) estimator for its
  volume-weighted branch than its own `stats_mw()`. See the Phase A
  report for the full list of ported-vs-settled differences.
