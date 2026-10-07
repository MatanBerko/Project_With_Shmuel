"""
Step 2c (B): the HIM-flag coefficient sweep, and the TIGRESS reference file.

The flag is p_nT < C * P_min with C = 0.5. C is a choice, so the sweep in
scripts/compute/compute_him_coeff_sensitivity.py reruns the flag over
{0.1, 0.25, 0.5, 1.0}. What has to hold:

1. At C = HIM_FLAG_COEFF the swept flag IS the production flag -- so the
   sweep is a sweep of the real thing and not of a lookalike. Checked
   twice: against src.physics.him.him_flag on synthetic input, and
   against the agreement the compute script measured over the whole box
   when it rebuilt the flag from the source cube.
2. The flagged fraction is monotonic in C. p_nT < C P_min is a nested
   family of sets, so this must hold in every z bin, not just the box.
3. The contamination at C = 0.5 is REPORTED, not asserted to be zero.
   If a future cube catches CNM or UNM cells at the fiducial coefficient
   that is a finding, and a test that forbids it would hide it. What is
   asserted is the structure: contamination is a fraction, it is zero
   below the measured onset C_crit, and nonzero at or above it.
4. The reference CSV loads and carries alpha_2p = 3.80 -- the one number
   in it that is derived rather than quoted, so the one that can rot.

Tests 1b, 2 and 3 read the cache the compute script wrote; they are
skipped if it is absent, like the other cache-backed tests here.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from src.conventions import HIM_FLAG_COEFF, HIM_FLAG_COEFF_GRID, HIM_THRESHOLD_FACTOR
from src.physics.him import him_flag

CACHE = Path("cache/core/him_coeff_sensitivity.npz")
REFERENCE_CSV = Path("results/reference/ok22_tigress_R8.csv")

C_GRID = np.asarray(HIM_FLAG_COEFF_GRID, dtype=float)


@pytest.fixture(scope="module")
def sweep():
    if not CACHE.exists():
        pytest.skip(f"{CACHE} not built; run scripts/compute/"
                    "compute_him_coeff_sensitivity.py")
    return np.load(CACHE, allow_pickle=True)


def swept_flag(p_nT, Pmin, C):
    """The flag rule the sweep applies, written once here so the test
    compares the rule against him_flag rather than against another copy
    of itself."""
    return (p_nT < C * Pmin) & np.isfinite(Pmin)


# --- 1. the sweep passes through the production flag ----------------------
def test_coefficient_is_an_alias_not_a_second_constant():
    assert HIM_FLAG_COEFF == HIM_THRESHOLD_FACTOR
    assert HIM_FLAG_COEFF in HIM_FLAG_COEFF_GRID


def test_swept_flag_at_fiducial_is_the_production_flag():
    rng = np.random.default_rng(2026)
    Pmin = 10.0 ** rng.uniform(2.0, 4.5, size=20000)
    # Straddle the threshold deliberately: a broad draw would put almost
    # nothing within a percent of it and the test would pass vacuously.
    scale = rng.uniform(0.9, 1.1, size=Pmin.size) * HIM_FLAG_COEFF
    p_nT = scale * Pmin
    Pmin[:50] = np.nan                       # unconstrained I_UV cells
    produced = him_flag(p_nT, Pmin)
    assert np.array_equal(swept_flag(p_nT, Pmin, HIM_FLAG_COEFF), produced)
    with np.errstate(invalid="ignore"):
        near = np.abs(p_nT / Pmin - HIM_FLAG_COEFF) < 0.01 * HIM_FLAG_COEFF
    assert np.count_nonzero(near) > 100      # the straddle is real
    assert np.count_nonzero(produced[np.isfinite(Pmin)]) > 0
    assert not produced[:50].any()           # NaN P_min is never flagged


def test_box_reproduction_of_the_cached_flag(sweep):
    agree = float(sweep["flag_reproduction_agreement"][0])
    assert agree == 1.0, (
        f"the sweep reproduced the cached flag on only {agree:.8%} of the box")
    assert float(sweep["C_fiducial"][0]) == HIM_FLAG_COEFF
    assert np.array_equal(sweep["C_grid"], C_GRID)


# --- 2. nested sets, so a monotonic flagged fraction ----------------------
def test_flagged_fraction_is_monotonic_in_C_synthetic():
    rng = np.random.default_rng(7)
    Pmin = 10.0 ** rng.uniform(2.0, 4.5, size=50000)
    p_nT = Pmin * 10.0 ** rng.uniform(-1.5, 1.0, size=Pmin.size)
    flags = [swept_flag(p_nT, Pmin, C) for C in C_GRID]
    for lo, hi in zip(flags, flags[1:]):
        assert np.all(hi[lo]), "a cell flagged at small C must stay flagged"
        assert hi.mean() >= lo.mean()
    assert flags[-1].mean() > flags[0].mean()   # not all four identical


def test_flagged_fraction_is_monotonic_in_C_on_the_cube(sweep):
    for key in ("box_flag_frac_vol", "box_flag_frac_mw"):
        f = sweep[key]
        assert np.all(np.diff(f) > 0), f"{key} is not increasing in C: {f}"
    for key in ("flag_frac_vol", "flag_frac_mw"):
        prof = sweep[key]                        # (n_C, n_bins)
        d = np.diff(prof, axis=0)
        assert np.all(d >= -1e-12), (
            f"{key} decreases with C in {int(np.count_nonzero(d < -1e-12))} bins")


# --- 3. contamination: reported, and structurally consistent --------------
@pytest.mark.parametrize("scheme", ["dPdn", "temperature"])
def test_contamination_is_reported_and_consistent_with_C_crit(sweep, scheme, capsys):
    vol = sweep[f"box_contam_{scheme}__vol"]
    mw = sweep[f"box_contam_{scheme}__mw"]
    C_crit = float(sweep[f"C_crit_{scheme}"][0])
    i_fid = int(np.argmin(np.abs(C_GRID - HIM_FLAG_COEFF)))

    with capsys.disabled():
        print(f"\n  [{scheme}] CNM+UNM contamination of the flagged set:")
        for C, v, m in zip(C_GRID, vol, mw):
            print(f"    C = {C:<5g} {100 * v:8.4f}% by volume  "
                  f"{100 * m:8.4f}% by mass")
        print(f"    first CNM/UNM cell caught at C = {C_crit:.4f}; "
              f"at the fiducial C = {HIM_FLAG_COEFF} contamination is "
              f"{100 * vol[i_fid]:.4f}% by volume")

    assert np.all((vol >= 0.0) & (vol <= 1.0))
    assert np.all((mw >= 0.0) & (mw <= 1.0))
    assert np.isfinite(C_crit) and C_crit > 0.0
    # The onset is the definition of C_crit, so it must agree with the
    # per-C numbers from the sweep -- this is what would break if the two
    # were computed off different phase labels.
    for C, v in zip(C_GRID, vol):
        if C < C_crit:
            assert v == 0.0, f"contamination at C = {C} < C_crit = {C_crit}"
        else:
            assert v > 0.0, f"no contamination at C = {C} >= C_crit = {C_crit}"


def test_fiducial_contamination_matches_the_measured_onset(sweep):
    # Not "contamination at 0.5 is zero" -- "contamination at 0.5 is zero
    # IF AND ONLY IF 0.5 is below the measured onset". On a different cube
    # the onset moves and this test follows it instead of failing.
    C_crit = float(sweep["C_crit_dPdn"][0])
    vol = float(sweep["box_contam_dPdn__vol"][
        int(np.argmin(np.abs(C_GRID - HIM_FLAG_COEFF)))])
    assert (vol == 0.0) == (HIM_FLAG_COEFF < C_crit)


# --- 4. the reference values the profiles are compared against ------------
def test_reference_csv_loads_and_alpha_is_the_quoted_ratio():
    assert REFERENCE_CSV.exists(), f"{REFERENCE_CSV} is missing"
    rows = {}
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(r for r in fh if not r.startswith("#"))
        assert reader.fieldnames == ["quantity", "value", "units", "source"]
        for row in reader:
            rows[row["quantity"]] = row

    for key in ("P_tot_2p", "P_th_2p", "n_H_2p", "alpha_2p", "Sigma_gas"):
        assert key in rows, f"{key} missing from {REFERENCE_CSV}"
        assert rows[key]["source"].strip(), f"{key} has no source"

    assert float(rows["alpha_2p"]["value"]) == pytest.approx(3.80, abs=5e-3)
    assert rows["alpha_2p"]["units"] == "dimensionless"
    # alpha_2p is derived, not quoted, so check it against its ingredients.
    derived = float(rows["P_tot_2p"]["value"]) / float(rows["P_th_2p"]["value"])
    assert derived == pytest.approx(float(rows["alpha_2p"]["value"]), rel=2e-3)
