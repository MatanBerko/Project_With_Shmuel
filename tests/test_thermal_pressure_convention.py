"""
Step 1d: the helium / particle-count correction to the thermal pressure.

What has to be true, and is checked here:

1. The two pressures stay apart. p_nT = n_H*T is the BS19 convention and
   feeds classification only; p_th_phys = 1.1 n_H*T feeds alpha. Nothing
   about the HIM flag or either phase scheme may move.
2. RAW, non-HIM cells: alpha_physical / alpha_nT = 1/1.1 EXACTLY.
3. sigma_eff is identical under both conventions (the particle-count
   factor cancels), and equals sqrt(P_tot/rho).
4. The HIM_A/HIM_B substitution balances the physical pressure with
   FULLY IONIZED gas: n_H = 1.1 P / (2.3 T_HIM), p_th_phys = 1.1 P.
5. THERMAL_PRESSURE_CONVENTION = "nT" reproduces the pre-Step-1d
   behaviour exactly. This is checked twice, from both ends:
     (a) elementwise, against the pre-Step-1d formulas written out
         longhand in this file -- independent of src, so it cannot drift
         with the code it is testing;
     (b) against the real Step 1c-prep numbers table, committed as
         tests/reference/step1c_numbers_table.csv, for every row whose
         exact relationship to the new numbers is known in closed form.
   (a) is what makes (b) sufficient: the pipeline only ever applies these
   functions elementwise over the cube, so elementwise identity under nT
   implies identical cubes and therefore identical statistics.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from src.conventions import (
    HIM_THRESHOLD_FACTOR,
    K_B,
    M_H,
    MU,
    PARTICLES_PER_H_BY_CONVENTION,
    PARTICLES_PER_H_IONIZED,
    PARTICLES_PER_H_NEUTRAL,
    THERMAL_PRESSURE_CONVENTION_DEFAULT,
    THERMAL_PRESSURE_CONVENTION_NT,
    THERMAL_PRESSURE_CONVENTION_PHYSICAL,
    T_HIM_K,
)
from src.physics import derived
from src.physics.him import PHASE_HIM, apply_variant, him_flag, phase_flag_dpdn, phase_flag_temperature
from src.physics.thermal import p_nT, p_th_phys_over_kb, particles_per_h

PHYS = THERMAL_PRESSURE_CONVENTION_PHYSICAL
NT = THERMAL_PRESSURE_CONVENTION_NT

REFERENCE_CSV = Path(__file__).resolve().parent / "reference" / "step1c_numbers_table.csv"

# ---------------------------------------------------------------------------
# Pre-Step-1d formulas, written out longhand. These are deliberately NOT
# imported from src -- they are the independent statement of what "nT"
# must reproduce.
# ---------------------------------------------------------------------------
KB_OVER_14_MH = K_B / (1.4 * M_H)
KMS = 1.0e5


def old_p_th(n_H, T):
    return n_H * T


def old_alpha(Ptot_kB, n_H, T):
    return Ptot_kB / (n_H * T)


def old_him_substituted_density(P, T_him=T_HIM_K):
    return P / T_him


def old_sound_speed(T):
    return np.sqrt(KB_OVER_14_MH * T) / KMS


def old_sigma_nt(alpha_val, T):
    """Pre-Step-1d sigma_nt, with the pre-Step-1e clamp of alpha-1 at zero.

    Step 1e changed that clamp to NaN (alpha < 1 means there is no
    non-thermal support to measure), so comparisons against this helper
    are restricted to alpha >= 1 -- the region where the clamp never
    applied and the two agree exactly. The NaN behaviour itself is
    covered in tests/test_alpha_estimators.py, not here: it is a Step 1e
    change and nothing to do with the thermal-pressure convention.
    """
    return np.sqrt(3.0 * np.maximum(alpha_val - 1.0, 0.0) * KB_OVER_14_MH * T) / KMS


# ---------------------------------------------------------------------------
# 1. the two pressures, and classification's immunity to the switch
# ---------------------------------------------------------------------------
def test_constants_are_the_stated_particle_counts():
    assert PARTICLES_PER_H_NEUTRAL == 1.1
    assert PARTICLES_PER_H_IONIZED == 2.3
    assert THERMAL_PRESSURE_CONVENTION_DEFAULT == PHYS
    assert particles_per_h(PHYS) == (1.1, 2.3)
    assert particles_per_h(NT) == (1.0, 1.0)
    assert set(PARTICLES_PER_H_BY_CONVENTION) == {PHYS, NT}
    with pytest.raises(ValueError, match="Unknown thermal pressure convention"):
        particles_per_h("with_helium_maybe")
    # 1.4 is a MASS factor and 1.1 a PARTICLE count; their ratio is the
    # mean mass per particle of neutral gas. Pin it so nobody "fixes" MU.
    assert MU == 1.4
    assert MU / PARTICLES_PER_H_NEUTRAL == pytest.approx(1.2727, abs=1e-4)


def test_p_nT_is_the_bs19_convention_and_takes_no_convention_argument():
    n = np.array([0.1, 1.0, 10.0])
    T = np.array([9000.0, 1000.0, 50.0])
    np.testing.assert_array_equal(p_nT(n, T), n * T)
    # p_th_phys is 1.1x that under "physical" and identical under "nT"
    np.testing.assert_allclose(p_th_phys_over_kb(n, T, PHYS), 1.1 * n * T, rtol=0, atol=0)
    np.testing.assert_array_equal(p_th_phys_over_kb(n, T, NT), old_p_th(n, T))


def test_classification_is_untouched_by_the_convention():
    """The HIM flag and both phase schemes must be fed p_nT and must not
    see the particle-count factor at all. Feeding them p_th_phys instead
    would move the HIM threshold by 10%, which this pins down."""
    Pmin = np.full(4, 2000.0)
    Pmax = np.full(4, 7000.0)
    threshold = HIM_THRESHOLD_FACTOR * Pmin[0]  # 1000 K cm^-3
    # Cell 1 is chosen to sit between threshold/1.1 and threshold: it is
    # HIM under p_nT and NOT HIM under the (incorrect) 1.1x pressure. That
    # is what makes this a real constraint rather than a tautology.
    n = np.array([0.05, 0.95, 5.0, 50.0])
    T = np.array([9000.0, 1000.0, 200.0, 40.0])
    assert threshold / 1.1 < n[1] * T[1] < threshold

    him_correct = him_flag(p_nT(n, T), Pmin)
    # the flag is a pure function of p_nT -- no convention argument exists
    np.testing.assert_array_equal(him_correct, (n * T) < threshold)
    assert him_correct[1]

    him_wrong = him_flag(p_th_phys_over_kb(n, T, PHYS), Pmin)
    assert not him_wrong[1]
    assert not np.array_equal(him_correct, him_wrong)

    # phase classification depends only on n_H and T, never on a pressure,
    # so neither scheme can see the switch; HIM still wins under both.
    nw, nc = np.full(4, 1.2), np.full(4, 8.0)
    flag_d = phase_flag_dpdn(n, nw, nc, him_correct)
    flag_t = phase_flag_temperature(T, him_correct)
    assert np.all(flag_d[him_correct] == PHASE_HIM)
    assert np.all(flag_t[him_correct] == PHASE_HIM)
    # and the non-HIM cells are classified purely on n / T
    np.testing.assert_array_equal(flag_d[~him_correct],
                                    phase_flag_dpdn(n, nw, nc, np.zeros(4, bool))[~him_correct])

    # The variant construction must not feed p_th_phys back into the flag:
    # apply_variant takes p_nT, and HIM_A's substituted cell is the one
    # him_correct marked, not the one him_wrong would have.
    v = apply_variant("HIM_A", n, p_nT(n, T), him_correct, Pmin, Pmax, PHYS)
    assert v.n_model[1] != n[1] and v.n_model[0] != n[0]
    np.testing.assert_allclose(v.n_model[2:], n[2:], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2 + 3. alpha scales by exactly 1/1.1; sigma_eff does not move at all
# ---------------------------------------------------------------------------
def test_raw_non_him_alpha_ratio_is_exactly_one_over_1_point_1():
    n = np.array([0.08, 0.4, 2.0, 12.0])
    T = np.array([8500.0, 5000.0, 400.0, 60.0])
    Ptot = np.array([3000.0, 5000.0, 9000.0, 20000.0])
    him = np.zeros(4, dtype=bool)
    Pmin, Pmax = np.full(4, 2000.0), np.full(4, 7000.0)
    pnT = p_nT(n, T)

    a_phys = derived.alpha(Ptot, apply_variant("RAW", n, pnT, him, Pmin, Pmax, PHYS).p_th_phys)
    a_nT = derived.alpha(Ptot, apply_variant("RAW", n, pnT, him, Pmin, Pmax, NT).p_th_phys)

    np.testing.assert_allclose(a_phys / a_nT, 1.0 / PARTICLES_PER_H_NEUTRAL, rtol=1e-15, atol=0)
    # and the nT branch is the pre-Step-1d number, longhand
    np.testing.assert_allclose(a_nT, old_alpha(Ptot, n, T), rtol=1e-15, atol=0)


def test_sigma_eff_is_the_same_under_both_conventions_and_equals_sqrt_ptot_over_rho():
    n = np.array([0.08, 0.4, 2.0, 12.0])
    T = np.array([8500.0, 5000.0, 400.0, 60.0])
    Ptot = np.array([3000.0, 5000.0, 9000.0, 20000.0])

    a_phys = derived.alpha(Ptot, p_th_phys_over_kb(n, T, PHYS))
    a_nT = derived.alpha(Ptot, p_th_phys_over_kb(n, T, NT))

    se_phys = derived.sigma_eff_kmps(a_phys, T, PHYS)
    se_nT = derived.sigma_eff_kmps(a_nT, T, NT)
    np.testing.assert_allclose(se_phys, se_nT, rtol=1e-14, atol=0)

    # ... because it is sqrt(P_tot / rho), which this step did not touch
    rho = MU * M_H * n
    np.testing.assert_allclose(se_phys, np.sqrt(Ptot * K_B / rho) / KMS, rtol=1e-12, atol=0)

    # sigma_nt, by contrast, IS convention-dependent: the "-1" breaks the
    # cancellation. Check it moves, and that nT gives the old number.
    sn_phys = derived.sigma_nt_kmps(a_phys, T, PHYS)
    sn_nT = derived.sigma_nt_kmps(a_nT, T, NT)
    assert np.all(sn_phys < sn_nT)
    np.testing.assert_allclose(sn_nT, old_sigma_nt(a_nT, T), rtol=1e-14, atol=0)

    # Mach is sigma_nt/c_s under both conventions (not sigma_eff/c_s)
    for a, conv in ((a_phys, PHYS), (a_nT, NT)):
        cs = derived.sound_speed_kmps(T, conv)
        np.testing.assert_allclose(derived.mach_number(a),
                                     derived.sigma_nt_kmps(a, T, conv) / cs, rtol=1e-12)
        np.testing.assert_allclose(derived.sigma_eff_kmps(a, T, conv) / cs,
                                     np.sqrt(a), rtol=1e-12)


def test_sound_speed_uses_the_particle_count_not_the_mass_factor():
    T = np.array([100.0, 8000.0])
    cs_phys = derived.sound_speed_kmps(T, PHYS)
    cs_nT = derived.sound_speed_kmps(T, NT)
    np.testing.assert_allclose(cs_nT, old_sound_speed(T), rtol=1e-15, atol=0)
    np.testing.assert_allclose(cs_phys / cs_nT, np.sqrt(PARTICLES_PER_H_NEUTRAL), rtol=1e-14)
    # i.e. c_s^2 = k_B T / (1.273 m_H) for neutral gas with helium
    np.testing.assert_allclose(
        cs_phys, np.sqrt(K_B * T / ((MU / PARTICLES_PER_H_NEUTRAL) * M_H)) / KMS, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. the HIM substitution, on a toy cube
# ---------------------------------------------------------------------------
def test_him_substitution_balances_physical_pressure_with_ionized_gas():
    # cell 0 is HIM, cells 1-2 are not
    n = np.array([1.0e-4, 0.5, 4.0])
    T = np.array([500.0, 8000.0, 100.0])
    pnT = p_nT(n, T)
    him = np.array([True, False, False])
    Pmin = np.full(3, 2000.0)
    Pmax = np.full(3, 7000.0)

    for variant, P in (("HIM_A", Pmin), ("HIM_B", Pmax)):
        v = apply_variant(variant, n, pnT, him, Pmin, Pmax, PHYS)

        # the substituted cell
        expected_n = PARTICLES_PER_H_NEUTRAL * P[0] / (PARTICLES_PER_H_IONIZED * T_HIM_K)
        assert v.n_model[0] == pytest.approx(expected_n, rel=1e-14)
        assert v.p_th_phys[0] == pytest.approx(PARTICLES_PER_H_NEUTRAL * P[0], rel=1e-14)

        # pressure balance actually holds: ionized gas at T_HIM with that
        # density has exactly the substituted physical pressure
        assert PARTICLES_PER_H_IONIZED * v.n_model[0] * T_HIM_K == pytest.approx(
            v.p_th_phys[0], rel=1e-14)

        # 0.478x the pre-Step-1d density, i.e. the mass really did change
        assert v.n_model[0] / old_him_substituted_density(P[0]) == pytest.approx(
            PARTICLES_PER_H_NEUTRAL / PARTICLES_PER_H_IONIZED, rel=1e-14)

        # non-HIM cells: density untouched, pressure simply 1.1x
        np.testing.assert_allclose(v.n_model[1:], n[1:], rtol=0, atol=0)
        np.testing.assert_allclose(v.p_th_phys[1:], PARTICLES_PER_H_NEUTRAL * pnT[1:], rtol=1e-15)

        # which boundary each variant uses is unchanged
        assert v.p_th_phys[0] == pytest.approx(
            PARTICLES_PER_H_NEUTRAL * (2000.0 if variant == "HIM_A" else 7000.0), rel=1e-14)

    # RAW is density-identical, so its mass (and therefore P_tot) is
    # untouched by Step 1d -- the reason RAW reproduces Step 1c's P_tot.
    raw = apply_variant("RAW", n, pnT, him, Pmin, Pmax, PHYS)
    np.testing.assert_allclose(raw.n_model, n, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5a. "nT" reproduces the pre-Step-1d formulas elementwise
# ---------------------------------------------------------------------------
def test_nt_convention_reproduces_pre_step1d_behaviour_elementwise():
    rng = np.random.default_rng(1104)
    n = 10.0 ** rng.uniform(-3, 1.5, 500)
    T = 10.0 ** rng.uniform(1.3, 4.0, 500)
    Ptot = 10.0 ** rng.uniform(2.5, 4.5, 500)
    Pmin = 10.0 ** rng.uniform(2.5, 3.8, 500)
    Pmax = Pmin * rng.uniform(1.5, 4.0, 500)
    pnT = p_nT(n, T)
    him = him_flag(pnT, Pmin)
    assert him.any() and not him.all(), "toy sample must contain both HIM and non-HIM cells"

    for variant in ("RAW", "HIM_A", "HIM_B"):
        v = apply_variant(variant, n, pnT, him, Pmin, Pmax, NT)

        # density: unchanged outside HIM, P/T_HIM inside it
        expected_n = n.copy()
        if variant != "RAW":
            P = Pmin if variant == "HIM_A" else Pmax
            expected_n[him] = old_him_substituted_density(P[him])
        np.testing.assert_array_equal(v.n_model, expected_n)

        # pressure: n*T outside HIM, P inside it
        expected_p = old_p_th(n, T)
        if variant != "RAW":
            P = Pmin if variant == "HIM_A" else Pmax
            expected_p[him] = P[him]
        np.testing.assert_array_equal(v.p_th_phys, expected_p)

        # alpha, sigma_nt, c_s and Mach all fall out of those
        a = derived.alpha(Ptot, v.p_th_phys)
        np.testing.assert_array_equal(a, Ptot / expected_p)
        np.testing.assert_allclose(derived.sound_speed_kmps(T, NT), old_sound_speed(T),
                                     rtol=1e-15, atol=0)
        # restricted to alpha >= 1: below that the old code clamped and
        # Step 1e returns NaN (see old_sigma_nt's docstring)
        ok = a >= 1.0
        assert ok.any() and not ok.all(), "sample must straddle alpha = 1"
        np.testing.assert_allclose(derived.sigma_nt_kmps(a[ok], T[ok], NT),
                                     old_sigma_nt(a[ok], T[ok]), rtol=1e-15, atol=0)
        assert np.all(np.isnan(derived.sigma_nt_kmps(a[~ok], T[~ok], NT)))


# ---------------------------------------------------------------------------
# 5b. "nT" reproduces the real Step 1c-prep numbers table
# ---------------------------------------------------------------------------
def _load_table(path, skip_comments=True):
    """Six-key -> value map for a numbers table.

    Step 1f added a cell_selection column, and RAW now has two rows for
    every key below (one per selection). The Step 1c fixture predates
    that, so rows are filtered to the DEFAULT selection before the map is
    built -- otherwise the two RAW rows would collide and silently leave
    whichever came last. "n/a" is kept because that is what the
    selection-independent quantities (Sigma_gas, the phase fractions)
    carry.
    """
    with open(path, newline="") as f:
        lines = [ln for ln in f if not (skip_comments and ln.startswith("#"))]
    out = {}
    for d in csv.DictReader(lines):
        if d.get("cell_selection", "exclude_him_flag") not in ("exclude_him_flag", "n/a"):
            continue
        key = (d["variant"], d["self_gravity"], d["phase_scheme"],
               d["quantity"], d["weighting"], d["stat"])
        assert key not in out, f"duplicate key after selection filter: {key}"
        out[key] = float(d["value"])
    return out


@pytest.fixture(scope="module")
def step1c():
    assert REFERENCE_CSV.exists(), f"missing reference fixture {REFERENCE_CSV}"
    return _load_table(REFERENCE_CSV)


@pytest.fixture(scope="module")
def current():
    path = Path(__file__).resolve().parent.parent / "results" / "numbers_table.csv"
    if not path.exists():
        pytest.skip("results/numbers_table.csv not present -- run pipeline/compute_all.py")
    return _load_table(path)


# float32 round-trips through the cache, so the exactness tolerance is
# float32 epsilon, not float64.
F32_RTOL = 2e-7

# Step 1e renamed the reported alpha rows: with three estimators in play,
# a bare "alpha" quantity with stat "z0_median" no longer identifies a
# number. The Step 1c fixture predates that, so its keys are translated
# here. Nothing about the VALUES being compared changed -- this is a
# key mapping, and keeping it explicit is what documents the rename.
STEP1C_ALPHA_STAT_TO_QUANTITY = {
    "median": "alpha_median_of_ratios",
    "mean": "alpha_mean_of_ratios",
    "p15": "alpha_p15_of_ratios",
    "p85": "alpha_p85_of_ratios",
}
# Step 1e also split Mach/sigma_nt per estimator. Step 1c/1d's midplane
# Mach and sigma_nt were both built from the vol-weighted MEAN of the
# ratios, so they map onto the _from_mean members.
STEP1C_MIDPLANE_RENAME = {
    "Mach": "Mach_from_mean",
    "sigma_nt": "sigma_nt_from_mean",
    "sigma_eff": "sigma_eff",
    "c_s": "c_s",
}


def _translate_step1c_key(key):
    """Step 1c key -> the current key for the same number, or None if the
    row has no single current counterpart."""
    variant, sg, scheme, qty, wt, stat = key
    if qty == "Pth":
        return (variant, sg, scheme, "Pth_phys", wt, stat)
    if qty == "alpha":
        where, _, estimator = stat.rpartition("_")
        new_qty = STEP1C_ALPHA_STAT_TO_QUANTITY.get(estimator)
        if new_qty is None:
            return None
        return (variant, sg, scheme, new_qty, wt, where)
    if stat == "midplane" and qty in STEP1C_MIDPLANE_RENAME:
        return (variant, sg, scheme, STEP1C_MIDPLANE_RENAME[qty], wt, stat)
    return key


def test_step1c_pth_rows_are_exactly_1_point_1_times_the_new_ones(step1c, current):
    """p_th_phys = 1.1 * p_nT in EVERY cell, HIM-substituted ones included,
    and the Pth statistics exclude HIM cells whose weights are the only
    ones Step 1d changed. So every Pth row must scale by exactly 1.1 --
    for all three variants, both weightings, every slab."""
    checked = 0
    for key, old_val in step1c.items():
        variant, sg, scheme, qty, wt, stat = key
        if qty != "Pth":
            continue
        new_val = current.get(_translate_step1c_key(key))
        assert new_val is not None, f"no Pth_phys counterpart for {key}"
        assert new_val == pytest.approx(PARTICLES_PER_H_NEUTRAL * old_val, rel=F32_RTOL), key
        checked += 1
    assert checked >= 48, f"only {checked} Pth rows compared"


def test_raw_reproduces_step1c_exactly_under_the_known_conversion(step1c, current):
    """RAW's density is identical under both conventions, so:
         P_tot, Sigma_gas, Mach(*) and the phase fractions are UNCHANGED,
         alpha is Step 1c's divided by exactly 1.1.
    (*) Mach = sqrt(3(alpha-1)) is built on alpha, so it is NOT unchanged
    -- it is checked against Step 1c's alpha explicitly below rather than
    assumed either way.
    """
    unchanged = 0
    scaled = 0
    for key, old_val in step1c.items():
        variant, sg, scheme, qty, wt, stat = key
        if variant != "RAW":
            continue
        if qty in ("Ptot", "Sigma_gas") or qty.startswith("phase_fraction_"):
            assert current[key] == pytest.approx(old_val, rel=F32_RTOL), key
            unchanged += 1
        elif qty == "alpha":
            new_key = _translate_step1c_key(key)
            assert new_key is not None, key
            assert current[new_key] == pytest.approx(old_val / PARTICLES_PER_H_NEUTRAL,
                                                      rel=F32_RTOL), key
            scaled += 1
    assert unchanged >= 40 and scaled >= 30, (unchanged, scaled)

    # Mach: rebuilt from Step 1c's alpha, pushed through the 1/1.1 scaling
    for sg in ("off", "mean"):
        old_key = ("RAW", sg, "n/a", "Mach", "vol", "midplane")
        new_key = _translate_step1c_key(old_key)
        # Step 1c's Mach came from its own vol-weighted mean of the ratios
        # at the midplane; recover that alpha, scale it, and re-derive.
        old_mach = step1c[old_key]
        old_alpha_mean = 1.0 + old_mach ** 2 / 3.0
        expected = float(derived.mach_number(
            np.array([old_alpha_mean / PARTICLES_PER_H_NEUTRAL]))[0])
        assert current[new_key] == pytest.approx(expected, rel=1e-5), old_key


def test_step1c_sigma_eff_row_is_reproduced_by_the_new_sigma_nt_row(step1c, current):
    """Step 1c's "sigma_eff" was the NON-THERMAL dispersion
    sqrt(3(alpha-1))*c_s, which is now reported as sigma_nt. Under nT it
    would be bit-identical; under the physical convention it drops,
    because both alpha and c_s move. Check the new sigma_nt equals what
    the Step 1c numbers imply after the conversion -- i.e. that the
    renamed row really is the same quantity, computed the same way.
    """
    for variant in ("RAW", "HIM_A", "HIM_B"):
        for sg in ("off", "mean"):
            old_sigma = step1c[(variant, sg, "n/a", "sigma_eff", "vol", "midplane")]
            old_mach = step1c[(variant, sg, "n/a", "Mach", "vol", "midplane")]
            # Step 1c: sigma = Mach * c_s_nT  =>  c_s_nT = sigma / Mach
            c_s_nT = old_sigma / old_mach
            new_mach = current[(variant, sg, "n/a", "Mach_from_mean", "vol", "midplane")]
            new_c_s = current[(variant, sg, "n/a", "c_s", "vol", "midplane")]
            new_sigma_nt = current[(variant, sg, "n/a", "sigma_nt_from_mean", "vol", "midplane")]

            # c_s only gained the sqrt(1.1) particle-count factor; the mean
            # neutral T it is built from did not change for RAW.
            if variant == "RAW":
                assert new_c_s == pytest.approx(
                    c_s_nT * np.sqrt(PARTICLES_PER_H_NEUTRAL), rel=1e-4)
            # and sigma_nt is still exactly Mach * c_s
            assert new_sigma_nt == pytest.approx(new_mach * new_c_s, rel=1e-5)


def test_him_variants_ptot_moved_and_only_through_the_density_change(step1c, current):
    """HIM_A/HIM_B P_tot MUST move (their HIM cells now hold 0.478x the
    mass, which is part of the hydrostatic weight and of the self-gravity
    source), and RAW's must not. A pipeline that accidentally left the old
    densities in place would pass every other test in this file, so this
    is the one that catches it.
    """
    raw_moved = []
    him_moved = []
    for key, old_val in step1c.items():
        variant, sg, scheme, qty, wt, stat = key
        if qty != "Ptot":
            continue
        ratio = current[key] / old_val
        (raw_moved if variant == "RAW" else him_moved).append(ratio)

    assert raw_moved and him_moved
    np.testing.assert_allclose(raw_moved, 1.0, rtol=F32_RTOL)
    # the shift is real but small -- HIM cells carry a few percent of the
    # mass near the midplane, rising with height
    assert max(abs(np.array(him_moved) - 1.0)) > 1e-4, "HIM_A/HIM_B P_tot did not move at all"
    assert max(abs(np.array(him_moved) - 1.0)) < 0.5, "HIM_A/HIM_B P_tot moved implausibly far"
