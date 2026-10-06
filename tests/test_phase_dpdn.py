"""
Shelest+26 dP/dn phase classification.

Two layers:

1. The turning-point LOCATOR
   (src.physics.thermal.locate_scurve_turning_points) is driven with a
   synthetic, analytically-constructed S-curve whose two dP/dn = 0 points
   are known in closed form. This checks the locator itself, independent
   of any real table -- if it silently picked, say, an inflection point
   or the wrong end of the unstable branch, the recovered densities would
   not match the analytic ones.

2. The CLASSIFIER (src.physics.him.phase_flag_dpdn) is checked on cells
   placed deliberately either side of known boundaries, including the
   inclusive UNM edges, the HIM precedence rule, and the non-finite-
   boundary fallback.

Plus two checks against the REAL BS19 table: the two turning points are
correctly ordered (n_W,max < n_C,min) at every I_UV that has an S-curve,
and their pressures agree with the independently-computed P_max/P_min
from the P-grid tabulation -- the cross-tabulation consistency argument
that justifies using T_2d_n for the densities while P_min/P_max (and
therefore the HIM flag) keep coming from T_2d_P.
"""

import numpy as np
import pytest

from src.config_loader import load_resolved_config
from src.conventions import PHASE_SCHEME_DPDN, PHASE_SCHEME_TEMPERATURE
from src.physics.him import (
    PHASE_CNM,
    PHASE_HIM,
    PHASE_UNM,
    PHASE_WNM,
    classify_phases,
    dpdn_unresolved_fraction,
    phase_flag_dpdn,
)
from src.physics.thermal import (
    build_phase_density_bounds,
    build_pmin_pmax,
    locate_scurve_turning_points,
    turning_point_densities,
)

# ---------------------------------------------------------------------------
# A toy S-curve with turning points we can write down exactly.
#
# Work in x = log10(n) and build P(x) as a cubic in x with a local maximum
# at x = X_W and a local minimum at x = X_C:
#
#     dP/dx = K * (x - X_W) * (x - X_C)       (K > 0)
#
# Integrating gives P(x) = P0 + K*(x^3/3 - (X_W+X_C)*x^2/2 + X_W*X_C*x),
# which rises, falls between X_W and X_C, then rises again -- the shape of
# a thermally unstable equilibrium curve. dP/dx and dP/dn vanish at the
# same two points (dP/dn = dP/dx / (n ln 10), and n > 0), so the
# turning-point densities are exactly 10**X_W and 10**X_C.
#
# The locator is handed T(n) = P(n)/n, which is what a BS19-style table
# stores, so it has to reconstruct P itself.
# ---------------------------------------------------------------------------
X_W = np.log10(1.2)    # warm-branch turning point: n_W,max = 1.2 cm^-3
X_C = np.log10(8.0)    # cold-branch turning point: n_C,min = 8.0 cm^-3
K_CUBIC = 3.0e4
P_OFFSET = 4.0e4       # keeps P > 0 across the toy grid (TOY_LOG_N_LO..HI)
TOY_LOG_N_LO, TOY_LOG_N_HI = -1.0, 2.0


def _toy_pressure(log_n):
    return P_OFFSET + K_CUBIC * (
        log_n ** 3 / 3.0 - (X_W + X_C) * log_n ** 2 / 2.0 + X_W * X_C * log_n
    )


def _toy_table(n_points=601, log_n_lo=TOY_LOG_N_LO, log_n_hi=TOY_LOG_N_HI):
    """(n_grid, T_of_n) for the toy S-curve, on a log-uniform n grid like
    the real BS19 n_ grid."""
    log_n = np.linspace(log_n_lo, log_n_hi, n_points)
    n_grid = 10.0 ** log_n
    P = _toy_pressure(log_n)
    assert np.all(P > 0), "toy curve must stay at positive pressure"
    return n_grid, P / n_grid


def test_locator_recovers_known_toy_turning_points():
    n_grid, T_of_n = _toy_table()
    res = locate_scurve_turning_points(n_grid, T_of_n)
    assert res is not None
    n_w, n_c, P_w, P_c = res

    # densities: exact, to within the grid's own refinement accuracy
    assert np.isclose(n_w, 10.0 ** X_W, rtol=2e-3), (n_w, 10.0 ** X_W)
    assert np.isclose(n_c, 10.0 ** X_C, rtol=2e-3), (n_c, 10.0 ** X_C)
    assert n_w < n_c

    # the warm turning point is the local pressure MAXIMUM and the cold one
    # the local pressure MINIMUM -- i.e. they are the right way round, and
    # the unstable branch really does lie between them with dP/dn < 0.
    assert np.isclose(P_w, _toy_pressure(X_W), rtol=2e-3)
    assert np.isclose(P_c, _toy_pressure(X_C), rtol=2e-3)
    assert P_w > P_c
    mid = 0.5 * (X_W + X_C)
    assert _toy_pressure(mid) < P_w and _toy_pressure(mid) > P_c


def test_locator_returns_none_without_an_s_curve():
    """A monotonically rising P(n) (no thermal instability) has no turning
    points, and must be reported as such rather than fabricating two."""
    log_n = np.linspace(-2.0, 3.0, 401)
    n_grid = 10.0 ** log_n
    P = 1.0e3 * n_grid  # isothermal: P strictly increasing, dP/dn > 0 everywhere
    assert locate_scurve_turning_points(n_grid, P / n_grid) is None


def test_classifier_sorts_cells_against_toy_boundaries():
    n_w, n_c = 1.2, 8.0
    # below n_w -> warm; above n_c -> cold; between (inclusive) -> unstable
    n = np.array([0.01, 1.1999, n_w, 3.0, n_c, 8.0001, 100.0])
    nw = np.full_like(n, n_w)
    nc = np.full_like(n, n_c)
    him = np.zeros(n.shape, dtype=bool)

    flag = phase_flag_dpdn(n, nw, nc, him)
    np.testing.assert_array_equal(
        flag,
        [PHASE_WNM, PHASE_WNM, PHASE_UNM, PHASE_UNM, PHASE_UNM, PHASE_CNM, PHASE_CNM],
    )
    # the three classes partition every cell: no cell is left unclassified
    assert set(np.unique(flag)).issubset({PHASE_CNM, PHASE_UNM, PHASE_WNM})


def test_classifier_him_takes_precedence_over_every_density_class():
    n = np.array([0.01, 3.0, 100.0])  # would be WNM, UNM, CNM
    nw = np.full_like(n, 1.2)
    nc = np.full_like(n, 8.0)
    flag = phase_flag_dpdn(n, nw, nc, np.array([True, True, True]))
    np.testing.assert_array_equal(flag, [PHASE_HIM, PHASE_HIM, PHASE_HIM])


def test_classifier_defaults_unresolvable_cells_to_unm_and_reports_them():
    n = np.array([0.01, 3.0, 100.0, np.nan])
    nw = np.array([1.2, np.nan, 1.2, 1.2])
    nc = np.array([8.0, 8.0, np.nan, 8.0])
    him = np.array([False, False, False, False])

    flag = phase_flag_dpdn(n, nw, nc, him)
    # cell 0 is fully resolvable (warm); 1, 2, 3 each lack something
    np.testing.assert_array_equal(flag, [PHASE_WNM, PHASE_UNM, PHASE_UNM, PHASE_UNM])
    assert dpdn_unresolved_fraction(n, nw, nc, him) == pytest.approx(3 / 4)

    # HIM cells are not counted as "unresolved" -- they never needed a boundary
    him2 = np.array([False, True, True, True])
    assert dpdn_unresolved_fraction(n, nw, nc, him2) == pytest.approx(0.0)


def test_classify_phases_dispatches_and_rejects_bad_input():
    n = np.array([0.01, 3.0, 100.0])
    T = np.array([9000.0, 1000.0, 50.0])
    nw = np.full_like(n, 1.2)
    nc = np.full_like(n, 8.0)
    him = np.zeros(n.shape, dtype=bool)

    by_density = classify_phases(him, PHASE_SCHEME_DPDN, n_cm3=n, n_w_max=nw, n_c_min=nc)
    by_temperature = classify_phases(him, PHASE_SCHEME_TEMPERATURE, T_K=T)
    np.testing.assert_array_equal(by_density, [PHASE_WNM, PHASE_UNM, PHASE_CNM])
    np.testing.assert_array_equal(by_temperature, [PHASE_WNM, PHASE_UNM, PHASE_CNM])

    with pytest.raises(ValueError, match="requires T_K"):
        classify_phases(him, PHASE_SCHEME_TEMPERATURE)
    with pytest.raises(ValueError, match="requires n_cm3"):
        classify_phases(him, PHASE_SCHEME_DPDN, T_K=T)
    with pytest.raises(ValueError, match="Unknown phase scheme"):
        classify_phases(him, "by_vibes", T_K=T)


# ---------------------------------------------------------------------------
# Against the real BS19 table
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bs19_paths():
    return load_resolved_config()["bs19_mat_path"]


def test_real_table_turning_points_are_ordered_and_match_pmin_pmax(bs19_paths):
    IUV, n_w, n_c, P_w, P_c = turning_point_densities(bs19_paths)
    pm = build_pmin_pmax(bs19_paths)

    ok = np.isfinite(n_w) & np.isfinite(n_c)
    assert ok.sum() >= 20, f"only {ok.sum()} I_UV rows have an S-curve"
    assert np.all(n_w[ok] < n_c[ok]), "n_W,max must stay below n_C,min at every I_UV"

    # Same two physical points as P_min/P_max, located on the other
    # tabulation of the same curve -- so the pressures must agree to within
    # the two estimators' difference, not exactly.
    ratio_max = P_w[ok] / pm.p_max(IUV[ok])
    ratio_min = P_c[ok] / pm.p_min(IUV[ok])
    assert np.median(ratio_max) == pytest.approx(1.0, abs=0.10), np.median(ratio_max)
    assert np.median(ratio_min) == pytest.approx(1.0, abs=0.10), np.median(ratio_min)


def test_real_bounds_interpolators_are_monotonic_in_iuv_and_bracket_solar(bs19_paths):
    b = build_phase_density_bounds(bs19_paths)
    assert "dP/dn" in b.method  # the method string is reported, not implied

    iuv = np.logspace(-3, 2, 60)
    n_w = b.n_w_max(iuv)
    n_c = b.n_c_min(iuv)
    assert np.all(np.isfinite(n_w)) and np.all(np.isfinite(n_c))
    assert np.all(n_w < n_c)
    # stronger radiation field -> both phase boundaries move to higher
    # density (more heating needs more density to stay cold)
    assert np.all(np.diff(n_w) > 0)

    # at the solar value the boundaries must sit either side of ~1 cm^-3,
    # i.e. the classification is actually doing work on real ISM densities
    solar = np.array([1.0])
    assert 0.3 < b.n_w_max(solar)[0] < 3.0
    assert 3.0 < b.n_c_min(solar)[0] < 30.0
