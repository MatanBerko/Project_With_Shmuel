"""
HIM classification, phase classification, and RAW/HIM_A/HIM_B variant
construction on a small synthetic (toy) cube, using the REAL BS19
P_min(I_UV)/P_max(I_UV) interpolators (so the threshold values are
physically real, not invented), with n/T chosen deterministically relative
to those thresholds so the expected classification is known in advance.
"""

import numpy as np
import pytest

from src.config_loader import load_resolved_config
from src.conventions import T_HIM_K
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM, \
    apply_variant, him_flag, phase_flag
from src.physics.thermal import build_pmin_pmax


@pytest.fixture(scope="module")
def pmin_pmax():
    cfg = load_resolved_config()
    return build_pmin_pmax(cfg["bs19_mat_path"])


def test_him_and_phase_and_variants_on_toy_cube(pmin_pmax):
    iuv = np.array([1.0, 1.0, 1.0, 1.0])
    Pmin = pmin_pmax.p_min(iuv)
    Pmax = pmin_pmax.p_max(iuv)
    assert np.all(np.isfinite(Pmin)) and np.all(np.isfinite(Pmax))
    assert np.all(Pmax >= Pmin)  # UNM/CNM boundary should sit above WNM/UNM boundary

    # cell 0: deep HIM (Pth << 0.5*Pmin)
    # cell 1: safely non-HIM, UNM temperature
    # cell 2: safely non-HIM, CNM temperature (T < 300)
    # cell 3: safely non-HIM, WNM temperature (T > 6000)
    T = np.array([500.0, 1000.0, 200.0, 10000.0])
    Pth_target = np.array([0.1 * Pmin[0], 10 * Pmax[1], 10 * Pmax[2], 10 * Pmax[3]])
    n_raw = Pth_target / T
    Pth_raw = n_raw * T

    him = him_flag(Pth_raw, Pmin)
    np.testing.assert_array_equal(him, [True, False, False, False])

    phase = phase_flag(T, him)
    np.testing.assert_array_equal(
        phase, [PHASE_HIM, PHASE_UNM, PHASE_CNM, PHASE_WNM]
    )

    raw = apply_variant("RAW", n_raw, Pth_raw, him, Pmin, Pmax)
    np.testing.assert_allclose(raw.n_model, n_raw)
    np.testing.assert_allclose(raw.Pth_model, Pth_raw)

    him_a = apply_variant("HIM_A", n_raw, Pth_raw, him, Pmin, Pmax)
    assert np.isclose(him_a.Pth_model[0], Pmin[0])
    assert np.isclose(him_a.n_model[0], Pmin[0] / T_HIM_K)
    np.testing.assert_allclose(him_a.n_model[1:], n_raw[1:])
    np.testing.assert_allclose(him_a.Pth_model[1:], Pth_raw[1:])

    him_b = apply_variant("HIM_B", n_raw, Pth_raw, him, Pmin, Pmax)
    assert np.isclose(him_b.Pth_model[0], Pmax[0])
    assert np.isclose(him_b.n_model[0], Pmax[0] / T_HIM_K)
    np.testing.assert_allclose(him_b.n_model[1:], n_raw[1:])
    np.testing.assert_allclose(him_b.Pth_model[1:], Pth_raw[1:])


def test_masked_variant_not_implemented(pmin_pmax):
    """No MASKED definition exists in any of the four reference scripts
    (grepped exhaustively) -- confirms this is a deliberate, reported
    omission, not an accidental gap.
    """
    n = np.array([1.0])
    Pth = np.array([1.0])
    him = np.array([False])
    Pmin = np.array([1.0])
    Pmax = np.array([2.0])
    with pytest.raises(ValueError, match="MASKED"):
        apply_variant("MASKED", n, Pth, him, Pmin, Pmax)
