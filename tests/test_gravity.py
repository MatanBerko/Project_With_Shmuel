"""
g_ext must be symmetric in z. g_gas must vanish at z=0 and equal
2*pi*G*Sigma_total at the domain edges.
"""

import numpy as np

from src.physics.derived import sigma_gas_map
from src.physics.gravity import GRAV_PREFACTOR_CGS, g_ext_cgs, g_gas_cgs


def test_g_ext_symmetric():
    z = np.linspace(-500, 500, 501)
    g = g_ext_cgs(z)
    assert np.allclose(g, g_ext_cgs(-z), rtol=1e-12)
    assert np.all(g >= 0)


def test_g_gas_zero_at_midplane_and_total_at_edges():
    z = np.arange(-400.0, 400.0 + 1.0, 2.0)
    rng = np.random.default_rng(0)
    n = 0.3 + 0.2 * np.abs(np.sin(z / 87.0)) + 0.05 * rng.random(z.shape)  # positive, non-trivial

    g_gas = g_gas_cgs(z, n)

    i_mid = np.argmin(np.abs(z))
    assert abs(g_gas[i_mid]) < 1e-8 * max(abs(g_gas).max(), 1.0)

    sigma_total = sigma_gas_map(z, n)
    expected_edge_g = GRAV_PREFACTOR_CGS * sigma_total

    i_top = np.argmax(z)
    i_bot = np.argmin(z)
    assert np.isclose(g_gas[i_top], expected_edge_g, rtol=1e-6)
    assert np.isclose(g_gas[i_bot], expected_edge_g, rtol=1e-6)


def test_g_gas_monotonic_with_z_for_positive_density():
    z = np.arange(0.0, 400.0 + 1.0, 2.0)
    n = np.full_like(z, 0.4)
    z_full = np.concatenate([-z[::-1][:-1], z])
    n_full = np.concatenate([n[::-1][:-1], n])
    g_gas = g_gas_cgs(z_full, n_full)
    pos = z_full >= 0
    g_pos = g_gas[pos]
    assert np.all(np.diff(g_pos) >= -1e-12)
