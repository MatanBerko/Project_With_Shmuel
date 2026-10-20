"""
Step 2f: the random-patch machinery.

1. Summed-area patch sums equal brute-force sums on a toy cube, for many
   patch sizes and positions, including patches touching the far edges.
2. Combining planes into a z group by summing per-plane patch sums gives
   the same patch quantities as computing them from the group's cells
   directly -- and differs from averaging the per-plane means, which is
   what the script must NOT do.
3. A patch with no non-HIM cell in a group gives NaN for every P_th-based
   quantity (and a finite <P_tot>), and is counted as NaN across patches.
4. The seed makes the patch list reproducible, every patch lies inside
   the footprint, and a different seed gives a different list.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compute_patch_alpha",
    Path(__file__).resolve().parents[1] / "scripts" / "compute" / "compute_patch_alpha.py")
cpa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpa)

NF = 23      # odd footprint, like the real 501
NZ = 4


def _toy(seed=7, him_frac=0.3):
    rng = np.random.default_rng(seed)
    ptot = {sg: np.exp(rng.normal(8.5, 0.5, (NZ, NF, NF))) for sg in cpa.SELF_GRAVITY_SETTINGS}
    pth = np.exp(rng.normal(7.8, 0.9, (NZ, NF, NF)))
    n_h = np.exp(rng.normal(0.0, 1.5, (NZ, NF, NF)))
    him = rng.random((NZ, NF, NF)) < him_frac
    return ptot, pth, n_h, him


def _fields(ptot, pth, n_h, him, k):
    return cpa.plane_fields({sg: p[k] for sg, p in ptot.items()}, pth[k], n_h[k], him[k])


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 5, 11, NF])
def test_summed_area_patch_sums_match_brute_force(m):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(NF, NF))
    offs = cpa.draw_patch_offsets(NF, m, n_patch=200, seed=3)
    # make sure both far edges are exercised
    offs = np.vstack([offs, [[NF - m, NF - m], [0, NF - m], [NF - m, 0]]])
    got = cpa.patch_sums(cpa.summed_area_table(a), offs, m)
    want = np.array([a[iy:iy + m, ix:ix + m].sum() for iy, ix in offs])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_group_sums_equal_direct_cell_means_not_mean_of_plane_means():
    ptot, pth, n_h, him = _toy()
    m = 9
    offs = cpa.draw_patch_offsets(NF, m, n_patch=50, seed=2)
    acc = {k: np.zeros((1, len(offs))) for k in cpa.SUM_KEYS}
    for k in range(NZ):
        f = _fields(ptot, pth, n_h, him, k)
        sats = {key: cpa.summed_area_table(a) for key, a in f.items()}
        cpa.accumulate_plane(acc, [0], sats, offs, m)
    q = cpa.quantities_from_sums(acc, m * m, [NZ])

    plane_means = []
    for j, (iy, ix) in enumerate(offs):
        sl = (slice(None), slice(iy, iy + m), slice(ix, ix + m))
        neu = ~him[sl]
        p = pth[sl][neu]
        n = n_h[sl][neu]
        pt = ptot["mean"][sl]
        want = {
            "Ptot": pt.mean(),
            "Pth_inv_inv_vol": 1.0 / np.mean(1.0 / p),
            "Pth_inv_inv_mw": np.sum(n) / np.sum(n / p),
            "alpha_vol": pt.mean() * np.mean(1.0 / p),
            "alpha_mw": pt.mean() * np.sum(n / p) / np.sum(n),
            "alpha_area_vol": pt.mean() / p.mean(),
        }
        for name, v in want.items():
            assert q["mean"][name][0, j] == pytest.approx(v, rel=1e-12), name
        # the wrong way: mean over planes of each plane's harmonic mean
        per_plane = [1.0 / np.mean(1.0 / pth[k, iy:iy + m, ix:ix + m][~him[k, iy:iy + m, ix:ix + m]])
                     for k in range(NZ)]
        plane_means.append(np.mean(per_plane))
    # unequal non-HIM counts per plane make the two differ
    assert not np.allclose(plane_means, q["mean"]["Pth_inv_inv_vol"][0], rtol=1e-6)


def test_patch_without_neutral_cells_is_nan():
    ptot, pth, n_h, him = _toy()
    m = 5
    offs = np.array([[0, 0], [10, 10]])
    him[:, 0:m, 0:m] = True          # first patch: every cell HIM-flagged
    acc = {k: np.zeros((1, 2)) for k in cpa.SUM_KEYS}
    for k in range(NZ):
        f = _fields(ptot, pth, n_h, him, k)
        sats = {key: cpa.summed_area_table(a) for key, a in f.items()}
        cpa.accumulate_plane(acc, [0], sats, offs, m)
    q = cpa.quantities_from_sums(acc, m * m, [NZ])["mean"]
    assert np.isfinite(q["Ptot"][0, 0])                 # P_tot keeps every cell
    for name in ("Pth_inv_inv_vol", "Pth_inv_inv_mw", "alpha_vol", "alpha_mw",
                 "alpha_area_vol"):
        assert np.isnan(q[name][0, 0]), name
        assert np.isfinite(q[name][0, 1]), name
    stats, n_nan = cpa.across_patches(q["alpha_vol"])
    assert n_nan[0] == 1
    assert stats[0, 1] == q["alpha_vol"][0, 1]          # median of the one finite patch


def test_seed_makes_patch_list_reproducible():
    a = cpa.draw_patch_offsets(501, 250)
    b = cpa.draw_patch_offsets(501, 250)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (cpa.N_PATCH, 2)
    assert a.min() >= 0 and a.max() <= 501 - 250
    c = cpa.draw_patch_offsets(501, 250, seed=cpa.PATCH_SEED + 1)
    assert not np.array_equal(a, c)


def test_overlap_and_coverage():
    offs = np.array([[0, 0], [0, 5], [5, 5], [2, 0]])
    ov = cpa.overlap_fractions(offs, 5)
    # pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    np.testing.assert_allclose(ov, [0.0, 0.0, 0.6, 0.0, 0.0, 0.0])
    cov = cpa.coverage_map(offs, 5, 10)
    brute = np.zeros((10, 10), int)
    for iy, ix in offs:
        brute[iy:iy + 5, ix:ix + 5] += 1
    np.testing.assert_array_equal(cov, brute)
