"""
Fig. 6: the density-PDF statistics of scripts/compute/compute_turbulence_pdf.py.

1. A SYNTHETIC LOGNORMAL FIELD with known sigma_s: std and (p84 - p16)/2 of
   s = ln(n/<n>) recover it, are independent of the normalisation of n, and
   M2_pdf = (exp(sigma_s^2) - 1) / (3 b^2) for every b.
2. c_s AND THE SAMPLES are compute_velocity.py's: the same cells per sample
   (NEUTRAL = every non-HIM cell) and the same mass-weighted c_s.
3. PHASE HISTOGRAMS: counts + under/overflow = the cell count of each phase.
4. BLOCK AVERAGING: block n is the plain 8-cell mean; the phase is the
   strict majority (>= 5), blocks without one are skipped.
5. The cached outputs (skipped if absent): c_s equals velocity.npz's, and
   the stored histograms sum to the stored cell counts.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.conventions import K_B, M_H, MU
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compute_turbulence_pdf", _ROOT / "scripts" / "compute" / "compute_turbulence_pdf.py")
ct = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ct)
cv = ct.cv

SHAPE = (4, 20, 20)


def test_lognormal_sigma_and_mach():
    rng = np.random.default_rng(11)
    sigma = 0.8
    n = np.exp(rng.normal(1.3, sigma, 2_000_000))
    w = ct.pdf_width(n, n_min=1)
    assert abs(w["sigma_s"] - sigma) < 2e-3
    assert abs(w["sigma_s_robust"] - sigma) < 5e-3
    # s is relative to the VOLUME mean: <s> = -sigma^2/2 for a lognormal
    assert abs(w["mean_s"] + 0.5 * sigma ** 2) < 5e-3
    # independent of the normalisation of n
    w2 = ct.pdf_width(37.0 * n, n_min=1)
    np.testing.assert_allclose(w2["sigma_s"], w["sigma_s"], rtol=1e-12)
    for b in (1 / 3, 0.4, 1.0):
        np.testing.assert_allclose(ct.mach2_3d(sigma, b), (np.exp(sigma ** 2) - 1) / b ** 2,
                                   rtol=1e-14)
        np.testing.assert_allclose(ct.mach2_1d(sigma, b), (np.exp(sigma ** 2) - 1) / (3 * b ** 2),
                                   rtol=1e-14)
    # b = 1/3 gives 9x the M^2 of b = 1
    np.testing.assert_allclose(ct.mach2_1d(sigma, 1 / 3), 9 * ct.mach2_1d(sigma, 1.0), rtol=1e-13)
    # and the derived keys
    d = ct.derived({"sigma_s": np.array([sigma]), "sigma_s_robust": np.array([sigma])},
                   np.array([2.0]))
    np.testing.assert_allclose(d["sigma_turb_pdf_b0p4"],
                               2.0 * np.sqrt(np.expm1(sigma ** 2) / (3 * 0.16)), rtol=1e-14)
    # below the cell threshold everything is NaN
    assert np.isnan(ct.pdf_width(n[:50])["sigma_s"])


def test_uniform_density_has_zero_width():
    w = ct.pdf_width(np.full(1000, 0.3), n_min=1)
    assert w["sigma_s"] < 1e-14 and w["sigma_s_robust"] < 1e-14     # rounding of ln(n/<n>)
    assert ct.mach2_1d(w["sigma_s"], 0.4) < 1e-27


def test_samples_and_cs_identical_to_compute_velocity():
    rng = np.random.default_rng(7)
    phase = rng.choice([PHASE_CNM, PHASE_UNM, PHASE_WNM, PHASE_HIM], SHAPE,
                       p=[0.2, 0.1, 0.6, 0.1])
    n_H = np.exp(rng.normal(0.0, 1.5, SHAPE))
    p_th = rng.uniform(500.0, 8000.0, SHAPE)
    W = 7000.0
    comb = cv.combine(cv.phase_sums(n_H, p_th, phase, W))
    st = {k: cv.velocity_stats(comb[k], W, comb["ALL"]["n"], n_min=1)[0] for k in ct.SAMPLES}
    masks = ct.sample_masks(phase)
    assert not masks["NEUTRAL"][phase == PHASE_HIM].any()
    assert masks["NEUTRAL"].sum() == (masks["CNM"] | masks["UNM"] | masks["WNM"]).sum()
    for k in ct.SAMPLES:
        m = masks[k]
        assert st[k]["n_cells"] == m.sum() == ct.pdf_width(n_H[m], n_min=1)["n"]
        cs = np.sqrt(K_B * p_th[m].sum() / (MU * M_H * n_H[m].sum())) / 1e5
        np.testing.assert_allclose(st[k]["sigma_th"], cs, rtol=1e-13)


def test_phase_histograms_sum_to_cell_counts():
    rng = np.random.default_rng(2)
    phase = rng.integers(0, 4, 50_000)
    log_n = rng.normal(-0.5, 2.0, 50_000)           # spills past both edges
    counts, under, over = ct.phase_histograms(log_n, phase)
    assert under.sum() > 0 and over.sum() > 0
    for j, (_, code) in enumerate(ct.HIST_PHASES):
        assert counts[j].sum() + under[j] + over[j] == np.sum(phase == code)


def test_block_average_majority():
    n2 = np.zeros((2, 4, 4))
    ph2 = np.zeros((2, 4, 4), dtype=np.int8)
    n2[:, :2, :2] = np.arange(8.0).reshape(2, 2, 2)  # block (0,0): mean 3.5
    ph2[:, :2, :2] = PHASE_WNM
    ph2[0, 0, 0] = PHASE_CNM                          # 7 WNM, 1 CNM -> WNM
    ph2[:, :2, 2:] = PHASE_CNM
    ph2[0, :2, 2:] = PHASE_WNM                        # 4 / 4 -> no majority
    ph2[:, 2:, :2] = PHASE_HIM
    ph2[0, 2, 0] = PHASE_WNM                          # 7 HIM -> HIM, not neutral
    ph2[:, 2:, 2:] = PHASE_HIM
    ph2[:, 2, 2:] = PHASE_CNM
    ph2[0, 3, 2] = PHASE_WNM                          # 4 CNM, 1 WNM, 3 HIM: no phase, neutral
    n_b, code, neu = ct.block_average(n2, ph2)
    assert n_b[0] == 3.5
    np.testing.assert_array_equal(code, [PHASE_WNM, -1, PHASE_HIM, -1])
    np.testing.assert_array_equal(neu, [True, True, False, True])   # 4 CNM + 4 WNM is neutral
    # an odd x, y extent is cropped to whole blocks
    n_b, code, _ = ct.block_average(np.ones((2, 5, 5)), np.zeros((2, 5, 5), int))
    assert n_b.size == 4
    # only complete, grid-aligned z pairs inside the group are used
    pl = {i: (np.full((4, 4), float(i)), np.zeros((4, 4), int)) for i in (3, 4, 5, 6)}
    n_b, _, _ = ct.blocks_of_group(pl)
    np.testing.assert_array_equal(np.unique(n_b), [4.5])


def test_resolution_lowers_width():
    # white-noise lognormal: block averaging must narrow the distribution
    rng = np.random.default_rng(4)
    n = np.exp(rng.normal(0.0, 1.0, (2, 60, 60)))
    ph = np.full(n.shape, PHASE_WNM)
    n_b, code, _ = ct.block_average(n, ph)
    assert ct.pdf_width(n_b[code == PHASE_WNM])["sigma_s"] < ct.pdf_width(n)["sigma_s"]


_OUT = _ROOT / "cache" / "turbulence_pdf" / "turbulence_pdf.npz"
_VEL = _ROOT / "cache" / "velocity" / "velocity.npz"


@pytest.mark.skipif(not (_OUT.exists() and _VEL.exists()), reason="caches not built")
def test_cached_outputs_consistent():
    d = np.load(_OUT, allow_pickle=False)
    v = np.load(_VEL, allow_pickle=False)
    np.testing.assert_array_equal(d["z_pc"], v["z_pc"])
    for k in ct.SAMPLES:
        for pre in ("", "slab__"):
            np.testing.assert_array_equal(d[f"{pre}{k}__c_s"], v[f"{pre}{k}__sigma_th"])
            np.testing.assert_array_equal(d[f"{pre}{k}__n"], v[f"{pre}{k}__n_cells"])
        s = d[f"{k}__sigma_s"]
        assert np.all(s[np.isfinite(s)] > 0)
    for pre in ("", "slab__"):
        tot = d[f"{pre}hist_log_n"].sum(axis=2) + d[f"{pre}hist_log_n_under"] \
            + d[f"{pre}hist_log_n_over"]
        np.testing.assert_array_equal(tot, d[f"{pre}hist_n_cells"])
