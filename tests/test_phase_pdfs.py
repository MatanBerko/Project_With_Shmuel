"""
Fig. 5: the slab histograms of scripts/design/fig_phase_pdfs.py.

1. NORMALISATION: volume curves integrate to each phase's volume fraction
   (HIM in the total) and the four fractions sum to 1; mass curves integrate
   to each phase's fraction of the NEUTRAL mass and the three sum to 1.
2. SLAB SELECTION on a synthetic grid: a slab made of whole bins is accepted,
   one that cuts a bin is refused.
3. The cached histograms (skipped if the cache or the cube coordinates are
   absent): per slab, counts + under/overflow equal the cell counts, the mass
   histograms summed over bins equal the cached sum of n_H of the phase's
   cells, both sets of fractions sum to 1 to 1e-12, and every curve's area
   equals its fraction.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "fig_phase_pdfs", _ROOT / "scripts" / "design" / "fig_phase_pdfs.py")
fp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fp)

EDGES = np.round(np.arange(-4.0, 3.0 + 1e-9, 0.05), 10)
NEUTRAL = [fp.ALL_PHASES.index(p) for p in fp.PANEL_PHASES]


def _synthetic_slab(seed=1):
    rng = np.random.default_rng(seed)
    nb = len(EDGES) - 1
    counts = rng.integers(0, 50, (4, nb)).astype(float)
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    mass = counts * 10.0 ** centres * rng.uniform(0.95, 1.05, (4, nb))
    zeros = np.zeros(4)
    return {"counts": counts, "under": zeros, "over": zeros, "n_cells": counts.sum(axis=1),
            "mass": mass, "m_under": zeros, "m_over": zeros, "sum_n": mass.sum(axis=1)}


def test_volume_and_mass_normalisation():
    h = _synthetic_slab()
    s = fp.slab_summary(h, EDGES)
    assert abs(s["vol_frac"].sum() - 1.0) < 1e-12
    assert abs(sum(s["mass_frac"].values()) - 1.0) < 1e-12
    m_neu = h["mass"][NEUTRAL].sum()
    assert fp.neutral_mass(h) == pytest.approx(m_neu, rel=1e-14)
    for j, p in enumerate(fp.ALL_PHASES):
        np.testing.assert_allclose(s["area_v"][p], h["n_cells"][j] / h["n_cells"].sum(),
                                   rtol=1e-12)
        if p != "HIM":
            np.testing.assert_allclose(s["area_m"][p], h["mass"][j].sum() / m_neu, rtol=1e-12)
            np.testing.assert_allclose(s["area_m"][p], s["mass_frac"][p], rtol=1e-12)
    # HIM mass does not enter the neutral normalisation
    h2 = dict(h, mass=h["mass"].copy())
    h2["mass"][fp.ALL_PHASES.index("HIM")] *= 1e6
    assert fp.slab_summary(h2, EDGES)["mass_frac"] == pytest.approx(s["mass_frac"], rel=1e-14)


def test_mass_weighting_shifts_median_up():
    # same counts, mass proportional to n: the mass-weighted median is never lower
    h = _synthetic_slab(3)
    s = fp.slab_summary(h, EDGES)
    for p in fp.PANEL_PHASES:
        assert s["m"]["median"][p] >= s["v"]["median"][p]


def test_slab_selection_whole_bins_only():
    z_box = np.arange(-400.0, 400.0 + 1e-9, 2.0)
    bins = [(c, np.where((z_box >= c - 2) & (z_box < c + 2))[0]) for c in np.arange(-398, 400, 4.0)]
    bins[-1] = (bins[-1][0], np.where(z_box >= 396)[0])
    # z_mid = -12 -> slab -42 .. +18: cuts the bin [-44, -40)
    with pytest.raises(RuntimeError, match="whole"):
        fp.slab_bin_sets(z_box, bins, -12.0, d_list=(0.0,))
    # z_mid = -10 -> slab -40 .. +20: the +20 plane cuts the bin [20, 24)
    with pytest.raises(RuntimeError):
        fp.slab_bin_sets(z_box, bins, -10.0, d_list=(0.0,))
    # z_mid = -11 -> planes -40 .. 18: exactly fifteen bins
    out = fp.slab_bin_sets(z_box, bins, -11.0, d_list=(0.0, 100.0))
    assert len(out[0.0]) == 15 and len(out[100.0]) == 30


def _load():
    if not fp.CACHE.exists():
        pytest.skip("cache/turbulence_pdf/turbulence_pdf.npz not built")
    try:
        return fp.load_slabs()
    except (FileNotFoundError, KeyError, OSError) as e:
        pytest.skip(f"cube coordinates not available: {e}")


def test_cached_slabs():
    d, edges, slabs = _load()
    for dd, h in slabs.items():
        np.testing.assert_array_equal(h["counts"].sum(axis=1) + h["under"] + h["over"],
                                      h["n_cells"])
        # mass histograms summed over bins == sum of n_H of the phase's cells
        np.testing.assert_allclose(h["mass"].sum(axis=1) + h["m_under"] + h["m_over"],
                                   h["sum_n"], rtol=1e-12, atol=0.0)
        s = fp.slab_summary(h, edges)
        assert abs(s["vol_frac"].sum() - 1.0) < 1e-12
        assert abs(sum(s["mass_frac"].values()) - 1.0) < 1e-12
        m_neu = fp.neutral_mass(h)
        for j, p in enumerate(fp.ALL_PHASES):
            cv_ = fp.normalised(h["counts"][j], h["n_cells"].sum(), edges)
            assert abs(np.sum(cv_ * np.diff(edges)) - s["vol_frac"][j]) < 1e-12
            if p != "HIM":
                cm = fp.normalised(h["mass"][j], m_neu, edges)
                assert abs(np.sum(cm * np.diff(edges)) - s["mass_frac"][p]) < 1e-12
