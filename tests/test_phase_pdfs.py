"""
Fig. 5: the slab histograms of scripts/design/fig_phase_pdfs.py.

1. NORMALISATION: counts / (N_slab x bin width) integrates to each phase's
   volume fraction, and the four fractions sum to 1.
2. SLAB SELECTION on a synthetic grid: a slab made of whole bins is accepted,
   one that cuts a bin is refused.
3. The cached histograms (skipped if the cache or the cube coordinates are
   absent): every slab's counts + under/overflow equal its cell counts, the
   fractions sum to 1 to 1e-12, and every curve's area equals its fraction.
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


def test_area_is_volume_fraction():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 50, (4, len(EDGES) - 1)).astype(float)
    n_cells = counts.sum(axis=1)
    s = fp.slab_summary(counts, np.zeros(4), np.zeros(4), n_cells, EDGES)
    assert abs(s["vol_frac"].sum() - 1.0) < 1e-12
    for j, p in enumerate(fp.ALL_PHASES):
        np.testing.assert_allclose(s["area"][p], n_cells[j] / n_cells.sum(), rtol=1e-12)
    assert abs(sum(s["mass_frac"].values()) - 1.0) < 1e-12


def test_slab_selection_whole_bins_only():
    z_box = np.arange(-400.0, 400.0 + 1e-9, 2.0)
    bins = [(c, np.where((z_box >= c - 2) & (z_box < c + 2))[0]) for c in np.arange(-398, 400, 4.0)]
    bins[-1] = (bins[-1][0], np.where(z_box >= 396)[0])
    # z_mid = -12 -> slabs from -42 to +18: cuts the bin [-44, -40)
    with pytest.raises(RuntimeError, match="whole"):
        fp.slab_bin_sets(z_box, bins, -12.0, d_list=(0.0,))
    # z_mid = -10 -> slab -40 .. 20: bins [-40, -36) .. [16, 20) plus the 20 plane -> cuts [20, 24)
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
    for dd, (counts, under, over, n_cells, _) in slabs.items():
        np.testing.assert_array_equal(counts.sum(axis=1) + under + over, n_cells)
        s = fp.slab_summary(counts, under, over, n_cells, edges)
        assert abs(s["vol_frac"].sum() - 1.0) < 1e-12
        for j, p in enumerate(fp.ALL_PHASES):
            c = fp.normalised(counts[j], n_cells.sum(), edges)
            assert abs(np.sum(c * np.diff(edges)) - s["vol_frac"][j]) < 1e-12
