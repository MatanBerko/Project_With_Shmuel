"""
fig_phase_pdfs -- Fig. 5: phase density distributions vs height.

Reads the volume-weighted log10(n_H) histograms stored per signed 4 pc bin in
cache/turbulence_pdf/turbulence_pdf.npz (scripts/compute/compute_turbulence_pdf.py;
CNM, UNM, WNM, HIM from the dP/dn phase flag, as Figs. 6-7). Nothing is
recomputed from the cube: the plane z coordinates are read (coordinates only)
to select slabs with the same helpers the compute scripts use.

Slabs, |z - z_c| <= 30 pc (src.physics.loading.slab_z_indices about
midplane.slab_center(0, z_mid)):
  d = 0               z_c = z_mid
  d = 100, 200, 300   the union of the slabs at z_mid - d and z_mid + d
Each slab is asserted to be an exact union of whole 4 pc bins, and its
histogram is the sum of theirs; d = 0 and +-300 are checked against the
slab histograms the compute script stored for the same slabs.

Two rows x three columns (CNM | UNM | WNM), x shared within a column, log y
shared within a row:
  top    (a)-(c) VOLUME weighted: cell counts / (N_slab x Delta log10 n),
         N_slab = every cell of the slab (HIM included), so the area under a
         curve is the phase's volume fraction of the slab.
  bottom (d)-(f) MASS weighted: the cached sums of n_H per bin (mass is
         proportional to n_H: 2 pc cubes, rho = 1.4 m_H n_H) / (M_neutral x
         Delta log10 n), M_neutral = sum n_H over the slab's CNM + UNM + WNM
         cells, so the area is the phase's fraction of the NEUTRAL mass. HIM
         is excluded from this normalisation because its n_H is not physical.
HIM is not drawn; its volume fraction is printed in the top WNM panel.
Within a panel the four slabs are shades of the phase colour, darkest =
midplane. A curve (both rows) is dropped when the phase has fewer than
N_MIN_CELLS cells in the slab.

Also writes results/phase_pdfs.txt (volume and exact neutral-mass fractions,
volume- and mass-weighted median and 16-84% of log10 n_H per phase and slab).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    CNM,
    MIDPLANE_MODE,
    PERCENTILE_SCHEME_16_84,
    PROVISIONAL_CUBE_HEADER,
    THERMAL_PRESSURE_HEADER,
    UNM,
    WNM,
)
from src.physics import midplane  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    panel_label,
    provisional_label,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/turbulence_pdf/turbulence_pdf.npz")
NAME = "fig_phase_pdfs"
OUT_TXT_PATH = Path("results/phase_pdfs.txt")
FIG_W, FIG_H = 6.6, 4.9
SLAB_D_PC = (0.0, 100.0, 200.0, 300.0)       # |z_c - z_mid|
PANEL_PHASES = ("CNM", "UNM", "WNM")
N_MIN_CELLS = 100                            # drop a curve below this many cells
X_MARGIN_DEX = 0.15                          # around the drawn non-zero range
Y_MARGIN_DEX = 0.25                          # below the drawn non-zero range
Y_TOP_MARGIN_DEX = 0.75                      # above it: room for the panel labels
LINE_LW = 1.1
TEXT_FS = 6.5
# shades per phase, darkest (d = 0) to lightest (d = 300): (mix target, fraction)
BASE = {"CNM": CNM, "UNM": UNM, "WNM": WNM}
SHADES = {
    "CNM": (("k", 0.40), ("k", 0.0), ("w", 0.30), ("w", 0.52)),
    "UNM": (("k", 0.70), ("k", 0.55), ("k", 0.40), ("k", 0.25)),   # base is too pale on white
    "WNM": (("k", 0.45), ("k", 0.15), ("w", 0.20), ("w", 0.42)),
}
LEGEND_GREYS = ("0.10", "0.35", "0.55", "0.70")
MIN_CONTRAST = 1.9                           # lightest shade vs white (WCAG ratio), warn below
PCT_SCHEME = PERCENTILE_SCHEME_16_84
MASS_SUM_RTOL = 1e-12                        # mass histograms vs the cached sum n_H
# Normalisation of the drawn curves:
#   "unit_area"  each curve integrates to 1 over log10 n_H (shape only; linear y
#                from 0 without tick labels, arbitrary units -- Shelest+26 Fig. 3a)
#   "fraction"   area = volume fraction (top) / neutral mass fraction (bottom);
#                log y with tick labels and the HIM text (the previous version)
NORM = "unit_area"
ROW_LABELS = {"unit_area": ("volume-weighted PDF (arb. units)", "mass-weighted PDF (arb. units)"),
              "fraction": ("volume fraction per dex", "neutral mass fraction per dex")}
PER_COLUMN_X = True                          # each column zooms to its own phase's range
Y_TOP_PAD_LINEAR = 0.08                      # unit_area: headroom above the highest curve
# ═══════════════════════════════════════════════════════════════════════

ALL_PHASES = ("CNM", "UNM", "WNM", "HIM")


def shade(rgb, target, frac):
    """Mix an RGB colour with black ('k') or white ('w') by frac."""
    c = np.asarray(to_rgb(rgb))
    t = np.zeros(3) if target == "k" else np.ones(3)
    return tuple((1 - frac) * c + frac * t)


def contrast_vs_white(rgb):
    """WCAG contrast ratio of a colour against white."""
    c = np.asarray(to_rgb(rgb))
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 1.05 / (lum + 0.05)


def slab_bin_sets(z_box, bin_groups, z_mid, d_list=SLAB_D_PC):
    """{d: sorted bin indices} for the slab(s) at z_mid +- d.

    Raises if a slab is not an exact union of whole bins.
    """
    from src.physics.loading import slab_z_indices
    plane_bin = np.full(len(z_box), -1)
    for b, (_, idx) in enumerate(bin_groups):
        plane_bin[np.asarray(idx, int)] = b
    center = midplane.slab_center(0.0, z_mid, MIDPLANE_MODE)
    out = {}
    for dd in d_list:
        planes = set()
        for sgn in ((+1,) if dd == 0 else (-1, +1)):
            planes |= set(slab_z_indices(z_box, center + sgn * dd).tolist())
        bins = sorted({int(plane_bin[i]) for i in planes})
        if -1 in bins:
            raise RuntimeError(f"d = {dd}: slab planes outside every bin")
        covered = set(np.concatenate([np.asarray(bin_groups[b][1], int) for b in bins]).tolist())
        if covered != planes:
            raise RuntimeError(f"d = {dd}: the slab is not a union of whole 4 pc bins")
        out[dd] = bins
    return out


def slab_histogram(d, bins):
    """Summed volume and mass histograms of a set of 4 pc bins (dict of (4, ...) arrays)."""
    b = np.asarray(bins, int)
    keys = {"counts": "hist_log_n", "under": "hist_log_n_under", "over": "hist_log_n_over",
            "n_cells": "hist_n_cells", "mass": "hist_mass_log_n",
            "m_under": "hist_mass_log_n_under", "m_over": "hist_mass_log_n_over",
            "sum_n": "hist_sum_n_H"}
    return {k: d[v][b].sum(axis=0) for k, v in keys.items()}


def normalised(counts, n_total, edges):
    """weights / (total * bin width in dex): area = fraction of the total."""
    return counts / (n_total * np.diff(edges))


def plotted_curve(weights, total, edges, norm=NORM):
    """The drawn density per dex: unit area ("unit_area") or area = weights.sum()/total ("fraction")."""
    if norm == "unit_area":
        return normalised(weights, weights.sum(), edges)
    if norm == "fraction":
        return normalised(weights, total, edges)
    raise ValueError(f"unknown NORM {norm!r}")


def neutral_mass(h):
    """Total n_H of the slab's CNM + UNM + WNM cells (the mass normalisation)."""
    j = [ALL_PHASES.index(p) for p in PANEL_PHASES]
    return float((h["mass"].sum(axis=1) + h["m_under"] + h["m_over"])[j].sum())


def slab_summary(h, edges):
    """Volume and neutral-mass fractions, areas and log10 n_H percentiles of one slab."""
    n_total = float(h["n_cells"].sum())
    m_neu = neutral_mass(h)
    m_phase = h["mass"].sum(axis=1) + h["m_under"] + h["m_over"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {"N": n_total, "M_neutral": m_neu, "vol_frac": h["n_cells"] / n_total,
           "mass_frac": {}, "area_v": {}, "area_m": {}, "n": {},
           "v": {"median": {}, "p_lo": {}, "p_hi": {}}, "m": {"median": {}, "p_lo": {}, "p_hi": {}}}
    for j, p in enumerate(ALL_PHASES):
        out["n"][p] = float(h["n_cells"][j])
        out["area_v"][p] = float(np.sum(normalised(h["counts"][j], n_total, edges)
                                        * np.diff(edges)))
        if p == "HIM":
            continue
        out["mass_frac"][p] = float(m_phase[j] / m_neu)
        out["area_m"][p] = float(np.sum(normalised(h["mass"][j], m_neu, edges) * np.diff(edges)))
        for tag, w in (("v", h["counts"][j]), ("m", h["mass"][j])):
            if w.sum() == 0:
                vals = (np.nan, np.nan, np.nan)
            else:
                st = weighted_stats(centres, w, PCT_SCHEME)
                vals = (st.median, st.p_lo, st.p_hi)
            for q, v in zip(("median", "p_lo", "p_hi"), vals):
                out[tag][q][p] = v
    return out


def load_slabs():
    """The cache, its edges and {d: slab histogram dict (+ "bins")}, with all checks."""
    import pipeline.compute_all as pipe
    from src.physics.loading import open_zarr

    d = np.load(CACHE, allow_pickle=False)
    if tuple(d["hist_phases"]) != ALL_PHASES:
        raise RuntimeError(f"unexpected histogram phases {tuple(d['hist_phases'])}")
    if "hist_mass_log_n" not in d.files:
        raise RuntimeError("the cache has no mass histograms: rerun compute_turbulence_pdf.py")
    z_mid = float(d["z_mid_pc"][0])
    _, z_box = pipe._box_z_slice(np.asarray(open_zarr().z_pc, dtype=np.float64))
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    if not np.array_equal([c for c, _ in bin_groups], d["z_pc"]):
        raise RuntimeError("profile bins differ from the cache's")
    edges = d["log_n_edges"]
    slabs = {}
    for dd, bins in slab_bin_sets(z_box, bin_groups, z_mid).items():
        h = slab_histogram(d, bins)
        if not np.array_equal(h["counts"].sum(axis=1) + h["under"] + h["over"], h["n_cells"]):
            raise RuntimeError(f"d = {dd}: histogram counts differ from the cell counts")
        m_tot = h["mass"].sum(axis=1) + h["m_under"] + h["m_over"]
        rel = np.abs(m_tot - h["sum_n"]) / np.maximum(h["sum_n"], 1e-300)
        if rel.max() > MASS_SUM_RTOL:
            raise RuntimeError(f"d = {dd}: mass histograms differ from sum n_H by {rel.max():.2e}")
        h["bins"] = bins
        slabs[dd] = h
    # the cached slabs at z_mid + {0, -300, +300}
    off = list(d["slab_offsets_pc"])
    for key, hk in (("slab__hist_n_cells", "n_cells"), ("slab__hist_log_n", "counts"),
                    ("slab__hist_mass_log_n", "mass")):
        cached = {0.0: d[key][off.index(0.0)],
                  300.0: d[key][off.index(-300.0)] + d[key][off.index(300.0)]}
        for dd, v in cached.items():
            if not np.allclose(slabs[dd][hk], v, rtol=MASS_SUM_RTOL, atol=0.0):
                raise RuntimeError(f"d = {dd}: summed bins differ from the cached {key}")
    return d, edges, slabs


def main():
    apply_paper_style()
    d, edges, slabs = load_slabs()
    centres = 0.5 * (edges[:-1] + edges[1:])
    summ = {dd: slab_summary(slabs[dd], edges) for dd in SLAB_D_PC}
    colors = {p: [shade(BASE[p], *s) for s in SHADES[p]] for p in PANEL_PHASES}
    for p in PANEL_PHASES:
        cr = contrast_vs_white(colors[p][-1])
        print(f"  {p} lightest shade contrast vs white: {cr:.2f}"
              + ("  (WARNING: below MIN_CONTRAST)" if cr < MIN_CONTRAST else ""))

    # curves per row: 0 = volume, 1 = neutral mass
    curves, dropped = ({}, {}), []
    for p in PANEL_PHASES:
        j = ALL_PHASES.index(p)
        for dd in SLAB_D_PC:
            h = slabs[dd]
            if h["n_cells"][j] < N_MIN_CELLS:
                dropped.append((p, dd, int(h["n_cells"][j])))
                continue
            curves[0][(p, dd)] = plotted_curve(h["counts"][j], h["n_cells"].sum(), edges)
            curves[1][(p, dd)] = plotted_curve(h["mass"][j], neutral_mass(h), edges)
    hw = 0.5 * np.diff(edges)[0]
    xlim = {}
    for p in PANEL_PHASES:
        sel = [cv_ for row in curves for (q, _), cv_ in row.items() if q == p or not PER_COLUMN_X]
        nz = np.concatenate([centres[cv_ > 0] for cv_ in sel])
        xlim[p] = (nz.min() - hw - X_MARGIN_DEX, nz.max() + hw + X_MARGIN_DEX)
    linear = NORM == "unit_area"
    ylim = []
    for row in curves:
        pos = np.concatenate([cv_[cv_ > 0] for cv_ in row.values()])
        ylim.append((0.0, pos.max() * (1 + Y_TOP_PAD_LINEAR)) if linear else
                    (pos.min() / 10 ** Y_MARGIN_DEX, pos.max() * 10 ** Y_TOP_MARGIN_DEX))

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H), sharex="col", sharey="row")
    for r in range(2):
        for c, p in enumerate(PANEL_PHASES):
            ax = axes[r, c]
            for i, dd in enumerate(SLAB_D_PC):
                if (p, dd) not in curves[r]:
                    continue
                cv_ = curves[r][(p, dd)]
                if linear:          # trim to the non-zero span and close down to y = 0
                    k = np.where(cv_ > 0)[0]
                    ax.stairs(cv_[k[0]:k[-1] + 1], edges[k[0]:k[-1] + 2], color=colors[p][i],
                              lw=LINE_LW, zorder=5 - i, baseline=0.0)
                else:
                    ax.stairs(cv_, edges, color=colors[p][i], lw=LINE_LW, zorder=5 - i,
                              baseline=None)
            if not linear:
                ax.set_yscale("log")
            ax.set_xlim(*xlim[p])
            ax.set_ylim(*ylim[r])
            if linear:
                ax.set_yticks([])
                ax.minorticks_off()
            ax.text(0.96, 0.96, p, transform=ax.transAxes, ha="right", va="top",
                    fontsize=TEXT_FS + 1.5, color=colors[p][1], fontweight="bold")
            panel_label(ax, "abcdef"[3 * r + c], loc="upper left")
        axes[r, 0].set_ylabel(ROW_LABELS[NORM][r])
    for ax in axes[1]:
        ax.set_xlabel(r"$\log_{10}(n_\mathrm{H}\,/\,\mathrm{cm^{-3}})$")
    if NORM == "fraction":   # the HIM volume fractions (in the caption in unit_area mode)
        him = "\n".join([r"HIM $f_V$:"] + [f"{dd:.0f} pc: {summ[dd]['vol_frac'][3]:.3f}"
                                             for dd in SLAB_D_PC])
        axes[0, 2].text(0.96, 0.62, him, transform=axes[0, 2].transAxes, ha="right", va="top",
                        fontsize=TEXT_FS - 0.5, color="0.25", linespacing=1.15)
    handles = [Line2D([], [], color=g, lw=LINE_LW) for g in LEGEND_GREYS]
    labels = [rf"$|z - z_\mathrm{{mid}}| = {dd:.0f}$ pc" for dd in SLAB_D_PC]
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False,
               fontsize=TEXT_FS, handlelength=2.0, columnspacing=1.4,
               title="darker = closer to the midplane", title_fontsize=TEXT_FS)
    provisional_label(fig, loc="upper right")
    print(f"  NORM = {NORM}; " + "; ".join(f"{p} x {xlim[p][0]:.2f}..{xlim[p][1]:.2f}"
                                           for p in PANEL_PHASES))
    print(f"  y volume {ylim[0][0]:.3g} .. {ylim[0][1]:.3g}; mass {ylim[1][0]:.3g} .. {ylim[1][1]:.3g}")
    print("  dropped curves: " + (", ".join(f"{p} d={dd:.0f} ({n} cells)" for p, dd, n in dropped)
                                  or "none"))
    # area under each drawn curve: 1 (unit_area) or its volume / neutral-mass fraction
    for r, row in enumerate(curves):
        for (p, dd), cv_ in row.items():
            a = float(np.sum(cv_ * np.diff(edges)))
            want = 1.0 if linear else (summ[dd]["vol_frac"][ALL_PHASES.index(p)] if r == 0
                                       else summ[dd]["mass_frac"][p])
            if abs(a - want) > 1e-12:
                raise RuntimeError(f"{p} d={dd} row {r}: area {a} differs from {want}")
    save_figure(fig, NAME)
    write_report(d, summ, dropped)


def write_report(d, summ, dropped):
    pl, ph = percentile_levels(PCT_SCHEME)
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Fig. 5: phase density distributions vs height",
        "=" * 104,
        f"RAW, footprint |x|,|y| <= 500 pc, balanced midplane z_mid = "
        f"{float(d['z_mid_pc'][0]):+.3f} pc. Phases: cached dP/dn flag (CNM, UNM, WNM, HIM),",
        "as Figs. 6-7. Source: the per-4-pc-bin log10 n_H histograms (cell counts and sums of n_H)",
        f"in cache/turbulence_pdf/turbulence_pdf.npz (edges {d['log_n_edges'][0]:g} .. "
        f"{d['log_n_edges'][-1]:g} in {np.diff(d['log_n_edges'])[0]:.2f} dex; under/overflow kept).",
        "Slabs: |z - z_c| <= 30 pc; d = 0 is z_c = z_mid, d = 100/200/300 the union of z_c = "
        "z_mid - d and z_mid + d.",
        "Each slab is an exact union of fifteen (d = 0) or thirty 4 pc bins (asserted).",
        "",
        "  f_V     volume (cell) fraction of the slab, HIM included in the total",
        "  f_m     fraction of the NEUTRAL (CNM+UNM+WNM) mass: sum of n_H over the phase's cells /",
        "          sum of n_H over the slab's neutral cells (mass is proportional to n_H, as every",
        "          cell is a 2 pc cube). HIM is excluded: its n_H is not physical",
        f"  log n   median and p{pl:g}-p{ph:g} of log10(n_H/cm^-3), volume or mass weighted,",
        "          nearest rank on the binned distribution (src.physics.stats.weighted_stats on bin",
        "          centres): resolution 0.05 dex",
        "",
        f"{'d [pc]':>7}{'N cells':>11}{'f_V CNM':>9}{'UNM':>8}{'WNM':>8}{'HIM':>8}{'sum-1':>10}"
        f"{'f_m CNM':>9}{'UNM':>8}{'WNM':>8}{'sum-1':>10}",
    ]
    for dd in SLAB_D_PC:
        s = summ[dd]
        f = s["vol_frac"]
        fm = [s["mass_frac"][p] for p in PANEL_PHASES]
        L.append(f"{dd:>7.0f}{int(s['N']):>11d}{f[0]:>9.4f}{f[1]:>8.4f}{f[2]:>8.4f}{f[3]:>8.4f}"
                 f"{f.sum() - 1:>10.1e}{fm[0]:>9.4f}{fm[1]:>8.4f}{fm[2]:>8.4f}{sum(fm) - 1:>10.1e}")
    for tag, name in (("v", "VOLUME"), ("m", "MASS")):
        L += ["", f"log10 n_H [cm^-3], {name} weighted: median [p{pl:g}, p{ph:g}]",
              f"{'d [pc]':>7}" + "".join(f"{p:>24}" for p in PANEL_PHASES)]
        for dd in SLAB_D_PC:
            s = summ[dd]
            row = f"{dd:>7.0f}"
            for p in PANEL_PHASES:
                if s["n"][p] < N_MIN_CELLS:
                    row += f"{'(' + str(int(s['n'][p])) + ' cells)':>24}"
                else:
                    row += (f"{s[tag]['median'][p]:>9.3f} [{s[tag]['p_lo'][p]:>6.3f}, "
                            f"{s[tag]['p_hi'][p]:>6.3f}]")
            L.append(row)
    L += ["", f"Cells per phase (curves need >= {N_MIN_CELLS}):",
          f"{'d [pc]':>7}" + "".join(f"{p:>11}" for p in ALL_PHASES)]
    for dd in SLAB_D_PC:
        L.append(f"{dd:>7.0f}" + "".join(f"{int(summ[dd]['n'][p]):>11d}" for p in ALL_PHASES))
    L += ["", "Dropped curves: " + (", ".join(f"{p} at d = {dd:.0f} pc ({n} cells)"
                                              for p, dd, n in dropped) or "none"),
          "",
          "CHECKS (the script raises if any fails)",
          "  every slab is a union of whole 4 pc bins; summed counts + under + over == cell counts;",
          f"  summed mass histograms + under + over == the cached sum of n_H per phase (rel <= "
          f"{MASS_SUM_RTOL:.0e});",
          "  d = 0 and d = 300 equal the cache's own slab histograms (counts, cells and mass);",
          "  area under every drawn curve == its f_V (top row) or f_m (bottom row) to 1e-12.",
          f"  max |sum f_V - 1| over the slabs: "
          f"{max(abs(summ[dd]['vol_frac'].sum() - 1) for dd in SLAB_D_PC):.1e};  "
          f"max |sum f_m - 1|: "
          f"{max(abs(sum(summ[dd]['mass_frac'].values()) - 1) for dd in SLAB_D_PC):.1e}"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
