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

Normalisation: counts / (N_slab x Delta log10 n), N_slab = every cell of the
slab (HIM included), so the area under a phase's curve is its VOLUME
fraction in the slab. HIM is not drawn (its density is not physical); its
volume fraction is printed in the WNM panel.

Three panels CNM | UNM | WNM, shared log y and log10 n_H x. Within a panel
the four slabs are shades of the phase colour, darkest = midplane. A curve
is dropped when the phase has fewer than N_MIN_CELLS cells in the slab.

Also writes results/phase_pdfs.txt (volume and mass fractions, median and
16-84% of log10 n_H per phase and slab).
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
FIG_W, FIG_H = 6.6, 2.7
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
    """Summed histogram of a set of bins: (counts (4, nb), under (4,), over (4,), n_cells (4,))."""
    b = np.asarray(bins, int)
    return (d["hist_log_n"][b].sum(axis=0), d["hist_log_n_under"][b].sum(axis=0),
            d["hist_log_n_over"][b].sum(axis=0), d["hist_n_cells"][b].sum(axis=0))


def normalised(counts, n_total, edges):
    """counts / (n_total * bin width in dex): area = volume fraction."""
    return counts / (n_total * np.diff(edges))


def slab_summary(counts, under, over, n_cells, edges):
    """Volume and mass fractions and log10 n_H percentiles of one slab."""
    n_total = float(n_cells.sum())
    centres = 0.5 * (edges[:-1] + edges[1:])
    mass = counts @ 10.0 ** centres                     # sum n_H at bin centres, per phase
    neutral = [ALL_PHASES.index(p) for p in PANEL_PHASES]
    out = {"N": n_total, "vol_frac": n_cells / n_total,
           "mass_frac": {p: mass[ALL_PHASES.index(p)] / mass[neutral].sum() for p in PANEL_PHASES},
           "area": {}, "median": {}, "p_lo": {}, "p_hi": {}, "n": {}}
    for j, p in enumerate(ALL_PHASES):
        out["n"][p] = float(n_cells[j])
        out["area"][p] = float(np.sum(normalised(counts[j], n_total, edges) * np.diff(edges)))
        if p == "HIM" or counts[j].sum() == 0:
            out["median"][p] = out["p_lo"][p] = out["p_hi"][p] = np.nan
            continue
        st = weighted_stats(centres, counts[j], PCT_SCHEME)
        out["median"][p], out["p_lo"][p], out["p_hi"][p] = st.median, st.p_lo, st.p_hi
    return out


def load_slabs():
    """The cache, its edges and {d: (counts, under, over, n_cells, bins)}, with all checks."""
    import pipeline.compute_all as pipe
    from src.physics.loading import open_zarr

    d = np.load(CACHE, allow_pickle=False)
    if tuple(d["hist_phases"]) != ALL_PHASES:
        raise RuntimeError(f"unexpected histogram phases {tuple(d['hist_phases'])}")
    z_mid = float(d["z_mid_pc"][0])
    _, z_box = pipe._box_z_slice(np.asarray(open_zarr().z_pc, dtype=np.float64))
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    if not np.array_equal([c for c, _ in bin_groups], d["z_pc"]):
        raise RuntimeError("profile bins differ from the cache's")
    edges = d["log_n_edges"]
    slabs = {}
    for dd, bins in slab_bin_sets(z_box, bin_groups, z_mid).items():
        counts, under, over, n_cells = slab_histogram(d, bins)
        if not np.array_equal(counts.sum(axis=1) + under + over, n_cells):
            raise RuntimeError(f"d = {dd}: histogram counts differ from the cell counts")
        slabs[dd] = (counts, under, over, n_cells, bins)
    # the cached slabs at z_mid + {0, -300, +300}
    off = list(d["slab_offsets_pc"])
    cached = {0.0: d["slab__hist_n_cells"][off.index(0.0)],
              300.0: d["slab__hist_n_cells"][off.index(-300.0)]
              + d["slab__hist_n_cells"][off.index(300.0)]}
    cached_h = {0.0: d["slab__hist_log_n"][off.index(0.0)],
                300.0: d["slab__hist_log_n"][off.index(-300.0)]
                + d["slab__hist_log_n"][off.index(300.0)]}
    for dd in cached:
        if not (np.array_equal(slabs[dd][3], cached[dd])
                and np.array_equal(slabs[dd][0], cached_h[dd])):
            raise RuntimeError(f"d = {dd}: summed bins differ from the cached slab histogram")
    return d, edges, slabs


def main():
    apply_paper_style()
    d, edges, slabs = load_slabs()
    centres = 0.5 * (edges[:-1] + edges[1:])
    summ = {dd: slab_summary(*slabs[dd][:4], edges) for dd in SLAB_D_PC}
    colors = {p: [shade(BASE[p], *s) for s in SHADES[p]] for p in PANEL_PHASES}
    for p in PANEL_PHASES:
        cr = contrast_vs_white(colors[p][-1])
        print(f"  {p} lightest shade contrast vs white: {cr:.2f}"
              + ("  (WARNING: below MIN_CONTRAST)" if cr < MIN_CONTRAST else ""))

    # curves and their ranges
    curves, dropped = {}, []
    for p in PANEL_PHASES:
        j = ALL_PHASES.index(p)
        for dd in SLAB_D_PC:
            counts, _, _, n_cells, _ = slabs[dd]
            if n_cells[j] < N_MIN_CELLS:
                dropped.append((p, dd, int(n_cells[j])))
                continue
            curves[(p, dd)] = normalised(counts[j], n_cells.sum(), edges)
    pos = np.concatenate([c[c > 0] for c in curves.values()])
    nz = np.concatenate([centres[c > 0] for c in curves.values()])
    hw = 0.5 * np.diff(edges)[0]
    xlim = (nz.min() - hw - X_MARGIN_DEX, nz.max() + hw + X_MARGIN_DEX)
    ylim = (pos.min() / 10 ** Y_MARGIN_DEX, pos.max() * 10 ** Y_TOP_MARGIN_DEX)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
    for ax, p, letter in zip(axes, PANEL_PHASES, "abc"):
        for i, dd in enumerate(SLAB_D_PC):
            if (p, dd) not in curves:
                continue
            ax.stairs(curves[(p, dd)], edges, color=colors[p][i], lw=LINE_LW,
                      zorder=5 - i, baseline=None)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\log_{10}(n_\mathrm{H}\,/\,\mathrm{cm^{-3}})$")
        ax.text(0.96, 0.96, p, transform=ax.transAxes, ha="right", va="top",
                fontsize=TEXT_FS + 1.5, color=colors[p][1], fontweight="bold")
        panel_label(ax, letter, loc="upper left")
    axes[0].set_ylabel(r"volume fraction per dex")
    him = "\n".join([r"HIM $f_V$:"] + [f"{dd:.0f} pc: {summ[dd]['vol_frac'][3]:.3f}"
                                         for dd in SLAB_D_PC])
    axes[2].text(0.96, 0.62, him, transform=axes[2].transAxes, ha="right", va="top",   # empty high-n side
                 fontsize=TEXT_FS - 0.5, color="0.25", linespacing=1.15)
    handles = [Line2D([], [], color=g, lw=LINE_LW) for g in LEGEND_GREYS]
    labels = [rf"$|z - z_\mathrm{{mid}}| = {dd:.0f}$ pc" for dd in SLAB_D_PC]
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False,
               fontsize=TEXT_FS, handlelength=2.0, columnspacing=1.4,
               title="darker = closer to the midplane", title_fontsize=TEXT_FS)
    provisional_label(fig, loc="upper right")
    print(f"  x {xlim[0]:.2f} .. {xlim[1]:.2f} dex; y {ylim[0]:.2e} .. {ylim[1]:.2e}")
    print("  dropped curves: " + (", ".join(f"{p} d={dd:.0f} ({n} cells)" for p, dd, n in dropped)
                                  or "none"))
    # area under each drawn curve == the phase's volume fraction
    for (p, dd), c in curves.items():
        a = float(np.sum(c * np.diff(edges)))
        if abs(a - summ[dd]["vol_frac"][ALL_PHASES.index(p)]) > 1e-12:
            raise RuntimeError(f"{p} d={dd}: area {a} differs from the volume fraction")
    save_figure(fig, NAME)
    write_report(d, summ, slabs, dropped)


def write_report(d, summ, slabs, dropped):
    pl, ph = percentile_levels(PCT_SCHEME)
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Fig. 5: phase density distributions vs height",
        "=" * 104,
        f"RAW, footprint |x|,|y| <= 500 pc, balanced midplane z_mid = "
        f"{float(d['z_mid_pc'][0]):+.3f} pc. Phases: cached dP/dn flag (CNM, UNM, WNM, HIM),",
        "as Figs. 6-7. Source: the per-4-pc-bin log10 n_H histograms in "
        "cache/turbulence_pdf/turbulence_pdf.npz",
        f"(edges {d['log_n_edges'][0]:g} .. {d['log_n_edges'][-1]:g} in "
        f"{np.diff(d['log_n_edges'])[0]:.2f} dex; under/overflow kept in the counts).",
        "Slabs: |z - z_c| <= 30 pc; d = 0 is z_c = z_mid, d = 100/200/300 the union of z_c = "
        "z_mid - d and z_mid + d.",
        "Each slab is an exact union of fifteen (d = 0) or thirty 4 pc bins (asserted).",
        "",
        "  f_V     volume (cell) fraction of the slab, HIM included in the total",
        "  f_m     fraction of the NEUTRAL (CNM+UNM+WNM) mass, mass = sum over histogram bins of",
        "          counts x 10^(bin centre): an approximation, each cell's n_H is taken at its",
        "          bin centre, i.e. good to +-0.025 dex (+-6%) per cell;",
        "          HIM is excluded (its density is not physical)",
        f"  log n   median and p{pl:g}-p{ph:g} of log10(n_H/cm^-3), volume weighted, nearest rank on",
        "          the binned distribution (src.physics.stats.weighted_stats on bin centres):",
        "          resolution 0.05 dex",
        "",
        f"{'d [pc]':>7}{'N cells':>11}{'f_V CNM':>9}{'UNM':>8}{'WNM':>8}{'HIM':>8}{'sum-1':>10}"
        f"{'f_m CNM':>9}{'UNM':>8}{'WNM':>8}",
    ]
    for dd in SLAB_D_PC:
        s = summ[dd]
        f = s["vol_frac"]
        L.append(f"{dd:>7.0f}{int(s['N']):>11d}{f[0]:>9.4f}{f[1]:>8.4f}{f[2]:>8.4f}{f[3]:>8.4f}"
                 f"{f.sum() - 1:>10.1e}" + "".join(f"{s['mass_frac'][p]:>8.4f}" if i else
                                                  f"{s['mass_frac'][p]:>9.4f}"
                                                  for i, p in enumerate(PANEL_PHASES)))
    L += ["", f"log10 n_H [cm^-3]: median [p{pl:g}, p{ph:g}]",
          f"{'d [pc]':>7}" + "".join(f"{p:>24}" for p in PANEL_PHASES)]
    for dd in SLAB_D_PC:
        s = summ[dd]
        row = f"{dd:>7.0f}"
        for p in PANEL_PHASES:
            if s["n"][p] < N_MIN_CELLS:
                row += f"{'(' + str(int(s['n'][p])) + ' cells)':>24}"
            else:
                row += f"{s['median'][p]:>9.3f} [{s['p_lo'][p]:>6.3f}, {s['p_hi'][p]:>6.3f}]"
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
          "  d = 0 and d = 300 equal the cache's own slab histograms (slab__hist_log_n, z_mid + 0, +-300);",
          "  area under every drawn curve == the phase's f_V to 1e-12.",
          f"  max |sum f_V - 1| over the slabs: "
          f"{max(abs(summ[dd]['vol_frac'].sum() - 1) for dd in SLAB_D_PC):.1e}"]
    OUT_TXT_PATH.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
