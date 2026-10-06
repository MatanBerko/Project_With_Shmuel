"""
fig_posterior_snr -- is the flagged gas's density measured or prior?

Reads ONLY cache/validation/posterior_snr.npz (written by
scripts/validation/posterior_snr.py). No physics here, no dustmaps query.

SNR = posterior mean / posterior standard deviation of the Edenhofer+23
differential extinction, per voxel. SNR < 1 means the posterior standard
deviation is larger than the value itself -- the density there is not
individually constrained by the data.

  (a) median SNR with the 15-85 band, for HIM-flagged and non-flagged
      cells separately, against |z|. Guides at SNR = 1 and 2.
  (b) the fraction of MASS below SNR = 1 in each class -- and below
      SNR = 2 as well. Mass rather than volume because the question is
      how much of the gas the analysis actually weighs is unconstrained.

      The SNR < 1 pair is the quantity this panel was specified to show;
      on this data it is ~0 everywhere (nothing is that badly measured),
      so on its own the panel would be two flat lines on the axis and
      would convey nothing. The SNR < 2 pair is drawn alongside it
      because that is where the actual structure is. Both are labelled;
      neither replaces the other.

|z| rather than signed z: the sample is stratified in |z| bins, and
folding doubles the cells per bin.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    BAND_ALPHA,
    FLAG_COLORS,
    FLAG_LABELS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_linear_ylim,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/validation/posterior_snr.npz")
NAME = "fig_posterior_snr"
CLASSES = (("neutral", "-"), ("flagged", "--"))
Z_TICKS = (0, 100, 200, 300, 400)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["abs_z_pc"]
    z_range = (0.0, float(d["bin_edges"][-1]))

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(5.0, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.45, 1.0]))

    # ---------------------------------------------------------------- (a)
    pooled = []
    for name, ls in CLASSES:
        lo = d[f"snr_p15__{name}"]
        hi = d[f"snr_p85__{name}"]
        med = d[f"snr_median__{name}"]
        ax_a.fill_between(z, lo, hi, color=FLAG_COLORS[name], alpha=BAND_ALPHA,
                          linewidth=0, zorder=1)
        ax_a.plot(z, med, color=FLAG_COLORS[name], linestyle=ls,
                  label=FLAG_LABELS[name], linewidth=1.5 if ls == "-" else 1.3)
        pooled += [lo, hi, med]

    y_lo, y_hi = robust_log_ylim(pooled, z=z, z_range=z_range)
    # Both guides are drawn, so both must be inside the axis.
    y_lo, y_hi = min(y_lo, 0.85), max(y_hi, 2.4)
    ax_a.set_yscale("log")
    # Modest headroom for the legend only: the curves span barely half a
    # decade, so a generous factor here flattens them into a line.
    ax_a.set_ylim(y_lo, y_hi * 1.35)
    guide_line(ax_a, 1.0, label="SNR = 1", label_loc="above", label_x=0.42)
    guide_line(ax_a, 2.0, label="SNR = 2", label_loc="above", label_x=0.42)
    ax_a.set_ylabel("posterior SNR\n(mean / std)")
    ax_a.legend(loc="upper right", bbox_to_anchor=(0.995, 1.0), ncols=2)
    panel_label(ax_a, "a")
    annotate(ax_a, "median, 15–85% band", xy=(0.135, 0.95), size=6.5, color="0.35")
    # Plain tick labels: over barely half a decade the default
    # "2 x 10^0" mathtext labels are harder to read than the numbers.
    ax_a.set_yticks([1.0, 1.5, 2.0, 3.0, 4.0])
    ax_a.yaxis.set_major_formatter(ScalarFormatter())
    ax_a.yaxis.set_minor_formatter(NullFormatter())

    # ---------------------------------------------------------------- (b)
    fracs = []
    for name, ls in CLASSES:
        for thr, alpha_line, lw in ((2.0, 1.0, 1.5), (1.0, 1.0, 1.0)):
            y = d[f"frac_snr_lt{thr:g}_mw__{name}"]
            ax_b.plot(z, y, color=FLAG_COLORS[name], linestyle=ls,
                      linewidth=lw if ls == "-" else lw * 0.9,
                      alpha=alpha_line,
                      label=f"{FLAG_LABELS[name]}, SNR $<$ {thr:g}")
            fracs.append(y)
    ax_b.set_ylabel("mass fraction\nbelow threshold")
    ax_b.set_xlabel("$|z|$  [pc]")
    lo, hi = robust_linear_ylim(fracs, z=z, z_range=z_range,
                                  lo_pct=0.0, hi_pct=100.0, include_zero=True)
    # Headroom for the four-entry legend.
    ax_b.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.995, 1.0), ncols=2)
    panel_label(ax_b, "b")

    box_mw = float(d["box_frac_snr_lt1_mw"][0])
    box_mw2 = float(d["box_frac_snr_lt2_mw"][0])
    box_det = float(d["box_frac_detected_mw"][0])
    annotate(ax_b, f"whole box, by mass: {box_mw:.2%} below SNR 1,\n"
                   f"{box_mw2:.1%} below SNR 2, {box_det:.1%} clearly detected\n"
                   f"(mean $-$ 2 std $>$ 0)",
             xy=(0.03, 0.14), size=6.5, color="0.35")

    for ax in (ax_a, ax_b):
        ax.set_xlim(*z_range)
        ax.set_xticks(Z_TICKS)
    provisional_label(fig)
    save_figure(fig, NAME)

    # ---- numbers for the report
    print(f"  (a) y-range {ax_a.get_ylim()[0]:.4g} to {ax_a.get_ylim()[1]:.4g} (log)")
    print(f"  (b) y-range {ax_b.get_ylim()[0]:.4g} to {ax_b.get_ylim()[1]:.4g} (linear)")
    for name, _ in CLASSES:
        m = d[f"snr_median__{name}"]
        print(f"  {name:<8} median SNR: {m[0]:.2f} at |z|~{z[0]:.0f} pc -> "
              f"{m[-1]:.2f} at |z|~{z[-1]:.0f} pc; "
              f"mass frac SNR<1 {d[f'frac_snr_lt1_mw__{name}'][0]:.3f} -> "
              f"{d[f'frac_snr_lt1_mw__{name}'][-1]:.3f}")
    print(f"  box: mass frac SNR<1 {box_mw:.4f}, SNR<2 {box_mw2:.4f}, "
          f"detected {box_det:.4f}; vol frac SNR<1 "
          f"{float(d['box_frac_snr_lt1_vol'][0]):.4f}, detected "
          f"{float(d['box_frac_detected_vol'][0]):.4f}")
    for name, _ in CLASSES:
        print(f"  {name:<8} detected (mass): {d[f'frac_detected_mw__{name}'][0]:.3f} at "
              f"|z|~{z[0]:.0f} -> {d[f'frac_detected_mw__{name}'][-1]:.3f} at |z|~{z[-1]:.0f} pc")
    print(f"  sanity: median n_H/A' {float(d['sanity_median_ratio'][0]):.1f}, "
          f"log-log r {float(d['sanity_log_r'][0]):.4f}")


if __name__ == "__main__":
    main()
