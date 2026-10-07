"""
fig_component_profiles -- the profiles our simulation collaborator asked
for, with TIGRESS R8 overlaid.

Reads ONLY cache/core/component_profiles.npz (written by
scripts/compute/compute_component_profiles.py, which carries the R8
reference values across from results/reference/ok22_tigress_R8.csv). No
physics here.

Three stacked panels on a shared signed-z axis:
  (a) P_tot and P_th of the neutral gas -- mass-weighted mean as lines,
      median with its 15-85 band -- plus the band HIM_A and HIM_B would
      IMPOSE on the flagged cells, hatched to mark it as an assumption
      rather than a measurement.
  (b) alpha three ways, with alpha = 1 marked.
  (c) n_H, volume- and mass-weighted, over all cells and over non-HIM
      cells. The two are different physical questions -- "mean density of
      the box" vs "mean density of the neutral gas" -- and a simulator
      comparing against a two-phase average wants the second.

R8 midplane values are open markers at z = 0. R8 has about twice the
local gas surface density, so absolute pressures are not expected to
match; alpha is a ratio and can be compared. That caveat is on the figure
itself, not just in the caption, because the figure will travel alone.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    BAND_ALPHA,
    PRESSURE_COLORS,
    REFERENCE_COLOR,
    REFERENCE_LABEL,
    REFERENCE_MARKER,
    WEIGHT_COLORS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/component_profiles.npz")
NAME = "fig_component_profiles"
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
HIM_BAND_COLOR = "#7b5ea7"
ALL_LS, NEUTRAL_LS = "--", "-"


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    pct_lo, pct_hi = d["percentile_levels"]
    band = f"{pct_lo:g}–{pct_hi:g}%"

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(7.0, 7.8), sharex=True)
    # Reserve a strip at the bottom for the comparability note. fig.text
    # does not reserve space under constrained_layout, so without this the
    # note lands on top of the x-label.
    # Top edge is pulled in too: panel (a) carries a three-row legend that
    # runs to the axes top, and a rect reaching y = 1.0 clips it.
    fig.get_layout_engine().set(rect=(0.0, 0.030, 1.0, 0.962))

    # ---------------------------------------------------------------- (a)
    # the assumed HIM band first, so it sits behind the measurements
    ax_a.fill_between(z, d["him_assumed_Pth_A"], d["him_assumed_Pth_B"],
                      facecolor="none", edgecolor=HIM_BAND_COLOR, hatch="//",
                      linewidth=0.0, alpha=0.30, zorder=1)
    for key, label in (("Ptot", "$P_\\mathrm{tot}$"),
                       ("Pth_neutral", "$P_\\mathrm{th}$, neutral")):
        col = PRESSURE_COLORS["Ptot" if key == "Ptot" else "Pth"]
        ax_a.fill_between(z, d[f"{key}_mw_p15"], d[f"{key}_mw_p85"],
                          color=col, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax_a.plot(z, d[f"{key}_mw_mean"], color=col, linestyle="-", linewidth=1.6,
                  zorder=4, label=f"{label} mean")
        ax_a.plot(z, d[f"{key}_mw_median"], color=col, linestyle=":", linewidth=1.4,
                  zorder=4, label=f"{label} median")
    for q, col in (("P_tot_2p", PRESSURE_COLORS["Ptot"]),
                   ("P_th_2p", PRESSURE_COLORS["Pth"])):
        ax_a.plot([0.0], [float(d[f"ref_{q}"][0])], color=REFERENCE_COLOR,
                  **REFERENCE_MARKER)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("$P/k_\\mathrm{B}$  [K cm$^{-3}$]")
    lo, hi = robust_log_ylim(
        [d[f"{k}_mw_{s}"] for k in ("Ptot", "Pth_neutral")
         for s in ("mean", "median", "p15", "p85")]
        + [d["him_assumed_Pth_A"], d["him_assumed_Pth_B"]], z=z, z_range=Z_RANGE)
    hi = max(hi, float(d["ref_P_tot_2p"][0]) * 1.15)
    ax_a.set_ylim(lo, hi * 2.6)
    panel_label(ax_a, "a")

    handles = [
        Line2D([], [], color=PRESSURE_COLORS["Ptot"], linewidth=1.6),
        Line2D([], [], color=PRESSURE_COLORS["Pth"], linewidth=1.6),
        Line2D([], [], color="0.35", linestyle="-", linewidth=1.6),
        Line2D([], [], color="0.35", linestyle=":", linewidth=1.4),
        Patch(facecolor="0.35", alpha=BAND_ALPHA, linewidth=0),
        Patch(facecolor="none", edgecolor=HIM_BAND_COLOR, hatch="//", linewidth=0.0,
              alpha=0.6),
        Line2D([], [], color=REFERENCE_COLOR, **REFERENCE_MARKER),
    ]
    labels = ["$P_\\mathrm{tot}$", "$P_\\mathrm{th}$, neutral",
              "mass-weighted mean", "median",
              f"median {band}", "assumed HIM (HIM$_\\mathrm{A}$–HIM$_\\mathrm{B}$)",
              REFERENCE_LABEL]
    ax_a.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncols=3)

    # ---------------------------------------------------------------- (b)
    ax_b.fill_between(z, d["alpha_p15_of_ratios_mw"], d["alpha_p85_of_ratios_mw"],
                      color=WEIGHT_COLORS["mw"], alpha=BAND_ALPHA, linewidth=0, zorder=1)
    ax_b.plot(z, d["alpha_ratio_of_means_vol"], color=WEIGHT_COLORS["vol"],
              linestyle=":", linewidth=1.6, zorder=3,
              label="ratio of means, volume-weighted")
    ax_b.plot(z, d["alpha_ratio_of_means_mw"], color=WEIGHT_COLORS["mw"],
              linestyle=":", linewidth=1.6, zorder=3,
              label="ratio of means, mass-weighted")
    ax_b.plot(z, d["alpha_median_of_ratios_mw"], color=WEIGHT_COLORS["mw"],
              linestyle="-", linewidth=1.6, zorder=4,
              label=f"median of ratios, mass-weighted ({band} band)")
    ax_b.plot([0.0], [float(d["ref_alpha_2p"][0])], color=REFERENCE_COLOR,
              label=REFERENCE_LABEL, **REFERENCE_MARKER)
    guide_line(ax_b, 1.0, label="$\\alpha = 1$", label_loc="above", label_x=0.02)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("$\\alpha = P_\\mathrm{tot}/P_\\mathrm{th}$")
    lo, hi = robust_log_ylim(
        [d[k] for k in ("alpha_ratio_of_means_vol", "alpha_ratio_of_means_mw",
                        "alpha_median_of_ratios_mw", "alpha_p15_of_ratios_mw",
                        "alpha_p85_of_ratios_mw")], z=z, z_range=Z_RANGE)
    ax_b.set_ylim(min(lo, 0.9), hi * 2.2)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    panel_label(ax_b, "b")

    # ---------------------------------------------------------------- (c)
    for tag, ls, lab in (("all", ALL_LS, "all cells"),
                         ("neutral", NEUTRAL_LS, "non-HIM cells")):
        ax_c.plot(z, d[f"nH_vol_mean_{tag}"], color=WEIGHT_COLORS["vol"],
                  linestyle=ls, linewidth=1.5, zorder=3)
        ax_c.plot(z, d[f"nH_mw_mean_{tag}"], color=WEIGHT_COLORS["mw"],
                  linestyle=ls, linewidth=1.5, zorder=3)
    ax_c.plot([0.0], [float(d["ref_n_H_2p"][0])], color=REFERENCE_COLOR,
              **REFERENCE_MARKER)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("$n_\\mathrm{H}$  [cm$^{-3}$]")
    ax_c.set_xlabel("$z$  [pc]")
    lo, hi = robust_log_ylim(
        [d[f"nH_{w}_mean_{t}"] for w in ("vol", "mw") for t in ("all", "neutral")],
        z=z, z_range=Z_RANGE)
    ax_c.set_ylim(lo, hi * 4.0)
    handles = [
        Line2D([], [], color=WEIGHT_COLORS["vol"], linewidth=1.5),
        Line2D([], [], color=WEIGHT_COLORS["mw"], linewidth=1.5),
        Line2D([], [], color="0.35", linestyle=NEUTRAL_LS, linewidth=1.5),
        Line2D([], [], color="0.35", linestyle=ALL_LS, linewidth=1.5),
        Line2D([], [], color=REFERENCE_COLOR, **REFERENCE_MARKER),
    ]
    labels = ["volume-weighted mean", "mass-weighted mean",
              "non-HIM cells", "all cells", REFERENCE_LABEL]
    ax_c.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    panel_label(ax_c, "c")

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(*Z_RANGE)
        ax.set_xticks(Z_TICKS)

    # The caveat travels with the figure, not only with the caption.
    fig.text(0.5, 0.008,
             f"TIGRESS R8 has $\\Sigma_\\mathrm{{gas}} \\approx "
             f"{float(d['ref_Sigma_gas'][0]):.1f}$ M$_\\odot$ pc$^{{-2}}$, about twice the "
             f"local value — absolute pressures are not expected to match; $\\alpha$ can.",
             ha="center", va="bottom", fontsize=6.8, color="0.3")
    # Upper left: the bottom strip is taken by the comparability note,
    # which runs almost the full width.
    provisional_label(fig, loc="upper left")
    save_figure(fig, NAME)

    # ---- numbers for the report
    def at(zt):
        return int(np.argmin(np.abs(z - zt)))

    print(f"  (a) y-range {ax_a.get_ylim()[0]:.4g} to {ax_a.get_ylim()[1]:.4g}")
    print(f"  (b) y-range {ax_b.get_ylim()[0]:.4g} to {ax_b.get_ylim()[1]:.4g}")
    print(f"  (c) y-range {ax_c.get_ylim()[0]:.4g} to {ax_c.get_ylim()[1]:.4g}")
    for zt in (-300, -150, 0, 150, 300):
        i = at(zt)
        print(f"  z={z[i]:>+5.0f}: Pth_n {d['Pth_neutral_mw_mean'][i]:>8.1f}  "
              f"Ptot {d['Ptot_mw_mean'][i]:>9.1f}  "
              f"a_RoM_vol {d['alpha_ratio_of_means_vol'][i]:>6.3f}  "
              f"a_RoM_mw {d['alpha_ratio_of_means_mw'][i]:>6.3f}  "
              f"a_med_mw {d['alpha_median_of_ratios_mw'][i]:>6.3f}  "
              f"nH_vol_all {d['nH_vol_mean_all'][i]:>7.4f}  "
              f"nH_mw_all {d['nH_mw_mean_all'][i]:>7.4f}  "
              f"nH_vol_neu {d['nH_vol_mean_neutral'][i]:>7.4f}")
    print(f"  R8: P_tot {float(d['ref_P_tot_2p'][0]):.4g}, P_th {float(d['ref_P_th_2p'][0]):.4g}, "
          f"alpha {float(d['ref_alpha_2p'][0]):.3g}, n_H {float(d['ref_n_H_2p'][0]):.3g}")


if __name__ == "__main__":
    main()
