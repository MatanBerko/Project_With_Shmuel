"""
fig_raw_diagnostics -- what the HIM flag is actually doing, vs height.

Reads ONLY cache/core/raw_diagnostics.npz (written by
scripts/compute/compute_raw_diagnostics.py). No physics here, no source
cube, no src.physics import -- that is the rule for design scripts.

Four panels, signed z on a shared x axis:

  (a) the HIM-flag fraction by volume and by observed mass. The two are
      far apart, which is the headline: the flag catches roughly half the
      VOLUME at the midplane but only a few percent of the MASS.
  (b) the two quantities that actually decide the flag -- p_nT = n_H T
      (the BS19 classification convention, not the physical 1.1 n_H T)
      against 0.5 P_min(I_UV). Where the p_nT curve crosses below the
      threshold curve is where the typical cell becomes flagged, and the
      15-85 band shows how much of the population is either side.
  (c) n_H for flagged vs non-flagged cells, and
  (d) T for the same two populations, with the 300 K / 6000 K phase cuts
      drawn in as guides so the flagged population's temperature can be
      read against them.

No XY maps or slices anywhere: the f98 cube is mirrored in XY, so only
z-dependent quantities are trustworthy at this stage.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    BAND_ALPHA,
    FLAG_COLORS,
    FLAG_LABELS,
    WEIGHT_COLORS,
    WEIGHT_LABELS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_linear_ylim,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/raw_diagnostics.npz")
NAME = "fig_raw_diagnostics"
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)

PNT_COLOR = "#1a1a2e"        # p_nT, the classified quantity
THRESHOLD_COLOR = "#7b5ea7"  # 0.5 P_min, the flag threshold (HIM purple)
CNM_CUT_K = 300.0
WNM_CUT_K = 6000.0


def first_crossings(z, y, level):
    """z values where y crosses `level`, by linear interpolation, ordered
    by |z|. Used only to annotate the figure with a readable number."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    z, y = z[ok], y[ok]
    s = np.sign(y - level)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    out = []
    for i in idx:
        y0, y1 = y[i], y[i + 1]
        t = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
        out.append(z[i] + t * (z[i + 1] - z[i]))
    return sorted(out, key=abs)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    pct_lo, pct_hi = d["percentile_levels"]
    # plain "%": matplotlib draws "\%" literally (usetex is off)
    band_label = f"{pct_lo:g}–{pct_hi:g}%"
    in_range = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), sharex=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---------------------------------------------------------------- (a)
    for wt in ("vol", "mw"):
        ax_a.plot(z, d[f"him_frac_{wt}"], color=WEIGHT_COLORS[wt],
                  linestyle="-" if wt == "vol" else "--",
                  label=WEIGHT_LABELS[wt])
    guide_line(ax_a, 0.5, label="50%", label_loc="above", label_x=0.72)
    ax_a.set_ylabel("HIM-flagged fraction")
    # Headroom above 1.0 so the legend sits clear of the curves rather
    # than on top of them: both curves reach ~0.9 at the box edges, so
    # there is no empty corner inside the data range.
    ax_a.set_ylim(-0.03, 1.30)
    ax_a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.10, 0.99), ncols=2)
    panel_label(ax_a, "a")

    cross = first_crossings(z, d["him_frac_vol"], 0.5)
    if cross:
        annotate(ax_a, f"volume curve crosses 50% at\n"
                       f"$z = {cross[0]:+.0f}$ and ${cross[1]:+.0f}$ pc",
                 loc="lower right", size=6.5)

    # ---------------------------------------------------------------- (b)
    ax_b.fill_between(z, d["pnT_p15"], d["pnT_p85"], color=PNT_COLOR,
                      alpha=BAND_ALPHA, linewidth=0,
                      label=f"$p_{{nT}}$ {band_label}")
    ax_b.plot(z, d["pnT_median"], color=PNT_COLOR, linestyle="-",
              label="$p_{nT}=n_\\mathrm{H}T$ median")
    ax_b.plot(z, d["half_pmin_median"], color=THRESHOLD_COLOR, linestyle="--",
              label="$0.5\\,P_\\mathrm{min}(I_\\mathrm{UV})$ median")
    ax_b.set_yscale("log")
    ax_b.set_ylabel("$P/k_\\mathrm{B}$  [K cm$^{-3}$]")
    b_lo, b_hi = robust_log_ylim(
        [d["pnT_p15"], d["pnT_p85"], d["pnT_median"], d["half_pmin_median"]],
        z=z, z_range=Z_RANGE)
    # Extra room BELOW the band so the three-row legend sits in empty
    # space instead of on top of the shading.
    ax_b.set_ylim(b_lo / 3.5, b_hi)
    ax_b.legend(loc="lower left", ncols=1)
    panel_label(ax_b, "b")

    # ---------------------------------------------------------------- (c)
    for state in ("neutral", "flagged"):
        ax_c.fill_between(z, d[f"n_p15__{state}"], d[f"n_p85__{state}"],
                          color=FLAG_COLORS[state], alpha=BAND_ALPHA, linewidth=0)
        ax_c.plot(z, d[f"n_median__{state}"], color=FLAG_COLORS[state],
                  linestyle="-" if state == "neutral" else "--",
                  label=FLAG_LABELS[state])
    ax_c.set_yscale("log")
    ax_c.set_ylabel("$n_\\mathrm{H}$  [cm$^{-3}$]")
    ax_c.set_xlabel("$z$  [pc]")
    c_lo, c_hi = robust_log_ylim(
        [d[f"n_{k}__{s}"] for k in ("p15", "p85", "median")
         for s in ("flagged", "neutral")], z=z, z_range=Z_RANGE)
    # Extra headroom at the top: the two populations fill the panel from
    # top to bottom, so the legend goes ABOVE the data rather than over it.
    ax_c.set_ylim(c_lo, c_hi * 6.0)
    ax_c.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.52, 1.0))
    panel_label(ax_c, "c")

    # ---------------------------------------------------------------- (d)
    for state in ("neutral", "flagged"):
        ax_d.fill_between(z, d[f"T_p15__{state}"], d[f"T_p85__{state}"],
                          color=FLAG_COLORS[state], alpha=BAND_ALPHA, linewidth=0)
        ax_d.plot(z, d[f"T_median__{state}"], color=FLAG_COLORS[state],
                  linestyle="-" if state == "neutral" else "--",
                  label=FLAG_LABELS[state])
    ax_d.set_yscale("log")
    ax_d.set_ylabel("$T$  [K]")
    ax_d.set_xlabel("$z$  [pc]")
    y_lo, y_hi = robust_log_ylim(
        [d[f"T_{k}__{s}"] for k in ("p15", "p85", "median")
         for s in ("flagged", "neutral")], z=z, z_range=Z_RANGE)
    # Both phase cuts are drawn, so both must be INSIDE the axis -- the
    # robust percentile range stops around 600 K and would put the 300 K
    # guide off-panel. Widen explicitly rather than letting a guide line
    # silently vanish.
    ax_d.set_ylim(min(y_lo, CNM_CUT_K * 0.75), max(y_hi, WNM_CUT_K * 1.15))
    guide_line(ax_d, CNM_CUT_K, label="300 K (CNM cut)", label_loc="above")
    guide_line(ax_d, WNM_CUT_K, label="6000 K (WNM cut)", label_loc="below",
               label_x=0.62)
    panel_label(ax_d, "d")
    # Every corner of this panel is taken -- curves at the top, the 300 K
    # guide label at the bottom -- so place the note in the empty mid-left.
    annotate(ax_d, f"median, {band_label} band,\nvolume-weighted",
             xy=(0.03, 0.42), size=6.5)

    # ---------------------------------------------------------------- shared
    for ax in axes.ravel():
        ax.set_xlim(*Z_RANGE)
        ax.set_xticks(Z_TICKS)
    provisional_label(fig)
    save_figure(fig, NAME)

    # ---- numbers for the report, read off the same arrays that were drawn
    iz0 = int(np.argmin(np.abs(z)))
    print(f"  z-range {Z_RANGE}, {in_range.sum()} bins plotted")
    print(f"  (a) HIM frac at z~0: vol {d['him_frac_vol'][iz0]:.3f}, "
          f"mw {d['him_frac_mw'][iz0]:.3f}; "
          f"50% crossings (|z| asc): {[f'{c:+.0f}' for c in cross[:4]]}")
    print(f"  (a) ylim {ax_a.get_ylim()}")
    print(f"  (b) ylim {ax_b.get_ylim()}; p_nT/0.5Pmin at z~0: "
          f"{d['pnT_median'][iz0]:.0f} / {d['half_pmin_median'][iz0]:.0f}")
    pnT_cross = first_crossings(z, d["pnT_median"] - d["half_pmin_median"], 0.0)
    print(f"      p_nT median crosses 0.5 P_min at z = "
          f"{[f'{c:+.0f}' for c in pnT_cross[:4]]} pc")
    print(f"  (c) ylim {ax_c.get_ylim()}; n_H median at z~0: "
          f"neutral {d['n_median__neutral'][iz0]:.3f}, "
          f"flagged {d['n_median__flagged'][iz0]:.4f} cm^-3")
    print(f"  (d) ylim {ax_d.get_ylim()}; T median at z~0: "
          f"neutral {d['T_median__neutral'][iz0]:.0f}, "
          f"flagged {d['T_median__flagged'][iz0]:.0f} K")
    print(f"  frac(alpha<1) non-HIM at z~0: vol {d['frac_alpha_lt1_vol'][iz0]:.4f}, "
          f"mw {d['frac_alpha_lt1_mw'][iz0]:.4f}")


if __name__ == "__main__":
    main()
