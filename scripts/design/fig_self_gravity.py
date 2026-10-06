"""
fig_self_gravity -- how much gas self-gravity moves alpha, RAW only.

Reads ONLY cache/core/summary.npz. No physics here.

Two stacked panels sharing the signed-z axis, volume-weighted RAW:

  (a) alpha from two estimators, with self-gravity OFF and with the
      fiducial footprint-averaged MEAN. Colour encodes the estimator and
      linestyle encodes the self-gravity setting -- the opposite
      assignment to fig_alpha_profiles, because here there is only one
      variant so colour is free, and because the off/on pair is the
      comparison the figure exists to make: putting it on linestyle keeps
      each pair the same colour and directly comparable.
  (b) the ratio mean/off for the same two estimators, which is the same
      comparison made quantitative and is flat enough to read a number
      off.

Only the median_of_ratios carries a band: it is the only estimator with a
matching pair of percentiles of the same quantity (p15/p85 OF THE
RATIOS). Both self-gravity settings get one, lightly, since they are
nested and the point is how far the whole distribution shifts.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.plotting.style import (  # noqa: E402
    ESTIMATOR_COLORS,
    ESTIMATOR_LABELS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_linear_ylim,
    robust_log_ylim,
    save_figure,
)

CACHE = Path("cache/core/summary.npz")
NAME = "fig_self_gravity"

VARIANT = "RAW"
WEIGHTING = "vol"
PROFILE = "signedz"
ESTIMATORS = ("alpha_median_of_ratios", "alpha_ratio_of_means")
# linestyle encodes the self-gravity setting
SG_STYLES = {"mean": dict(linestyle="-", linewidth=1.5),
             "off": dict(linestyle="--", linewidth=1.2)}
SG_LABELS = {"mean": "self-gravity: mean (fiducial)", "off": "self-gravity: off"}
BAND_ALPHA_PAIR = 0.13
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)


def key(sg, field):
    return f"{VARIANT}__sg{sg}__profile_{PROFILE}__{field}"


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d[key("mean", "z_pc")]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(5.0, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.0]))

    # ---------------------------------------------------------------- (a)
    pooled = []
    # Only the FIDUCIAL band is filled. Both settings have a 15-85 band
    # and they are nested, so drawing both produced a muddy double-grey
    # that hid the lines the panel exists to compare; the off/mean shift
    # is what panel (b) measures anyway.
    lo = d[key("mean", f"alpha_p15_of_ratios_{WEIGHTING}")]
    hi = d[key("mean", f"alpha_p85_of_ratios_{WEIGHTING}")]
    ax_a.fill_between(z, lo, hi,
                      color=ESTIMATOR_COLORS["alpha_median_of_ratios"],
                      alpha=BAND_ALPHA_PAIR, linewidth=0, zorder=1)
    pooled += [lo, hi]
    for sg in ("off", "mean"):
        for est in ESTIMATORS:
            y = d[key(sg, f"{est}_{WEIGHTING}")]
            ax_a.plot(z, y, color=ESTIMATOR_COLORS[est], **SG_STYLES[sg])
            pooled.append(y)

    guide_line(ax_a, 1.0, label="$\\alpha = 1$", label_loc="above")
    ax_a.set_yscale("log")
    ax_a.set_ylabel("$\\alpha = P_\\mathrm{tot}/P_\\mathrm{th}$")
    y_lo, y_hi = robust_log_ylim(pooled, z=z, z_range=Z_RANGE)
    y_lo, y_hi = min(y_lo, 0.95), max(y_hi, 1.05)
    # Headroom at the top for the two-block legend.
    ax_a.set_ylim(y_lo, y_hi * 2.4)
    panel_label(ax_a, "a")
    annotate(ax_a, f"{VARIANT}, volume-weighted\nband: 15–85% of ratios, "
                   f"self-gravity mean",
             xy=(0.97, 0.10), size=6.5, color="0.35")

    # Two legend blocks: colour = estimator, linestyle = self-gravity.
    # Split so neither encoding has to be inferred from the other.
    est_handles = [Line2D([], [], color=ESTIMATOR_COLORS[e], linestyle="-",
                          linewidth=1.5) for e in ESTIMATORS]
    sg_handles = [Line2D([], [], color="0.35", **SG_STYLES[sg])
                  for sg in ("mean", "off")]
    # Starts right of the "(a)" panel label so the legend title does not
    # land on top of it.
    leg1 = ax_a.legend(est_handles, [ESTIMATOR_LABELS[e] for e in ESTIMATORS],
                       loc="upper left", bbox_to_anchor=(0.13, 1.0),
                       title="colour", alignment="left")
    leg1.get_title().set_fontsize(6.5)
    ax_a.add_artist(leg1)
    leg2 = ax_a.legend(sg_handles, [SG_LABELS[sg] for sg in ("mean", "off")],
                       loc="upper right", bbox_to_anchor=(0.995, 1.0),
                       title="linestyle", alignment="left")
    leg2.get_title().set_fontsize(6.5)

    # ---------------------------------------------------------------- (b)
    ratios = []
    for est in ESTIMATORS:
        num = d[key("mean", f"{est}_{WEIGHTING}")]
        den = d[key("off", f"{est}_{WEIGHTING}")]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        ax_b.plot(z, r, color=ESTIMATOR_COLORS[est], linestyle="-",
                  linewidth=1.4)
        ratios.append(r)

    guide_line(ax_b, 1.0)
    ax_b.set_ylabel("$\\alpha_\\mathrm{mean}/\\alpha_\\mathrm{off}$")
    ax_b.set_xlabel("$z$  [pc]")
    ax_b.set_ylim(*robust_linear_ylim(ratios, z=z, z_range=Z_RANGE,
                                        lo_pct=0.0, hi_pct=100.0))
    panel_label(ax_b, "b")

    for ax in (ax_a, ax_b):
        ax.set_xlim(*Z_RANGE)
        ax.set_xticks(Z_TICKS)
    provisional_label(fig)
    save_figure(fig, NAME)

    # ---- numbers for the report
    iz0 = int(np.argmin(np.abs(z)))
    print(f"  (a) y-range {ax_a.get_ylim()[0]:.4g} to {ax_a.get_ylim()[1]:.4g} (log)")
    print(f"  (b) y-range {ax_b.get_ylim()[0]:.4g} to {ax_b.get_ylim()[1]:.4g} (linear)")
    for est, r in zip(ESTIMATORS, ratios):
        a_off = d[key("off", f"{est}_{WEIGHTING}")][iz0]
        a_mean = d[key("mean", f"{est}_{WEIGHTING}")][iz0]
        print(f"  {est:<24} at z~0: off {a_off:.3f} -> mean {a_mean:.3f} "
              f"(x{r[iz0]:.4f}); over the box x{np.nanmedian(r):.4f} "
              f"[{np.nanmin(r):.4f}, {np.nanmax(r):.4f}]")


if __name__ == "__main__":
    main()
