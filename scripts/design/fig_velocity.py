"""
fig_velocity -- Fig. 7: the missing support expressed as a velocity.

Reads ONLY cache/velocity/velocity.npz (scripts/compute/compute_velocity.py).
RAW, self-gravity mean (fiducial), mass-weighted, whole footprint, every phase
under the same W of its height bin. Our data only: the literature values are
quoted in the caption, not drawn.

Three stacked panels on a shared signed-z axis, lines for CNM, WNM (solid,
phase colours) and all neutral gas (thin black), with the mass-weighted
15-85% band of the per-cell quantity for CNM and WNM behind them:
  (a) sigma_eff = sqrt(k_B W / <rho>_k), log y
  (b) c_s = sigma_th = sqrt(sum P_th / sum rho), log y, SAME range as (a)
  (c) <M^2> = mass-weighted <W/P_th - 1>, linear y, M^2 = 0 guide

Each phase is drawn (line and band) only over the contiguous run of bins
from z_mid outward in which it has >= N_MIN_CELLS cells and >=
MIN_VOLUME_FRACTION of the bin volume (the fig_phase_alpha rule, walked
from z_mid). UNM and sigma_eff for all gas are in the npz but not drawn.
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

from src.conventions import CNM, WNM  # noqa: E402
from src.plotting.style import (  # noqa: E402
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_linear_ylim,
    robust_log_ylim,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/velocity/velocity.npz")
NAME = "fig_velocity"
FIG_W, FIG_H = 5.0, 6.2
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
MIN_VOLUME_FRACTION = 0.01     # as fig_phase_alpha
PHASE_COLORS = {"CNM": CNM, "WNM": WNM}
BAND_PHASES = ("CNM", "WNM")
BAND_ALPHA = {"CNM": 0.22, "WNM": 0.18}
BAND_LEGEND_COLOR = "0.6"
LINE_LW = 1.4
NEUTRAL_KW = dict(color="black", lw=0.8)
ZMID_KW = dict(color="0.6", lw=0.8, ls=":", zorder=2)   # as fig_alpha_z
ZMID_LABEL = r"$z_\mathrm{mid}$"
TEXT_FS = 6.5
LOG_PAD_DEX = 0.06
# per panel: (mean key, band key or None, y label)
PANELS = (
    ("sigma_eff", "sigma_eff_cell", r"$\sigma_\mathrm{eff}$ [km s$^{-1}$]"),
    ("sigma_th", "cs_cell", r"$c_\mathrm{s}$ [km s$^{-1}$]"),
    ("M2", "M2_cell", r"$\langle\mathcal{M}^2\rangle$"),
)
LEGEND_PANEL = 1               # index into PANELS: (b) is empty below the WNM line right of the CNM run
LEGEND_LOC = "lower right"
LEGEND_ANCHOR = (1.0, 0.30)    # axes fraction: above the CNM band, below the neutral line


def contiguous_from(z, usable, z0):
    """Mask of bins reachable from z0 outward through usable bins only."""
    keep = np.zeros_like(usable, dtype=bool)
    for side in (np.where(z > z0)[0], np.where(z < z0)[0][::-1]):
        for i in side:
            if not usable[i]:
                break
            keep[i] = True
    return keep


def drawn_run(d, z, z_mid, k):
    n_min = int(d["n_min_cells"][0])
    usable = (d[f"{k}__n_cells"] >= n_min) & (d[f"{k}__vol_frac"] >= MIN_VOLUME_FRACTION)
    return contiguous_from(z, usable, z_mid)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    z_mid = float(d["z_mid_pc"][0])
    keep = {k: drawn_run(d, z, z_mid, k) for k in ("CNM", "WNM", "NEUTRAL")}
    m = lambda k, key: np.where(keep[k], d[f"{k}__{key}"], np.nan)  # noqa: E731

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), sharex=True)
    drawn = []
    for ax, (key, band, ylabel) in zip(axes, PANELS):
        vals = []
        for k in BAND_PHASES:
            lo, hi = m(k, f"{band}_p_lo"), m(k, f"{band}_p_hi")
            ax.fill_between(z, lo, hi, where=np.isfinite(lo) & np.isfinite(hi),
                            color=PHASE_COLORS[k], alpha=BAND_ALPHA[k], lw=0, zorder=1)
            vals += [lo, hi]
        for k in BAND_PHASES:
            ax.plot(z, m(k, key), color=PHASE_COLORS[k], lw=LINE_LW, zorder=4)
            vals.append(m(k, key))
        ax.plot(z, m("NEUTRAL", key), zorder=5, **NEUTRAL_KW)
        vals.append(m("NEUTRAL", key))
        drawn.append(vals)
        ax.set_ylabel(ylabel)
        ax.axvline(z_mid, **ZMID_KW)

    ax_a, ax_b, ax_c = axes
    # (a) and (b): one shared log range over everything drawn in both
    lo, hi = robust_log_ylim([np.concatenate(drawn[0] + drawn[1])], lo_pct=0.0, hi_pct=100.0,
                             pad_dex=LOG_PAD_DEX)
    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
    lo_c, hi_c = robust_linear_ylim([np.concatenate(drawn[2])], lo_pct=0.0, hi_pct=100.0,
                                    include_zero=True)
    ax_c.set_ylim(lo_c, hi_c)
    guide_line(ax_c, 0.0)

    ax_a.text(z_mid + 6, 0.97, ZMID_LABEL, transform=ax_a.get_xaxis_transform(),
              ha="left", va="top", fontsize=TEXT_FS, color="0.35", zorder=6)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, loc="upper left")

    handles = [Line2D([], [], color=CNM, lw=LINE_LW), Line2D([], [], color=WNM, lw=LINE_LW),
               Line2D([], [], **NEUTRAL_KW),
               Patch(facecolor=BAND_LEGEND_COLOR, alpha=0.4, lw=0)]
    axes[LEGEND_PANEL].legend(handles, ["CNM", "WNM", "all neutral", "15–85%, mass-weighted"],
                              loc=LEGEND_LOC, bbox_to_anchor=LEGEND_ANCHOR, ncols=2, frameon=False, fontsize=TEXT_FS,
                              handlelength=1.8, columnspacing=1.2)

    ax_c.set_xlim(*Z_RANGE)
    ax_c.set_xticks(Z_TICKS)
    ax_c.set_xlabel(r"$z$ [pc]")
    provisional_label(fig, loc="lower right")
    print(f"  (a),(b) y {lo:.3f} .. {hi:.3f} km/s; (c) y {lo_c:.3f} .. {hi_c:.3f}")
    # where a mean line leaves its own band (inside the drawn run)
    for key, band, _ in PANELS:
        for k in BAND_PHASES:
            v, plo, phi = m(k, key), m(k, f"{band}_p_lo"), m(k, f"{band}_p_hi")
            ok = np.isfinite(v)
            out = ok & ((v < plo) | (v > phi))
            if out.any():
                print(f"  {k} {key}: outside its band in {int(out.sum())}/{int(ok.sum())} bins, "
                      f"z = {z[out].min():+.0f}..{z[out].max():+.0f}")
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
