"""
fig_velocity -- Fig. 7: the missing support expressed as a velocity.

Reads ONLY cache/velocity/velocity.npz (scripts/compute/compute_velocity.py).
RAW, self-gravity mean (fiducial), mass-weighted, whole footprint, every phase
under the same W of its height bin.

Two stacked panels on a shared signed-z axis:
  (a) sigma [km/s], log y: sigma_eff for all gas (thick black), sigma_eff for
      CNM and WNM (solid) and their mass-weighted thermal sigma_th (dashed);
      faint bands for the observed WNM total and thermal line-of-sight
      dispersions of Marchal & Miville-Deschenes (2021).
  (b) <M^2> = mass-weighted <alpha_cell - 1>, linear y, for CNM, WNM (solid)
      and all neutral gas (thin black); M^2 = 0 guide; Heiles & Troland
      (2003) CNM as a dotted line, MM21 WNM as a faint band, both converted
      to our 1D isothermal convention in the compute script.

Each phase is drawn only over the contiguous run of bins from z_mid outward
in which it has >= N_MIN_CELLS cells and >= MIN_VOLUME_FRACTION of the bin
volume (the fig_phase_alpha rule, walked from z_mid rather than z = 0). UNM
is not plotted.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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
FIG_W, FIG_H = 5.0, 4.6
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
MIN_VOLUME_FRACTION = 0.01     # as fig_phase_alpha
PHASE_COLORS = {"CNM": CNM, "WNM": WNM}
PLOT_PHASES = ("CNM", "WNM")
LINE_LW = 1.3
ALL_KW = dict(color="black", lw=2.2, zorder=5)
NEUTRAL_KW = dict(color="black", lw=0.8, zorder=4)
TH_LS = (0, (4.0, 2.0))
BAND_ALPHA = {"total": 0.13, "thermal": 0.13}
BAND_COLOR = {"total": "0.45", "thermal": WNM}
REF_BAND_ALPHA = 0.16
HT03_KW = dict(color=CNM, lw=1.1, ls=":", zorder=3)
ZMID_KW = dict(color="0.6", lw=0.8, ls=":", zorder=2)   # as fig_alpha_z
ZMID_LABEL = r"$z_\mathrm{mid}$"
TEXT_FS = 6.0
LOG_HEADROOM = 1.6                     # room above the all-gas curve for the panel letter


def contiguous_from(z, usable, z0):
    """Mask of bins reachable from z0 outward through usable bins only."""
    keep = np.zeros_like(usable, dtype=bool)
    for side in (np.where(z > z0)[0], np.where(z < z0)[0][::-1]):
        for i in side:
            if not usable[i]:
                break
            keep[i] = True
    return keep


def masked(d, z, z_mid, k, key):
    """d[f"{k}__{key}"] with the bins outside the phase's drawn run set to NaN."""
    n_min = int(d["n_min_cells"][0])
    usable = ((d[f"{k}__n_cells"] >= n_min) & (d[f"{k}__vol_frac"] >= MIN_VOLUME_FRACTION)
              & np.isfinite(d[f"{k}__{key}"]))
    return np.where(contiguous_from(z, usable, z_mid), d[f"{k}__{key}"], np.nan)


def band_label(ax, y, text, va, color):
    ax.annotate(text, xy=(0.995, y), xycoords=("axes fraction", "data"),
                xytext=(0, 1.2 if va == "bottom" else -1.2), textcoords="offset points",
                ha="right", va=va, fontsize=TEXT_FS, color=color, zorder=9)


def main():
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    z = d["z_pc"]
    z_mid = float(d["z_mid_pc"][0])
    i_mid = int(np.where(d["slab_offsets_pc"] == 0.0)[0][0])

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)

    # ------------------------------------------------------------------ (a)
    tot, th = d["ref__MM21_sigma_total"], d["ref__MM21_sigma_th"]
    for key, (c, e) in (("total", tot), ("thermal", th)):
        ax_a.axhspan(c - e, c + e, color=BAND_COLOR[key], alpha=BAND_ALPHA[key], lw=0, zorder=0)
    band_label(ax_a, tot[0] + tot[1], "WNM total (MM21)", "bottom", "0.35")
    band_label(ax_a, th[0] - th[1], "WNM thermal (MM21)", "top", WNM)

    ax_a.plot(z, d["ALL__sigma_eff"], **ALL_KW)
    curves = [d["ALL__sigma_eff"], np.array([tot[0] + tot[1], th[0] - th[1]])]
    for k in PLOT_PHASES:
        se, st = masked(d, z, z_mid, k, "sigma_eff"), masked(d, z, z_mid, k, "sigma_th")
        ax_a.plot(z, se, color=PHASE_COLORS[k], lw=LINE_LW, zorder=4)
        ax_a.plot(z, st, color=PHASE_COLORS[k], lw=LINE_LW, ls=TH_LS, zorder=3)
        curves += [se, st]
    ax_a.set_yscale("log")
    lo, hi = robust_log_ylim([np.concatenate(curves)], lo_pct=0.0, hi_pct=100.0)
    ax_a.set_ylim(lo, hi * LOG_HEADROOM)
    ax_a.set_ylabel(r"$\sigma$ [km s$^{-1}$]")
    ax_a.axvline(z_mid, **ZMID_KW)
    # top of (a): the bottom is taken by the CNM sigma_th curve near z_mid
    ax_a.text(z_mid + 6, 0.97, ZMID_LABEL, transform=ax_a.get_xaxis_transform(),
              ha="left", va="top", fontsize=TEXT_FS, color="0.35", zorder=6)
    panel_label(ax_a, "a", loc="upper left")

    # ------------------------------------------------------------------ (b)
    m2_lo, m2_hi = float(d["ref__MM21_M2_lo"][0]), float(d["ref__MM21_M2_hi"][0])
    ht03 = float(d["ref__HT03_M2"][0])
    ax_b.axhspan(m2_lo, m2_hi, color=WNM, alpha=REF_BAND_ALPHA, lw=0, zorder=0)
    ax_b.axhline(ht03, **HT03_KW)
    guide_line(ax_b, 0.0)
    band_label(ax_b, ht03, "CNM (HT03)", "bottom", CNM)
    band_label(ax_b, m2_lo, "WNM (MM21)", "top", WNM)
    curves = [np.array([m2_lo, m2_hi, ht03, 0.0])]
    for k in PLOT_PHASES:
        m2 = masked(d, z, z_mid, k, "M2")
        ax_b.plot(z, m2, color=PHASE_COLORS[k], lw=LINE_LW, zorder=4)
        curves.append(m2)
    m2n = masked(d, z, z_mid, "NEUTRAL", "M2")
    ax_b.plot(z, m2n, **NEUTRAL_KW)
    curves.append(m2n)
    lo_b, hi_b = robust_linear_ylim([np.concatenate(curves)], lo_pct=0.5, hi_pct=99.5)
    ax_b.set_ylim(lo_b, hi_b)
    ax_b.set_ylabel(r"$\langle\mathcal{M}^2\rangle$")
    ax_b.axvline(z_mid, **ZMID_KW)
    panel_label(ax_b, "b", loc="upper left")

    # one compact legend: colour = gas, line style = which sigma
    handles = [Line2D([], [], **{k: v for k, v in ALL_KW.items() if k != "zorder"}),
               Line2D([], [], color=CNM, lw=LINE_LW), Line2D([], [], color=WNM, lw=LINE_LW),
               Line2D([], [], **{k: v for k, v in NEUTRAL_KW.items() if k != "zorder"}),
               Line2D([], [], color="0.3", lw=LINE_LW),
               Line2D([], [], color="0.3", lw=LINE_LW, ls=TH_LS)]
    labels = ["all gas", "CNM", "WNM", "all neutral (b)",
              r"$\sigma_\mathrm{eff}$", r"$\sigma_\mathrm{th}$"]
    fig.legend(handles, labels, loc="outside upper center", ncols=6, frameon=False,
               fontsize=TEXT_FS + 0.5, handlelength=1.9, columnspacing=1.1)

    ax_b.set_xlim(*Z_RANGE)
    ax_b.set_xticks(Z_TICKS)
    ax_b.set_xlabel(r"$z$ [pc]")
    provisional_label(fig, loc="lower right")
    print(f"  (a) y {lo:.2f} .. {hi * LOG_HEADROOM:.2f} km/s; (b) y {lo_b:.2f} .. {hi_b:.2f}")
    print(f"  midplane slab: sigma_eff all {d['slab__ALL__sigma_eff'][i_mid]:.2f}, "
          + ", ".join(f"{k} sigma_eff {d[f'slab__{k}__sigma_eff'][i_mid]:.2f} / th "
                      f"{d[f'slab__{k}__sigma_th'][i_mid]:.2f} / <M^2> "
                      f"{d[f'slab__{k}__M2'][i_mid]:.2f}" for k in PLOT_PHASES + ("NEUTRAL",)))
    save_figure(fig, NAME)


if __name__ == "__main__":
    main()
