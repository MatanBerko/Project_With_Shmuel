"""
fig_alpha_z -- Fig. 4, the headline figure: alpha(z) and the two pressures
it is the ratio of.

Reads ONLY cache/core/area_alpha.npz (scripts/compute/compute_area_alpha.py).
RAW, self-gravity footprint_mean, balanced midplane (Step 1i), 4 pc bins,
VOLUME weighting. fig_area_alpha stays as the diagnostic version.

  (a) <P_tot>_area (plain mean over ALL cells of the plane, HIM included)
      and the harmonic mean <1/P_th>^-1 over the NON-HIM cells, with the
      15-85% range of the cells' p_th. That band is derived exactly from
      the alpha_cell percentiles: alpha_cell = <P_tot>_area / p_th with
      one <P_tot>_area per bin is monotone decreasing in p_th, so
          P_th,p15 = <P_tot>_area / alpha_cell,p85
          P_th,p85 = <P_tot>_area / alpha_cell,p15
      (the cache stores no p_th percentiles of its own to compare against).
  (b) the headline <alpha_cell> = <P_tot>_area <1/P_th>, the median
      alpha_cell, and its 15-85% band; alpha = 1 guide.
A thin dotted line marks z_mid in both panels.

HEADLINE = "alpha_area" swaps the headline in (b) to <P_tot>_area/<P_th>
(arithmetic) and the thermal line in (a) to the arithmetic <P_th>.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import COLOR_ALPHA  # noqa: E402
from src.plotting.style import (  # noqa: E402
    PRESSURE_COLORS,
    annotate,
    apply_paper_style,
    guide_line,
    panel_label,
    provisional_label,
    robust_log_ylim,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
CACHE = Path("cache/core/area_alpha.npz")
NAME = "fig_alpha_z"
HEADLINE = "alpha_cell_mean"          # "alpha_cell_mean" | "alpha_area"
SG = "mean"                           # self-gravity footprint_mean (fiducial)
FIG_W, FIG_H = 5.0, 4.4
Z_RANGE = (-400.0, 400.0)
Z_TICKS = (-400, -200, 0, 200, 400)
PTOT_COLOR = PRESSURE_COLORS["Ptot"]
PTH_COLOR = PRESSURE_COLORS["Pth"]
PTOT_LW = 1.4                         # Fig. 3's red edge
PTH_LW = 1.4
PTH_BAND_ALPHA = 0.15
ALPHA_BAND_ALPHA = 0.15               # fig_area_alpha's BAND_ALPHA_FILL
HEADLINE_LW = 2.0
MEDIAN_LW = 1.2
MEDIAN_LS = (0, (4.5, 2.0))
ZMID_KW = dict(color="0.6", lw=0.8, ls=":", zorder=2)   # as fig_ptot_budget
ZMID_LABEL = r"$z_\mathrm{mid}$"
LOG_HEADROOM = 1.8                    # room for the legend in (a)
TEXT_FS = 6.5


AREA_ALPHA_TXT = Path("results/area_alpha.txt")
CROSSCHECK_RTOL = 1e-6


def report_numbers(d):
    """Print the numbers the figure shows: the cached slabs (midplane = |z - z_mid|
    <= 30 pc, +150, +300) and the 4 pc bins nearest z_mid +- 150 and +- 300
    (slabs centred there are not cached; bin medians/percentiles cannot be
    combined into slab ones exactly). Asserts the midplane slab against
    results/area_alpha.txt section 6 (volume-weighted, sg mean)."""
    import re
    k = {"Ptot": f"{SG}__Ptot_area", "harm": f"{SG}__Pth_harm_vol",
         "amean": f"{SG}__alpha_cell_mean_vol", "amed": f"{SG}__alpha_cell_median_vol",
         "alo": f"{SG}__alpha_cell_p_lo_vol", "ahi": f"{SG}__alpha_cell_p_hi_vol"}
    rows = []
    for i, lab in enumerate(("slab mid", "slab +150", "slab +300")):
        rows.append((lab, *[float(d["slab__" + v][i]) for v in k.values()]))
    z, z_mid = d["z_pc"], float(d[f"{SG}__z_mid_pc"][0])
    for off in (-300.0, -150.0, 150.0, 300.0):
        j = int(np.argmin(np.abs(z - (z_mid + off))))
        rows.append((f"bin {z[j]:+.0f}", *[float(d[v][j]) for v in k.values()]))
    print(f"{'':>11}{'<P_tot>':>9}{'<1/P_th>^-1':>13}{'<a_cell>':>10}{'median [15, 85]':>22}")
    for lab, pt, hm, am, md, lo, hi in rows:
        print(f"{lab:>11}{pt:>9.1f}{hm:>13.1f}{am:>10.3f}{md:>9.3f} [{lo:.3f}, {hi:.3f}]")
    # cross-check the midplane slab against area_alpha.txt section 6
    text = AREA_ALPHA_TXT.read_text().splitlines()
    i6 = next(i for i, ln in enumerate(text) if ln.startswith("6. PER-CELL"))
    iv = next(i for i in range(i6, len(text)) if text[i].startswith("VOLUME-weighted"))
    line = next(ln for ln in text[iv:] if re.match(r"^\s+(mid|0)\s+mean\s", ln))
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
    ref = {"Ptot": nums[0], "harm": nums[2], "amean": nums[4], "amed": nums[5],
           "alo": nums[6], "ahi": nums[7]}
    mid = dict(zip(k, rows[0][1:]))
    worst = 0.0
    for key, r in ref.items():
        # the txt rounds to 1 (pressures) or 3 decimals: compare the cache value rounded the same way
        nd = 1 if key in ("Ptot", "harm") else 3
        rel = abs(round(mid[key], nd) / r - 1)
        worst = max(worst, rel)
        assert rel <= CROSSCHECK_RTOL, (key, mid[key], r)
    print(f"cross-check vs area_alpha.txt section 6 ({line.strip()[:60]}...): max rel {worst:.1e}")
    return rows


def main(headline=HEADLINE, out_name=NAME, directory=None):
    if headline not in ("alpha_cell_mean", "alpha_area"):
        raise ValueError(f"HEADLINE must be 'alpha_cell_mean' or 'alpha_area', got {headline!r}")
    apply_paper_style()
    d = np.load(CACHE, allow_pickle=False)
    report_numbers(d)
    z = d["z_pc"]
    box = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
    z_mid = float(d[f"{SG}__z_mid_pc"][0])
    ptot = d[f"{SG}__Ptot_area"]
    a_lo, a_hi = d[f"{SG}__alpha_cell_p_lo_vol"], d[f"{SG}__alpha_cell_p_hi_vol"]
    a_med = d[f"{SG}__alpha_cell_median_vol"]
    pth_band_lo, pth_band_hi = ptot / a_hi, ptot / a_lo          # exact, see docstring
    if headline == "alpha_cell_mean":
        pth_line = d[f"{SG}__Pth_harm_vol"]
        a_head = d[f"{SG}__alpha_cell_mean_vol"]
        pth_label = r"$\langle 1/P_\mathrm{th}\rangle^{-1}$"
        head_label = r"$\langle\alpha_\mathrm{cell}\rangle$"
        slab_alpha = float(d[f"slab__{SG}__alpha_cell_mean_vol"][0])
    else:
        pth_line = d[f"{SG}__Pth_neutral_vol"]
        a_head = d[f"{SG}__alpha_area_vol"]
        pth_label = r"$\langle P_\mathrm{th}\rangle$"
        head_label = r"$\alpha_\mathrm{area}$"
        slab_alpha = float(d[f"slab__{SG}__alpha_area_vol"][0])

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)

    # ------------------------------------------------------------------ (a)
    ax_a.fill_between(z, pth_band_lo, pth_band_hi, color=PTH_COLOR, alpha=PTH_BAND_ALPHA,
                      lw=0, zorder=1)
    ax_a.plot(z, pth_line, color=PTH_COLOR, lw=PTH_LW, zorder=4)
    ax_a.plot(z, ptot, color=PTOT_COLOR, lw=PTOT_LW, zorder=5)
    ax_a.set_yscale("log")
    lo, hi = robust_log_ylim([ptot, pth_line, pth_band_lo, pth_band_hi], z=z, z_range=Z_RANGE)
    ax_a.set_ylim(lo, hi * LOG_HEADROOM)
    ax_a.set_ylabel(r"$W/k_\mathrm{B}$, $P_\mathrm{th}/k_\mathrm{B}$ [K cm$^{-3}$]")
    ax_a.axvline(z_mid, ymax=0.88, **ZMID_KW)   # stops below the legend row
    ax_a.text(z_mid - 6, 0.04, ZMID_LABEL, transform=ax_a.get_xaxis_transform(),
              ha="right", va="bottom", fontsize=TEXT_FS, color="0.35", zorder=6)
    panel_label(ax_a, "a", loc="upper left")
    ax_a.legend([Line2D([], [], color=PTOT_COLOR, lw=PTOT_LW),
                 Line2D([], [], color=PTH_COLOR, lw=PTH_LW),
                 Patch(facecolor=PTH_COLOR, alpha=PTH_BAND_ALPHA, lw=0)],
                [r"$\langle W\rangle$", pth_label, r"$P_\mathrm{th}$, 15–85%"],
                loc="upper right", bbox_to_anchor=(1.0, 0.96),   # clear of PROVISIONAL
                ncols=3, frameon=False, fontsize=TEXT_FS + 0.5,
                handlelength=1.8, columnspacing=1.2, borderaxespad=0.3)

    # ------------------------------------------------------------------ (b)
    ax_b.fill_between(z, a_lo, a_hi, color=COLOR_ALPHA, alpha=ALPHA_BAND_ALPHA, lw=0, zorder=1)
    guide_line(ax_b, 1.0, label=r"$\alpha = 1$", label_x=0.42, label_loc="below")
    ax_b.plot(z, a_med, color=COLOR_ALPHA, lw=MEDIAN_LW, ls=MEDIAN_LS, zorder=4)
    ax_b.plot(z, a_head, color=COLOR_ALPHA, lw=HEADLINE_LW, zorder=5)
    ax_b.axvline(z_mid, ymin=0.14, **ZMID_KW)   # starts above the bottom legend row
    top = float(np.ceil(np.nanpercentile(a_hi[box], 99.0)))
    ax_b.set_ylim(0.0, top)
    ax_b.set_ylabel(r"$\alpha$")
    panel_label(ax_b, "b", loc="upper left")
    ax_b.legend([Line2D([], [], color=COLOR_ALPHA, lw=HEADLINE_LW),
                 Line2D([], [], color=COLOR_ALPHA, lw=MEDIAN_LW, ls=MEDIAN_LS),
                 Patch(facecolor=COLOR_ALPHA, alpha=ALPHA_BAND_ALPHA, lw=0)],
                [head_label, "median", "15–85%"],
                # bottom centre: below alpha = 1 the band only reaches down at |z| > 250 pc
                loc="lower center", ncols=3, frameon=False, fontsize=TEXT_FS + 0.5,
                handlelength=1.8, columnspacing=1.2, borderaxespad=0.3)
    annotate(ax_b, r"midplane: $\langle\alpha\rangle$ = " f"{slab_alpha:.2f}",
             loc="upper right", size=TEXT_FS)

    ax_b.set_xlim(*Z_RANGE)
    ax_b.set_xticks(Z_TICKS)
    ax_b.set_xlabel(r"$z$ [pc]")
    provisional_label(fig, loc="upper right")
    print(f"  (a) y range {lo:.0f} .. {hi * LOG_HEADROOM:.0f} K cm^-3;  (b) y range 0 .. {top:.0f}")
    kw = {} if directory is None else {"directory": directory}
    save_figure(fig, out_name, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", default=HEADLINE, choices=("alpha_cell_mean", "alpha_area"))
    ap.add_argument("--out-dir", default=None, help="write here instead of figures/ (for checks)")
    a = ap.parse_args()
    main(a.headline, NAME if a.out_dir is None else f"{NAME}_{a.headline}", a.out_dir)
