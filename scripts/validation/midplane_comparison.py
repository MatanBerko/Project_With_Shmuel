"""
Step 1i: the Sun-centred vs the balanced midplane, side by side.

Reads the pre-Step-1i caches saved in cache/sun_reference/ (copied before the
rebuild) and the current ones (balanced), and tabulates at the midplane slab,
z = 150 and z = 300 pc (self-gravity mean, RAW, volume-weighted):
  <P_tot>_area, alpha_area, median and mean alpha_cell   (area_alpha.npz)
  500 pc patch median and 15/85 of <alpha>_i             (patch_alpha.npz)
plus the Fig. 3 midplane values (ptot_budget.npz) and the Fig. 1 numbers
(model_pressure.npz). The midplane slab is |z| <= 30 pc for "sun" and
|z - z_mid| <= 30 pc for "balanced".
Writes results/midplane_comparison.txt. Reads caches only.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import PROVISIONAL_CUBE_HEADER  # noqa: E402

SUN = Path("cache/sun_reference")
NEW = {"area": Path("cache/core/area_alpha.npz"), "patch": Path("cache/core/patch_alpha.npz"),
       "budget": Path("cache/ptot_budget/ptot_budget.npz"),
       "model": Path("cache/model/model_pressure.npz")}
OLD = {"area": SUN / "area_alpha.npz", "patch": SUN / "patch_alpha.npz",
       "budget": SUN / "ptot_budget.npz", "model": SUN / "model_pressure.npz"}
OUT = Path("results/midplane_comparison.txt")


def main():
    o = {k: np.load(p, allow_pickle=False) for k, p in OLD.items()}
    n = {k: np.load(p, allow_pickle=False) for k, p in NEW.items()}
    zm = float(n["area"]["mean__z_mid_pc"][0])
    L = [PROVISIONAL_CUBE_HEADER,
         "Step 1i: Sun-centred ('sun') vs balanced midplane ('balanced'), RAW, self-gravity mean",
         "=" * 104,
         f"balanced z_mid = {zm:+.3f} pc (self-gravity mean), "
         f"{float(n['area']['off__z_mid_pc'][0]):+.3f} pc (off). Midplane slab: |z| <= 30 pc (sun), "
         f"|z - z_mid| <= 30 pc (balanced).",
         "",
         f"{'slab':>6}{'mode':>10}{'<P_tot>_area':>14}{'alpha_area':>12}{'med a_cell':>12}"
         f"{'mean a_cell':>13}{'patch500 <alpha>_i med [p15, p85]':>36}",
         f"{'':>6}{'':>10}{'[K cm^-3]':>14}{'(vol)':>12}{'(vol)':>12}{'(vol)':>13}",
         "-" * 103]
    for i, lab in enumerate(("mid", "150", "300")):
        for mode, d in (("sun", o), ("balanced", n)):
            a, p = d["area"], d["patch"]
            s = "slab__mean__"
            q = "slab__L500__mean__alpha_vol__"
            L.append(f"{lab:>6}{mode:>10}{a[s + 'Ptot_area'][i]:>14.1f}{a[s + 'alpha_area_vol'][i]:>12.3f}"
                     f"{a[s + 'alpha_cell_median_vol'][i]:>12.3f}{a[s + 'alpha_cell_mean_vol'][i]:>13.3f}"
                     f"{p[q + 'median'][i]:>16.3f}  [{p[q + 'p_lo'][i]:.3f}, {p[q + 'p_hi'][i]:.3f}]")
        L.append("")
    ob, nb = o["budget"], n["budget"]
    L += ["Fig. 3 midplane (1-D weight budget):",
          f"  sun:      P(0+) = {ob['P0_total'][0]:.1f}, P(0-) = {ob['P0_total'][1]:.1f} K cm^-3 (the jump)",
          f"  balanced: P(z_mid) = {float(nb['P_zmid'][0]):.1f} K cm^-3 "
          f"(top-down {nb['P0_total'][0]:.4f}, bottom-up {nb['P0_total'][1]:.4f})",
          "  shares at z_mid: stars {:.1%}, DM {:.1%}, gas {:.1%}".format(*nb["shares_zmid"]),
          ""]
    om, nm = o["model"], n["model"]
    L += ["Fig. 1 numbers (midplane slab, volume-weighted):",
          f"{'':>4}{'':>10}{'<P_tot>':>10}{'<1/P_th>^-1':>13}{'I_UV,cross':>12}{'vol share < cross':>19}"]
    for mode, d in (("sun", om), ("balanced", nm)):
        L.append(f"{'':>4}{mode:>10}{float(d['midplane_Ptot_area'][0]):>10.1f}"
                 f"{float(d['midplane_Pth_inv_inv'][0]):>13.1f}{float(d['iuv_cross_pmax_eq_ptot'][0]):>12.4f}"
                 f"{float(d['f_vol_iuv_below_cross'][0]):>19.4f}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
