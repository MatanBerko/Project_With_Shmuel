"""
Design script for Fig 1 (BS19 thermochemical model, top + middle panels).

Loads cache/fig1_bs19_model.npz and produces a two-panel figure:
top panel T vs n, middle panel P/kB vs n. Presentation only — no
data-filtering or masking logic here (see compute_fig1.py for that).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.conventions import COLOR_PTH, apply_style

CACHE_PATH = "cache/fig1_bs19_model.npz"
OUTPUT_PATH = "figures/fig1_bs19_model.png"


def main():
    data = np.load(CACHE_PATH)
    n = data["n"]
    T = data["T"]
    P_over_kB = data["P_over_kB"]

    apply_style()
    fig, (ax_top, ax_mid) = plt.subplots(2, 1, figsize=(5, 7))

    ax_top.loglog(n, T, color=COLOR_PTH)
    ax_top.set_xlabel(r"$n$ (cm$^{-3}$)")
    ax_top.set_ylabel(r"$T$ (K)")

    ax_mid.loglog(n, P_over_kB, color=COLOR_PTH)
    ax_mid.set_xlabel(r"$n$ (cm$^{-3}$)")
    ax_mid.set_ylabel(r"$P_{\rm th}/k_B$ (K cm$^{-3}$)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
