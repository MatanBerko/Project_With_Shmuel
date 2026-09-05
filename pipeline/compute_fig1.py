"""
Compute script for Fig 1 (BS19 thermochemical model, top + middle panels).

Data source: Temp_Bialy_interp.mat, variable T_2d_n (shape (31, 591)), which
tabulates steady-state gas temperature T on a grid of (IUV_, n_) — 31 UV
field values by 591 density values. We select the row nearest IUV_ = 1
(the fiducial UV field for Fig 1 per the project's paper draft) and read
off T as a function of n_ at that fixed UV field.

NOTE: The standalone scalar `IUV` (0.001) and the standalone 1D `n`/`T`
(181-point) arrays in this .mat file were investigated and confirmed to be
leftover state from an unrelated loop iteration — they are NOT used here.

Pth/kB = n * T is Eq. 1 of the project's paper draft.
"""

import numpy as np
import scipy.io

from src.config_loader import load_resolved_config

CACHE_PATH = "cache/fig1_bs19_model.npz"


def main():
    cfg = load_resolved_config()
    mat = scipy.io.loadmat(str(cfg["bs19_mat_path"]))

    IUV_ = mat["IUV_"].flatten()
    n_ = mat["n_"].flatten()
    T_2d_n = mat["T_2d_n"]  # shape (len(IUV_), len(n_)), from T2d_Zm0_IUV_Pn.mat

    iuv_idx = np.argmin(np.abs(IUV_ - 1.0))
    iuv_value = IUV_[iuv_idx]

    T = T_2d_n[iuv_idx, :]  # T(n) at IUV_ ~= 1
    P_over_kB = n_ * T  # Eq. 1: Pth/kB = n*T [K cm^-3]

    np.savez(
        CACHE_PATH,
        n=n_,
        T=T,
        P_over_kB=P_over_kB,
        iuv_value=iuv_value,
    )
    print(f"Saved {CACHE_PATH} (IUV_ used = {iuv_value}, n points = {len(n_)})")


if __name__ == "__main__":
    main()
