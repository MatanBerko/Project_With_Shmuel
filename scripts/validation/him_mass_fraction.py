"""
Step 1b, results item 5: fraction of total gas mass (OBSERVED density,
i.e. the unmodified RAW variant's n_model = n_raw) in HIM-flagged cells,
per z-slab and total, within the +-500pc square footprint. Reads only
cache/core/alpha_core.zarr (produced by pipeline/compute_all.py) -- no
recomputation of any physics here.

HIM flag is variant-independent (same classification feeds RAW/HIM_A/
HIM_B alike -- only the density SUBSTITUTED into HIM cells differs by
variant), so this uses RAW's n_model, which for that variant equals the
raw observed density with no substitution.
"""

from pathlib import Path

import numpy as np
import zarr

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_TXT_PATH = Path("results/him_mass_fraction.txt")

SLAB_CENTERS_PC = (0.0, 150.0, 300.0)
SLAB_HALF_THICKNESS_PC = 25.0

PHASE_HIM = 3


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    z_pc = np.asarray(g["z_pc"][:], dtype=np.float64)
    vg = g["RAW"]
    n_observed = np.asarray(vg["n_model"][:], dtype=np.float64)  # RAW == observed, no substitution
    phase_flag = np.asarray(vg["phase_flag"][:])
    him = phase_flag == PHASE_HIM

    def mass_fraction(idxs):
        n_sub = n_observed[idxs]
        him_sub = him[idxs]
        total_mass = np.nansum(n_sub)
        him_mass = np.nansum(np.where(him_sub, n_sub, 0.0))
        return float(him_mass / total_mass) if total_mass > 0 else float("nan")

    rows = []
    for zc in SLAB_CENTERS_PC:
        lo, hi = zc - SLAB_HALF_THICKNESS_PC, zc + SLAB_HALF_THICKNESS_PC
        idxs = np.where((z_pc >= lo) & (z_pc <= hi))[0]
        rows.append((f"z={zc:.0f}pc slab ({zc - SLAB_HALF_THICKNESS_PC:.0f} to {zc + SLAB_HALF_THICKNESS_PC:.0f}pc)",
                     mass_fraction(idxs)))

    idxs_total = np.arange(len(z_pc))
    rows.append(("total (full +-750pc column)", mass_fraction(idxs_total)))

    header = "HIM gas-mass fraction (observed/RAW density, +-500pc square footprint)"
    line = " | ".join(f"{label}: {frac:.4%}" for label, frac in rows)
    report = header + "\n" + line + "\n"
    print(report)

    OUT_TXT_PATH.write_text(report)
    print(f"Saved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
