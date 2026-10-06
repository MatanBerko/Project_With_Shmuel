"""
Step 1b, results item 5: fraction of total gas mass (OBSERVED density,
i.e. the unmodified RAW variant's n_model = n_raw) in HIM-flagged cells,
per z-slab and total, within the +-500pc square footprint. Reads only
cache/core/alpha_core.zarr (produced by pipeline/compute_all.py) -- no
recomputation of any physics here.

HIM flag is variant-independent (same classification feeds RAW/HIM_A/
HIM_B alike -- only the density SUBSTITUTED into HIM cells differs by
variant), so this uses RAW's n_model, which for that variant equals the
raw observed density with no substitution. Since Step 1c the HIM flag is
also stored in its own array ("him"), independent of PHASE_SCHEME.

Step 1d does not change these numbers: RAW's density is the observed one
and Step 1d only altered the SUBSTITUTED density inside HIM_A/HIM_B's HIM
cells. Reading RAW is therefore still the right way to ask "how much
observed gas mass does the HIM treatment touch". The Step 1d substitution
puts 1.1/2.3 = 0.478x as much mass in those cells as before, so the
fractions below now overstate, by about a factor two, how much mass the
HIM_A/HIM_B models themselves carry there.

Step 1c-prep: slabs are the Shelest+26 60pc-thick ones (|z - z_c| <= 30
pc) and the headline total is over the STATS_BOX (|x|,|y| <= 500,
|z| <= 400 pc). The full +-750 pc column total is still reported
alongside it, because that is the column the P_tot integral actually
uses -- how much HIM mass sits in it is the relevant context for P_tot,
even though no statistic is taken outside the box.
"""

import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    PARTICLES_PER_H_IONIZED,
    PARTICLES_PER_H_NEUTRAL,
    PROVISIONAL_CUBE_HEADER,
    SLAB_CENTERS_PC,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
)

ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
OUT_TXT_PATH = Path("results/him_mass_fraction.txt")


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    z_pc = np.asarray(g["z_pc"][:], dtype=np.float64)
    vg = g["RAW"]
    n_observed = np.asarray(vg["n_model"][:], dtype=np.float64)  # RAW == observed, no substitution
    him = np.asarray(vg["him"][:]).astype(bool)

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

    idxs_box = np.where(np.abs(z_pc) <= STATS_BOX_Z_HALF_RANGE_PC)[0]
    rows.append((f"STATS_BOX (|z|<={STATS_BOX_Z_HALF_RANGE_PC:.0f}pc)", mass_fraction(idxs_box)))

    idxs_total = np.arange(len(z_pc))
    rows.append((f"full column (|z|<={np.abs(z_pc).max():.0f}pc, the P_tot integration range)",
                 mass_fraction(idxs_total)))

    header = (f"{PROVISIONAL_CUBE_HEADER} | HIM substitution now balances the physical\n"
              f"pressure with fully ionized gas ({PARTICLES_PER_H_NEUTRAL:g} P / "
              f"({PARTICLES_PER_H_IONIZED:g} T_HIM)); the fractions below are of the "
              f"OBSERVED density.\n\n"
              f"HIM gas-mass fraction (observed/RAW density, +-500pc square footprint, "
              f"slabs |z-z_c|<={SLAB_HALF_THICKNESS_PC:.0f}pc)")
    line = " | ".join(f"{label}: {frac:.4%}" for label, frac in rows)
    report = header + "\n" + line + "\n"
    print(report)

    OUT_TXT_PATH.write_text(report)
    print(f"Saved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
