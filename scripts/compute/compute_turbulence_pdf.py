"""
Fig. 6: turbulence TWO WAYS -- the density-PDF width next to the weight.

compute_velocity.py turns the missing support alpha - 1 into a non-thermal
velocity (an UPPER limit: all of alpha - 1 is assigned to motions). This
script makes an INDEPENDENT estimate from the width of the log-density
distribution in each height bin, using the supersonic-turbulence relation
between the density variance and the Mach number. It is a LOWER limit: the
map's resolution and regularisation remove small-scale density variance.

Definitions, per signed 4 pc z bin and per +-30 pc slab, over the whole
+-500 pc footprint, for each sample k in {CNM, UNM, WNM, NEUTRAL = all
non-HIM}, using the cells of sample k in the group (exactly the cells
compute_velocity.py uses):

  <n>_k             VOLUME (plain) mean of n_H over the cells of k
  s                 ln(n_H / <n>_k)
  sigma_s,k         std(s), volume weighted (every cell weight 1, ddof = 0)
  sigma_s,rob,k     (p84 - p16) / 2 of s, volume weighted (nearest rank,
                    src.physics.stats.weighted_stats, scheme "16_84");
                    equals sigma_s for a Gaussian, insensitive to the tails
  M3D^2(b)          (exp(sigma_s^2) - 1) / b^2,  b in {1/3, 0.4, 1}
                    (b = 1/3 solenoidal, 1 compressive forcing; 0.4 mixed)
  M2_pdf(b)         M3D^2(b) / 3: the 1D convention of Fig. 7 (isotropy,
                    M_1D^2 = sigma_1D^2 / c_s^2 = M3D^2 / 3)
  c_s,k             sqrt(sum(k_B p_th_phys) / sum(rho)) [km/s], rho = 1.4 m_H
                    n_H: the mass-weighted isothermal sound speed, computed by
                    compute_velocity's own phase_sums / velocity_stats
  sigma_turb_pdf(b) sqrt(M2_pdf(b)) c_s,k   [km/s, 1D]
and the same M3D^2, M2_pdf, sigma_turb_pdf from sigma_s,rob (suffix _rob).

Resolution test: n_H block-averaged over 2 x 2 x 2 cells (4 pc blocks).
Blocks are aligned on the box grid: z pairs (2j, 2j+1) of box planes, both
inside the group (a 4 pc bin is one block thick; the closed last bin's
third plane is dropped); in x, y the 501 x 501 footprint is cropped to its
first 500 x 500 cells (the last row and column are dropped). A block's
phase is the phase of the strict majority (>= 5) of its 8 cells; a block
with no majority is skipped. A block is NEUTRAL when >= 5 of its cells are
non-HIM. A block's n is the plain mean of all 8 cells. sigma_s_4pc,k is
sigma_s over the blocks of k, with s relative to their own volume mean.

Alpha-method band: the MASS-weighted (w = rho) 15-85% range of the per-cell
  sigma_nt,cell     sqrt(max(k_B (W - p_th_phys) / rho, 0)) [km/s]
for CNM, WNM, NEUTRAL, with W the group's weight read from velocity.npz
(fig_velocity draws no sigma_nt band, so velocity.npz does not hold one;
compute_velocity.py is not modified). The unclipped mass-weighted mean of
sigma_nt,cell^2 is compute_velocity's sigma_nt^2, so the band belongs to
the line it is drawn behind.

Phase histograms (for Fig. 5): per 4 pc bin and per slab, the cell
(= volume, all cells are 2 pc cubes) counts of log10(n_H) in
LOG_N_EDGES for CNM, UNM, WNM, HIM, plus the counts below and above the
edges, so that counts + under + over = cells of the phase.

s-PDFs (appendix): for |z - z_mid| = d in PDF_OFFSETS_PC, the union of the
4 pc bin(s) containing z_mid + d and z_mid - d (one bin at d = 0), with
sigma_s, <s> and counts of s in S_EDGES per sample, s relative to the
union's own <n>_k.

A statistic is NaN in a group with fewer than N_MIN_CELLS cells (blocks,
for sigma_s_4pc) of its sample -- the compute_velocity.py rule.

Inputs (nothing is rebuilt):
  cache/core/alpha_core.zarr   RAW n_model, p_th_phys, phase_flag_dpdn, him
  cache/velocity/velocity.npz  bins, slabs, W per group, and the alpha-method
                               numbers for the cross-checks and the table
Cross-checks (raise on failure): bins and slab centres equal velocity.npz's;
every sample's cell count equals velocity.npz's; c_s and <M^2>_NEUTRAL equal
velocity.npz's to CS_IDENTITY_RTOL; phase flag HIM == him flag.

COMPUTE script: writes cache/turbulence_pdf/turbulence_pdf.npz and
results/turbulence_pdf.txt. scripts/design/fig_turbulence.py and
fig_density_pdf_width.py read only the npz files.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conventions import (  # noqa: E402
    K_B,
    M_H,
    MIDPLANE_MODE,
    MU,
    PERCENTILE_SCHEME_16_84,
    PERCENTILE_SCHEME_DEFAULT,
    PROFILE_BIN_PC,
    PROVISIONAL_CUBE_HEADER,
    SLAB_HALF_THICKNESS_PC,
    STATS_BOX_XY_HALF_RANGE_PC,
    STATS_BOX_Z_HALF_RANGE_PC,
    THERMAL_PRESSURE_HEADER,
)
from src.physics import midplane  # noqa: E402
from src.physics.him import PHASE_CNM, PHASE_HIM, PHASE_UNM, PHASE_WNM  # noqa: E402
from src.physics.loading import open_zarr, slab_z_indices  # noqa: E402
from src.physics.stats import percentile_levels, weighted_stats  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "compute_velocity", Path(__file__).resolve().parent / "compute_velocity.py")
cv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cv)

# ═══════════════════════════════════════════════════════════════════════
# ██  CONTROL BLOCK
# ═══════════════════════════════════════════════════════════════════════
ALPHA_CORE_ZARR_PATH = Path("cache/core/alpha_core.zarr")
VELOCITY_NPZ_PATH = Path("cache/velocity/velocity.npz")
OUT_NPZ_PATH = Path("cache/turbulence_pdf/turbulence_pdf.npz")
OUT_TXT_PATH = Path("results/turbulence_pdf.txt")

VARIANT = cv.VARIANT                          # RAW
SG = cv.SG                                    # self-gravity mean (fiducial)
SLAB_OFFSETS_PC = cv.SLAB_OFFSETS_PC          # (-300, -150, 0, +150, +300) about z_mid
N_MIN_CELLS = cv.N_MIN_CELLS                  # 100
Z_CHUNK = cv.Z_CHUNK                          # 64 planes per read
B_VALUES = {"b1_3": 1.0 / 3.0, "b0p4": 0.4, "b1": 1.0}   # forcing parameter b
B_FIDUCIAL = "b0p4"
ROBUST_SCHEME = PERCENTILE_SCHEME_16_84       # (p84 - p16) / 2
BLOCK = 2                                     # 2 x 2 x 2 cells -> 4 pc blocks
BLOCK_MAJORITY = 5                            # strict majority of 8
LOG_N_EDGES = np.round(np.arange(-4.0, 3.0 + 1e-9, 0.05), 10)   # log10 n_H [cm^-3]
S_EDGES = np.round(np.arange(-10.0, 8.0 + 1e-9, 0.1), 10)       # s = ln(n/<n>)
PDF_OFFSETS_PC = (0.0, 100.0, 200.0, 300.0)   # |z - z_mid| of the appendix PDFs
CS_IDENTITY_RTOL = 1e-14                      # c_s / <M^2> vs velocity.npz
# ═══════════════════════════════════════════════════════════════════════

SAMPLES = ("CNM", "UNM", "WNM", "NEUTRAL")
HIST_PHASES = (("CNM", PHASE_CNM), ("UNM", PHASE_UNM), ("WNM", PHASE_WNM), ("HIM", PHASE_HIM))
BAND_SAMPLES = ("CNM", "WNM", "NEUTRAL")
REPORT_SAMPLES = ("CNM", "WNM", "NEUTRAL")
KMS = cv.KMS


def sample_masks(ph):
    """{sample: bool mask} on phase codes; NEUTRAL = every non-HIM cell."""
    ph = np.asarray(ph)
    return {"CNM": ph == PHASE_CNM, "UNM": ph == PHASE_UNM, "WNM": ph == PHASE_WNM,
            "NEUTRAL": ph != PHASE_HIM}


# ---------------------------------------------------------------------------
# the density-PDF statistics
# ---------------------------------------------------------------------------
def pdf_width(n, n_min=N_MIN_CELLS, s_edges=None):
    """sigma_s, the robust sigma_s and <s> of s = ln(n / <n>), <n> the volume mean.

    Returns {"n": count, "sigma_s", "sigma_s_robust", "mean_s"} (NaN below
    n_min), plus "s_hist" (counts in s_edges, under, over) if s_edges is given.
    """
    n = np.asarray(n, dtype=np.float64).ravel()
    out = {"n": float(n.size), "sigma_s": np.nan, "sigma_s_robust": np.nan, "mean_s": np.nan}
    if s_edges is not None:
        out["s_hist"] = np.zeros(len(s_edges) - 1)
        out["s_under"] = out["s_over"] = 0.0
    if n.size < n_min or n.size == 0:
        return out
    s = np.log(n / n.mean())
    st = weighted_stats(s, np.ones_like(s), ROBUST_SCHEME)
    out.update(sigma_s=float(s.std()), sigma_s_robust=0.5 * (st.p_hi - st.p_lo),
               mean_s=float(s.mean()))
    if s_edges is not None:
        out["s_hist"] = np.histogram(s, bins=s_edges)[0].astype(np.float64)
        out["s_under"] = float(np.sum(s < s_edges[0]))
        out["s_over"] = float(np.sum(s > s_edges[-1]))
    return out


def mach2_3d(sigma_s, b):
    """M3D^2 = (exp(sigma_s^2) - 1) / b^2."""
    return np.expm1(np.asarray(sigma_s, dtype=np.float64) ** 2) / b ** 2


def mach2_1d(sigma_s, b):
    """The 1D convention of Fig. 7 under isotropy: M3D^2 / 3."""
    return mach2_3d(sigma_s, b) / 3.0


def block_average(n2, ph2, block=BLOCK, majority=BLOCK_MAJORITY):
    """2 x 2 x 2 block average of a (2, ny, nx) pair of planes.

    The x, y extent is cropped to a whole number of blocks. Returns
    (n_block, code, neutral): the plain mean n of each block's 8 cells,
    the majority phase code (-1 if none reaches `majority`) and whether
    >= `majority` of its cells are non-HIM, all flat.
    """
    n2 = np.asarray(n2, dtype=np.float64)
    ph2 = np.asarray(ph2)
    nz, ny, nx = n2.shape
    if nz != block:
        raise ValueError(f"block_average needs {block} planes, got {nz}")
    by, bx = ny // block, nx // block
    shp = (block, by, block, bx, block)
    n_b = n2[:, :by * block, :bx * block].reshape(shp).mean(axis=(0, 2, 4)).ravel()
    ph_c = ph2[:, :by * block, :bx * block].reshape(shp)
    counts = np.stack([(ph_c == c).sum(axis=(0, 2, 4)).ravel() for c in range(cv.N_CODES)])
    best = counts.argmax(axis=0)
    code = np.where(counts.max(axis=0) >= majority, best, -1)
    neutral = (block ** 3 - counts[PHASE_HIM]) >= majority
    return n_b, code, neutral


def blocks_of_group(planes, block=BLOCK):
    """Block-average the complete z pairs of a group's planes.

    planes: {box plane index: (n 2D, phase 2D)}. Pairs are (2j, 2j+1) on the
    box grid with both planes in the group. Returns (n, code, neutral) flat.
    """
    idx = sorted(planes)
    out = []
    for i in idx:
        if i % block == 0 and all((i + d) in planes for d in range(1, block)):
            out.append(block_average(np.stack([planes[i + d][0] for d in range(block)]),
                                     np.stack([planes[i + d][1] for d in range(block)])))
    if not out:
        return np.zeros(0), np.zeros(0, int), np.zeros(0, bool)
    return tuple(np.concatenate(a) for a in zip(*out))


def block_sample_masks(code, neutral):
    return {"CNM": code == PHASE_CNM, "UNM": code == PHASE_UNM, "WNM": code == PHASE_WNM,
            "NEUTRAL": np.asarray(neutral, dtype=bool)}


def phase_histograms(log10_n, ph, edges=LOG_N_EDGES):
    """Cell counts of log10 n_H per phase code: (counts (4, nb), under (4,), over (4,))."""
    x = np.asarray(log10_n, dtype=np.float64).ravel()
    ph = np.asarray(ph).ravel()
    counts = np.zeros((len(HIST_PHASES), len(edges) - 1))
    under = np.zeros(len(HIST_PHASES))
    over = np.zeros(len(HIST_PHASES))
    for j, (_, code) in enumerate(HIST_PHASES):
        v = x[ph == code]
        counts[j] = np.histogram(v, bins=edges)[0]
        under[j] = np.sum(v < edges[0])
        over[j] = np.sum(v > edges[-1])
    return counts, under, over


def sigma_nt_cell_band(n_H, p_th, W, n_min=N_MIN_CELLS, scheme=PERCENTILE_SCHEME_DEFAULT):
    """Mass-weighted low/high percentiles of sqrt(max(k_B (W - p_th)/rho, 0)) [km/s]."""
    n = np.asarray(n_H, dtype=np.float64).ravel()
    p = np.asarray(p_th, dtype=np.float64).ravel()
    if n.size < n_min:
        return np.nan, np.nan
    rho = MU * M_H * n
    v = np.sqrt(np.maximum(K_B * (W - p) / rho, 0.0)) / KMS
    st = weighted_stats(v, rho, scheme)
    return st.p_lo, st.p_hi


def derived(sig, c_s):
    """{key: array} of M3D^2, M2_pdf and sigma_turb_pdf for every b, from sigma_s and its robust twin."""
    out = {}
    for suf, s in (("", sig["sigma_s"]), ("_rob", sig["sigma_s_robust"])):
        for tag, b in B_VALUES.items():
            m2 = mach2_1d(s, b)
            out[f"M3D2_pdf_{tag}{suf}"] = mach2_3d(s, b)
            out[f"M2_pdf_{tag}{suf}"] = m2
            out[f"sigma_turb_pdf_{tag}{suf}"] = np.sqrt(m2) * c_s
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    import pipeline.compute_all as pipe

    vel = np.load(VELOCITY_NPZ_PATH, allow_pickle=False)
    z_full = np.asarray(open_zarr().z_pc, dtype=np.float64)
    zsl, z_box = pipe._box_z_slice(z_full)
    core = zarr.open_group(str(ALPHA_CORE_ZARR_PATH), mode="r")
    vg = core[VARIANT]
    sg_attrs = vg[f"self_gravity_{SG}"].attrs
    if sg_attrs.get("midplane_mode", "sun") != MIDPLANE_MODE:
        raise RuntimeError(f"cache built with midplane {sg_attrs.get('midplane_mode')!r}, "
                           f"MIDPLANE_MODE is {MIDPLANE_MODE!r}")
    z_mid = float(sg_attrs.get("z_mid_pc", 0.0))
    if abs(float(vel["z_mid_pc"][0]) - z_mid) > 1e-9:
        raise RuntimeError("velocity.npz z_mid differs from the zarr's")

    # ---- groups: 4 pc bins, the five slabs (as compute_velocity), the PDF unions
    bin_groups = pipe.profile_bin_groups(z_box, use_abs=False)
    z_bins = np.array([c for c, _ in bin_groups], dtype=float)
    if not np.array_equal(z_bins, vel["z_pc"]):
        raise RuntimeError("bins differ from velocity.npz's")
    mid_center = midplane.slab_center(0.0, z_mid, MIDPLANE_MODE)
    slab_centers = np.array([mid_center + off for off in SLAB_OFFSETS_PC])
    if not np.array_equal(slab_centers, vel["slab_centers_pc"]):
        raise RuntimeError("slab centres differ from velocity.npz's")
    slab_idx = [np.asarray(slab_z_indices(z_box, zc), int) for zc in slab_centers]
    i_mid = SLAB_OFFSETS_PC.index(0.0)
    half = 0.5 * PROFILE_BIN_PC
    pdf_bins = []
    for d in PDF_OFFSETS_PC:
        sel = sorted({int(np.argmin(np.abs(z_bins - (z_mid + sgn * d)))) for sgn in (+1, -1)})
        for b in sel:
            zt = [z_mid + s * d for s in (+1, -1)]
            if not any(z_bins[b] - half <= t < z_bins[b] + half for t in zt):
                raise RuntimeError(f"no 4 pc bin contains z_mid +- {d}")
        pdf_bins.append(sel)
    groups = ([np.asarray(i, int) for _, i in bin_groups] + slab_idx
              + [np.concatenate([np.asarray(bin_groups[b][1], int) for b in sel])
                 for sel in pdf_bins])
    n_bins, n_slabs = len(bin_groups), len(slab_idx)
    n_full = n_bins + n_slabs                    # groups with the full treatment
    n_groups = len(groups)
    W = np.concatenate([vel["W"], vel["slab__W"]])
    plane_groups = [[] for _ in range(len(z_box))]
    for g, idxs in enumerate(groups):
        for i in idxs:
            plane_groups[int(i)].append(g)
    last_plane = np.array([g.max() if g.size else -1 for g in groups])
    print(f"z_mid = {z_mid:+.4f} pc; {n_bins} bins + {n_slabs} slabs + "
          f"{len(pdf_bins)} PDF groups (bins {pdf_bins})")

    # ---- accumulate over z chunks ------------------------------------------
    acc = {k: np.zeros((n_full, cv.N_CODES)) for k in cv.SUM_KEYS}
    nb_n, nb_s = len(LOG_N_EDGES) - 1, len(S_EDGES) - 1
    hist = np.zeros((n_full, len(HIST_PHASES), nb_n))
    hist_under = np.zeros((n_full, len(HIST_PHASES)))
    hist_over = np.zeros((n_full, len(HIST_PHASES)))
    res = {k: {q: np.full(n_groups, np.nan) for q in
               ("n", "sigma_s", "sigma_s_robust", "mean_s", "n_blocks", "sigma_s_4pc",
                "sigma_nt_cell_p_lo", "sigma_nt_cell_p_hi")} for k in SAMPLES}
    s_hist = {k: np.zeros((n_groups, nb_s)) for k in SAMPLES}
    s_uo = {k: np.zeros((n_groups, 2)) for k in SAMPLES}
    pending = {g: {} for g in range(n_groups)}
    n_arr, p_arr, ph_arr, him_arr = (vg["n_model"], vg["p_th_phys"],
                                     vg["phase_flag_dpdn"], vg["him"])
    plane_shape = n_arr.shape[1:]
    n_bad = 0

    def finish(g):
        planes = pending[g]
        idx = sorted(planes)
        n_g = np.concatenate([planes[i][0].ravel() for i in idx])
        ph_g = np.concatenate([planes[i][1].ravel() for i in idx])
        p_g = np.concatenate([planes[i][2].ravel() for i in idx])
        masks = sample_masks(ph_g)
        want_hist = g >= n_bins                  # slabs and PDF groups
        for k in SAMPLES:
            w = pdf_width(n_g[masks[k]], s_edges=S_EDGES if want_hist else None)
            for q in ("n", "sigma_s", "sigma_s_robust", "mean_s"):
                res[k][q][g] = w[q]
            if want_hist:
                s_hist[k][g] = w["s_hist"]
                s_uo[k][g] = (w["s_under"], w["s_over"])
        if g < n_full:
            for k in BAND_SAMPLES:
                m = masks[k]
                res[k]["sigma_nt_cell_p_lo"][g], res[k]["sigma_nt_cell_p_hi"][g] = \
                    sigma_nt_cell_band(n_g[m], p_g[m], W[g])
            nb, code, neu = blocks_of_group({i: (planes[i][0], planes[i][1]) for i in idx})
            for k, m in block_sample_masks(code, neu).items():
                w = pdf_width(nb[m])
                res[k]["n_blocks"][g] = w["n"]
                res[k]["sigma_s_4pc"][g] = w["sigma_s"]
        pending[g] = None

    print(f"Reducing {len(z_box)} box planes in chunks of {Z_CHUNK}...")
    for lo in range(0, len(z_box), Z_CHUNK):
        hi = min(lo + Z_CHUNK, len(z_box))
        sl = slice(zsl.start + lo, zsl.start + hi)
        # identical reads and reshapes to compute_velocity.py
        n_c = np.asarray(n_arr[sl], dtype=np.float64).reshape(hi - lo, -1)
        p_c = np.asarray(p_arr[sl], dtype=np.float64).reshape(hi - lo, -1)
        ph_c = np.asarray(ph_arr[sl]).reshape(hi - lo, -1)
        him_c = np.asarray(him_arr[sl]).astype(bool).reshape(hi - lo, -1)
        if not np.array_equal(him_c, ph_c == PHASE_HIM):
            raise RuntimeError("phase_flag_dpdn == HIM differs from the him flag")
        n_bad += int(np.sum(~(np.isfinite(n_c) & np.isfinite(p_c) & (n_c > 0) & (p_c > 0))))
        for k in range(hi - lo):
            gs = plane_groups[lo + k]
            if not gs:
                continue
            h = phase_histograms(np.log10(n_c[k]), ph_c[k])
            # copies, so a pending group does not keep the whole chunk alive
            planes_k = (n_c[k].reshape(plane_shape).copy(), ph_c[k].reshape(plane_shape).copy(),
                        p_c[k].copy())
            for g in gs:
                if g < n_full:
                    s = cv.phase_sums(n_c[k], p_c[k], ph_c[k], W[g])
                    for key in cv.SUM_KEYS:
                        acc[key][g] += s[key]
                    hist[g] += h[0]
                    hist_under[g] += h[1]
                    hist_over[g] += h[2]
                pending[g][lo + k] = planes_k
                if last_plane[g] == lo + k:
                    finish(g)
        del n_c, p_c, ph_c, him_c
        print(f"  planes {lo}..{hi - 1} done ({time.time() - t0:.0f} s)")
    if n_bad:
        raise RuntimeError(f"{n_bad} box cells have non-finite or non-positive n_H or p_th")
    if any(v is not None for v in pending.values()):
        raise RuntimeError("some groups were never completed")

    # ---- c_s through compute_velocity's own statistics -------------------------
    comb = cv.combine(acc)
    n_total = comb["ALL"]["n"]
    vstats = {k: cv.velocity_stats(comb[k], W, n_total)[0] for k in SAMPLES}
    c_s = {k: vstats[k]["sigma_th"] for k in SAMPLES}

    # ---- cross-checks against velocity.npz -------------------------------------
    vv = lambda k, q: np.concatenate([vel[f"{k}__{q}"], vel[f"slab__{k}__{q}"]])  # noqa: E731
    checks = {}
    for k in SAMPLES:
        if not np.array_equal(res[k]["n"][:n_full], vv(k, "n_cells")):
            raise RuntimeError(f"{k}: cell counts differ from velocity.npz")
        if not np.array_equal(vstats[k]["n_cells"], vv(k, "n_cells")):
            raise RuntimeError(f"{k}: phase_sums counts differ from velocity.npz")
        for q in ("sigma_th", "M2"):
            a, b = vstats[k][q], vv(k, q)
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise RuntimeError(f"{k} {q}: defined groups differ from velocity.npz")
            f = np.isfinite(a)
            r = float(np.max(np.abs(a[f] / b[f] - 1.0))) if f.any() else 0.0
            checks[(k, q)] = r
            if r > CS_IDENTITY_RTOL:
                raise RuntimeError(f"{k} {q}: differs from velocity.npz by {r:.3e}")
    cs_max = max(v for (k, q), v in checks.items() if q == "sigma_th")
    m2_max = max(v for (k, q), v in checks.items() if q == "M2")
    print(f"c_s vs velocity.npz: max rel {cs_max:.2e}; <M^2> max rel {m2_max:.2e}")
    hist_total = hist.sum(axis=2) + hist_under + hist_over
    if not np.array_equal(hist_total, acc["n"]):
        raise RuntimeError("phase histogram counts do not sum to the phase cell counts")

    # ---- save ----------------------------------------------------------------
    out = {
        "z_pc": z_bins, "bin_width_pc": np.array([PROFILE_BIN_PC]),
        "z_mid_pc": np.array([z_mid]),
        "slab_offsets_pc": np.array(SLAB_OFFSETS_PC), "slab_centers_pc": slab_centers,
        "slab_half_thickness_pc": np.array([SLAB_HALF_THICKNESS_PC]),
        "n_min_cells": np.array([N_MIN_CELLS]),
        "b_tags": np.array(list(B_VALUES)), "b_values": np.array(list(B_VALUES.values())),
        "b_fiducial": np.array([B_FIDUCIAL]),
        "robust_levels": np.array(percentile_levels(ROBUST_SCHEME)),
        "percentile_levels": np.array(percentile_levels(PERCENTILE_SCHEME_DEFAULT)),
        "log_n_edges": LOG_N_EDGES, "s_edges": S_EDGES,
        "hist_phases": np.array([p for p, _ in HIST_PHASES]),
        "hist_log_n": hist[:n_bins], "hist_log_n_under": hist_under[:n_bins],
        "hist_log_n_over": hist_over[:n_bins],
        "slab__hist_log_n": hist[n_bins:], "slab__hist_log_n_under": hist_under[n_bins:],
        "slab__hist_log_n_over": hist_over[n_bins:],
        "hist_n_cells": acc["n"][:n_bins], "slab__hist_n_cells": acc["n"][n_bins:],
        "pdfz__offsets_pc": np.array(PDF_OFFSETS_PC),
        "pdfz__bins": np.array([sel + [sel[0]] * (2 - len(sel)) for sel in pdf_bins]),
        "check_cs_max_rel": np.array([cs_max]), "check_M2_max_rel": np.array([m2_max]),
        "provisional": np.array([PROVISIONAL_CUBE_HEADER]),
        "self_gravity": np.array([SG]), "variant": np.array([VARIANT]),
    }
    parts = (("", slice(0, n_bins)), ("slab__", slice(n_bins, n_full)),
             ("pdfz__", slice(n_full, n_groups)))
    for k in SAMPLES:
        full = {**res[k], "c_s": np.concatenate([c_s[k], np.full(n_groups - n_full, np.nan)]),
                "vol_frac": np.concatenate([vstats[k]["vol_frac"],
                                            np.full(n_groups - n_full, np.nan)])}
        full.update(derived(full, full["c_s"]))
        for pre, sl in parts:
            for q, arr in full.items():
                if pre == "pdfz__" and q not in ("n", "sigma_s", "sigma_s_robust", "mean_s"):
                    continue
                out[f"{pre}{k}__{q}"] = np.asarray(arr)[sl]
        for pre, sl in parts[1:]:
            out[f"{pre}{k}__s_hist"] = s_hist[k][sl]
            out[f"{pre}{k}__s_under_over"] = s_uo[k][sl]
    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ_PATH, **out)
    print(f"Saved {OUT_NPZ_PATH}")
    write_report(out, vel, i_mid, time.time() - t0)


# ---------------------------------------------------------------------------
def write_report(out, vel, i_mid, runtime):
    g = lambda k, q, i: out[f"slab__{k}__{q}"][i]  # noqa: E731
    L = [
        PROVISIONAL_CUBE_HEADER,
        THERMAL_PRESSURE_HEADER,
        "Fig. 6: turbulence TWO WAYS -- density-PDF width vs the weight (alpha)",
        "=" * 118,
        f"{VARIANT}, self-gravity {SG} (fiducial), balanced midplane z_mid = "
        f"{float(out['z_mid_pc'][0]):+.3f} pc, footprint |x|,|y| <= "
        f"{STATS_BOX_XY_HALF_RANGE_PC:.0f} pc, box |z| <= {STATS_BOX_Z_HALF_RANGE_PC:.0f} pc.",
        "Phases: cached dP/dn flag (CNM = 0, UNM = 1, WNM = 2, HIM = 3); NEUTRAL = CNM+UNM+WNM.",
        "rho = 1.4 m_H n_H;  p_th_phys = 1.1 n_H k T;  all velocities 1D.",
        "",
        "DEFINITIONS (per 4 pc bin or +-30 pc slab; cells of sample k in the group):",
        "  s               ln(n_H / <n>_k), <n>_k the VOLUME mean of n_H over k",
        "  sig_s           std(s), volume weighted",
        "  sig_s_rob       (p84 - p16)/2 of s, volume weighted (nearest rank)",
        "  sig_s_4pc       sig_s after 2x2x2 block averaging of n_H (block phase = majority of",
        "                  its 8 cells, >= 5; no majority -> skipped; NEUTRAL = >= 5 non-HIM)",
        "  M2_pdf(b)       (exp(sig_s^2) - 1) / (3 b^2): M3D^2 = (exp(sig_s^2) - 1)/b^2, made 1D",
        "                  by isotropy (/3) to match Fig. 7's M^2 = sigma_1D^2 / c_s^2",
        "  c_s             sqrt(sum(k_B p_th_phys) / sum(rho)), identical to velocity.npz",
        "  sig_turb        sqrt(M2_pdf(b = 0.4)) c_s  [km/s]",
        "  <M^2>_a, sig_nt_a   the alpha method, read from cache/velocity/velocity.npz:",
        "                  <alpha_cell - 1> mass weighted, and sqrt(k_B W/<rho> - c_s^2)",
        "The PDF method is a LOWER limit (resolution and regularisation remove small-scale",
        "density variance), the alpha method an UPPER limit (all of alpha - 1 is motions).",
        f"Statistics are NaN with fewer than {N_MIN_CELLS} cells (blocks) of the sample.",
        "",
        "CROSS-CHECKS (the script raises if any fails)",
        f"  c_s vs velocity.npz: max rel {float(out['check_cs_max_rel'][0]):.2e};  "
        f"<M^2> via phase_sums: max rel {float(out['check_M2_max_rel'][0]):.2e} "
        f"(threshold {CS_IDENTITY_RTOL:.0e})",
        "  cell counts per sample == velocity.npz; phase histograms + under/overflow == cell counts",
        "",
        "-" * 118,
        "SLAB TABLE (+-30 pc slabs)",
        "-" * 118,
        f"{'sample':>8}{'N':>10}{'sig_s':>7}{'rob':>7}{'4pc':>7}{'4pc/s':>7}"
        f"{'M2 b=1/3':>9}{'b=0.4':>7}{'b=1':>7}{'M2rob.4':>8}{'c_s':>7}{'sig_turb':>9}"
        f"{'<M^2>_a':>9}{'sig_nt_a':>9}",
        f"{'':>8}{'':>10}{'':>7}{'':>7}{'':>7}{'':>7}{'':>9}{'':>7}{'':>7}{'':>8}"
        f"{'[km/s]':>7}{'[km/s]':>9}{'':>9}{'[km/s]':>9}",
    ]
    for i, (off, zc) in enumerate(zip(out["slab_offsets_pc"], out["slab_centers_pc"])):
        tag = "z_mid" if off == 0 else f"z_mid {off:+.0f}"
        L.append(f"{tag} (z_c = {zc:+.2f} pc)")
        for k in REPORT_SAMPLES:
            L.append(
                f"{k:>8}{int(g(k, 'n', i)):>10d}{g(k, 'sigma_s', i):>7.3f}"
                f"{g(k, 'sigma_s_robust', i):>7.3f}{g(k, 'sigma_s_4pc', i):>7.3f}"
                f"{g(k, 'sigma_s_4pc', i) / g(k, 'sigma_s', i):>7.3f}"
                f"{g(k, 'M2_pdf_b1_3', i):>9.3f}{g(k, 'M2_pdf_b0p4', i):>7.3f}"
                f"{g(k, 'M2_pdf_b1', i):>7.3f}{g(k, 'M2_pdf_b0p4_rob', i):>8.3f}"
                f"{g(k, 'c_s', i):>7.3f}{g(k, 'sigma_turb_pdf_b0p4', i):>9.3f}"
                f"{vel[f'slab__{k}__M2'][i]:>9.3f}{vel[f'slab__{k}__sigma_nt'][i]:>9.3f}")
        L.append("")
    pl, ph_ = out["percentile_levels"]
    L += [
        "-" * 118,
        f"MIDPLANE SLAB, the two methods side by side (bands: PDF b = 1/3 .. 1; alpha "
        f"mass-weighted p{pl:g}-p{ph_:g})",
        "-" * 118,
        f"{'sample':>8}{'sig_turb_pdf (b=0.4) [1/3..1]':>32}{'sig_nt_alpha [band]':>26}"
        f"{'M2_pdf (b=0.4) [1/3..1]':>28}{'<M^2>_alpha [band]':>24}",
    ]
    for k in REPORT_SAMPLES:
        L.append(
            f"{k:>8}{g(k, 'sigma_turb_pdf_b0p4', i_mid):>12.3f} "
            f"[{g(k, 'sigma_turb_pdf_b1', i_mid):.3f}, {g(k, 'sigma_turb_pdf_b1_3', i_mid):.3f}]"
            f"{vel[f'slab__{k}__sigma_nt'][i_mid]:>12.3f} "
            f"[{g(k, 'sigma_nt_cell_p_lo', i_mid):.3f}, {g(k, 'sigma_nt_cell_p_hi', i_mid):.3f}]"
            f"{g(k, 'M2_pdf_b0p4', i_mid):>12.3f} "
            f"[{g(k, 'M2_pdf_b1', i_mid):.3f}, {g(k, 'M2_pdf_b1_3', i_mid):.3f}]"
            f"{vel[f'slab__{k}__M2'][i_mid]:>10.3f} "
            f"[{vel[f'slab__{k}__M2_cell_p_lo'][i_mid]:.3f}, "
            f"{vel[f'slab__{k}__M2_cell_p_hi'][i_mid]:.3f}]")
    L += [
        "",
        "  sig_nt_alpha band: mass-weighted percentiles of sqrt(max(k_B (W - p_th_phys)/rho, 0)),",
        "  computed here (velocity.npz holds no sigma_nt band); the M^2 band is velocity.npz's.",
        "",
        "-" * 118,
        "s-PDF GROUPS (appendix): union of the 4 pc bins containing z_mid +- d",
        "-" * 118,
        f"{'d [pc]':>8}{'bins z':>20}" + "".join(f"{k + ' sig_s':>15}" for k in REPORT_SAMPLES),
    ]
    for j, d in enumerate(out["pdfz__offsets_pc"]):
        zb = ", ".join(f"{out['z_pc'][b]:+.0f}" for b in sorted(set(out["pdfz__bins"][j])))
        L.append(f"{d:>8.0f}{zb:>20}" + "".join(f"{out[f'pdfz__{k}__sigma_s'][j]:>15.3f}"
                                               for k in REPORT_SAMPLES))
    L += ["", f"Runtime: {runtime:.1f} s"]
    OUT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
