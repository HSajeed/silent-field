"""Noise-floor audit of the k-omega radiating-cone metric R.

Writes figures/JSON into this same directory (runs/noise_floor_check/).
Does not modify anything under the repo's src/, data/, scripts/ trees.

Run from the silent-field project root:
    python runs/noise_floor_check/analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
from scipy.ndimage import uniform_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.spectral import Grid
from src.acoustics.komega import (
    KOmegaSpectrum,
    komega_spectrum,
    welch_komega_spectrum,
    sound_speed,
)
from src.acoustics.radiating import effective_weight_in_cone
from src.acoustics.lighthill import lighthill_source_divergence_form
from src.solvers.io import load_dataset

OUT_DIR = Path(__file__).resolve().parent
MACHS = [0.05, 0.1, 0.2, 0.3]
DATASETS = {
    "baseline": REPO_ROOT / "data" / "turb2d.npz",
    "kf16": REPO_ROOT / "data" / "turb2d_kf16.npz",
}
SEED = 0
FLOOR_MARGIN = 10.0  # "within 10x of floor" / PASS margin threshold

timings: dict[str, float] = {}


# --------------------------------------------------------------------------
# Vector (u,v) k-omega spectra without ever materialising a stacked
# float64 (nt, ny, nx, 2) array: power(u) + power(v) computed and summed
# component-by-component.  Verified numerically identical (<1e-16 max abs
# diff) to calling komega_spectrum on a genuine stacked (u,v) array.
# --------------------------------------------------------------------------
def vector_komega_spectrum(u, v, grid, dt, window="hann", n_k_bins=None,
                            detrend_mean=True) -> KOmegaSpectrum:
    su = komega_spectrum(u, grid, dt, window=window, n_k_bins=n_k_bins,
                          detrend_mean=detrend_mean)
    sv = komega_spectrum(v, grid, dt, window=window, n_k_bins=n_k_bins,
                          detrend_mean=detrend_mean)
    return KOmegaSpectrum(power=su.power + sv.power, k_bins=su.k_bins, omega=su.omega)


def vector_welch_komega_spectrum(u, v, grid, dt, n_segments=4, overlap=0.5,
                                  window="hann", n_k_bins=None) -> KOmegaSpectrum:
    nt = u.shape[0]
    seg = int(nt / (1 + (n_segments - 1) * (1 - overlap)))
    step = max(1, int(seg * (1 - overlap)))
    acc = None
    count = 0
    last = None
    for start in range(0, nt - seg + 1, step):
        s = vector_komega_spectrum(u[start:start + seg], v[start:start + seg],
                                    grid, dt, window=window, n_k_bins=n_k_bins)
        acc = s.power if acc is None else acc + s.power
        count += 1
        last = s
    if acc is None:
        raise ValueError("record too short for the requested segmentation")
    return KOmegaSpectrum(power=acc / count, k_bins=last.k_bins, omega=last.omega)


def nearest_idx(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def k_index(spec: KOmegaSpectrum, k: int) -> int:
    # k_bins are centred on integer multiples of dk=1 for these grids (length=2pi)
    return nearest_idx(spec.k_bins, k)


def local_log_slope(omega, power, centre_idx, half_width=3):
    lo = max(0, centre_idx - half_width)
    hi = min(len(omega), centre_idx + half_width + 1)
    x = omega[lo:hi]
    y = np.log10(np.clip(power[lo:hi], 1e-300, None))
    if len(x) < 2:
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def floor_onset(power, start_idx=0, smooth_window=9, plateau_decades=1.0):
    """First index i (>= start_idx) beyond which the spectrum has plateaued:
    log10(power), smoothed over `smooth_window` bins to suppress single-bin
    Welch-averaging noise, never drops more than `plateau_decades` below its
    value at i for the remainder of the record.  A curve that is still
    decaying by orders of magnitude on its way to Nyquist will fail this test
    at every bin except close to the edge; a curve that has genuinely
    flattened onto a floor triggers it as soon as the floor is reached.
    Validated against the visually obvious Blackman-Harris/DPSS floor onset
    (~omega 30-41) vs. the Hann estimator's much later onset (~omega 67-75,
    i.e. it does not floor before Nyquist)."""
    n = len(power)
    log_p = np.log10(np.clip(power, 1e-300, None))
    sm = uniform_filter1d(log_p, size=smooth_window, mode="nearest")
    for i in range(start_idx, n - 1):
        if sm[i:].min() >= sm[i] - plateau_decades:
            return i
    return None


# --------------------------------------------------------------------------
def analyse_dataset(name: str, path: Path) -> dict:
    t_start = time.time()
    u, v, meta = load_dataset(path)
    grid = Grid(n=meta["n"] if "n" in meta else 128, length=meta["grid_length"])
    dt = meta["dt"]
    nt = u.shape[0]
    U = float(np.sqrt(np.mean(u ** 2 + v ** 2)))
    c0 = {m: sound_speed(U, m) for m in MACHS}

    print(f"[{name}] U={U:.6f}  nt={nt}  dt={dt}  grid.n={grid.n}  length={grid.length:.6f}")

    # ---- production and comparison estimators (all share the same Welch
    # segmentation -> identical omega grid -> direct index comparisons) ----
    t0 = time.time()
    spec_hann4 = vector_welch_komega_spectrum(u, v, grid, dt, n_segments=4, window="hann")
    t_hann4 = time.time() - t0

    t0 = time.time()
    spec_bh4 = vector_welch_komega_spectrum(u, v, grid, dt, n_segments=4, window="blackmanharris")
    t_bh4 = time.time() - t0

    t0 = time.time()
    spec_dpss4 = vector_welch_komega_spectrum(u, v, grid, dt, n_segments=4, window=("dpss", 4))
    t_dpss4 = time.time() - t0

    t0 = time.time()
    spec_hann1 = vector_komega_spectrum(u, v, grid, dt, window="hann")
    t_hann1 = time.time() - t0

    # ---- float32 storage-rounding floor ----
    rng = np.random.default_rng(SEED)
    u32 = u.astype(np.float32)
    v32 = v.astype(np.float32)
    eps_u = np.spacing(u32).astype(np.float64)
    eps_v = np.spacing(v32).astype(np.float64)
    delta_u = rng.uniform(-0.5, 0.5, size=u.shape) * eps_u
    delta_v = rng.uniform(-0.5, 0.5, size=v.shape) * eps_v

    t0 = time.time()
    spec_floor = vector_welch_komega_spectrum(delta_u, delta_v, grid, dt, n_segments=4, window="hann")
    t_floor = time.time() - t0

    t0 = time.time()
    spec_hann4_plus_floor = vector_welch_komega_spectrum(
        u + delta_u, v + delta_v, grid, dt, n_segments=4, window="hann"
    )
    t_plusfloor = time.time() - t0

    # ---- aliasing check: subsample time by 2 ----
    u_sub, v_sub, dt_sub = u[::2], v[::2], dt * 2.0
    t0 = time.time()
    spec_sub = vector_welch_komega_spectrum(u_sub, v_sub, grid, dt_sub, n_segments=4, window="hann")
    t_sub = time.time() - t0

    timings[f"{name}_hann4"] = t_hann4
    timings[f"{name}_bh4"] = t_bh4
    timings[f"{name}_dpss4"] = t_dpss4
    timings[f"{name}_hann1_full"] = t_hann1
    timings[f"{name}_float32floor"] = t_floor
    timings[f"{name}_plusfloor"] = t_plusfloor
    timings[f"{name}_subsampled"] = t_sub

    # ================================================================
    # CHECK A.1 / A.2 -- plot k=1,2,3 omega-profiles, all estimators + floor
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    estimators = [
        ("Hann 4-seg (production)", spec_hann4, "C0", "-"),
        ("Blackman-Harris 4-seg", spec_bh4, "C1", "-"),
        ("DPSS NW=4, 4-seg", spec_dpss4, "C2", "-"),
        ("Hann single-seg (full record)", spec_hann1, "C3", "-"),
        ("float32 rounding floor (Hann 4-seg)", spec_floor, "k", "--"),
    ]
    mach_colors = plt.cm.autumn(np.linspace(0.1, 0.8, len(MACHS)))
    for ax, k in zip(axes, [1, 2, 3]):
        for label, spec, color, ls in estimators:
            ik = k_index(spec, k)
            p = np.clip(spec.power[:, ik], 1e-300, None)
            ax.plot(spec.omega, np.log10(p), color=color, linestyle=ls, label=label, lw=1.3)
        for mcolor, m in zip(mach_colors, MACHS):
            ax.axvline(c0[m] * k, color=mcolor, linestyle=":", lw=1.2,
                       label=f"c0*k, M={m}")
        ax.set_title(f"{name}: k={k}")
        ax.set_xlabel("omega")
        ax.set_ylabel("log10 P(omega)")
        ax.set_xlim(0, max(spec_hann4.omega.max(), spec_hann1.omega.max()))
    axes[0].legend(fontsize=6, loc="upper right", ncol=1)
    fig.tight_layout()
    fig_path = OUT_DIR / f"{name}_komega_profiles.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ================================================================
    # R and R with float32 rounding noise added
    # ================================================================
    R = {m: effective_weight_in_cone(spec_hann4, c0[m], "l2") for m in MACHS}
    R_plusfloor = {m: effective_weight_in_cone(spec_hann4_plus_floor, c0[m], "l2") for m in MACHS}
    R_rel_change = {m: (R_plusfloor[m] - R[m]) / R[m] if R[m] != 0 else float("nan") for m in MACHS}

    # ================================================================
    # CHECK A.3 -- leakage: P_BH/P_Hann and P_DPSS/P_Hann at omega=c0*k
    # (hann4, bh4, dpss4 share the same omega grid by construction)
    # ================================================================
    leakage_rows = []
    for m in MACHS:
        for k in (1, 2):
            ik = k_index(spec_hann4, k)
            target_omega = c0[m] * k
            iw = nearest_idx(spec_hann4.omega, target_omega)
            p_h = spec_hann4.power[iw, ik]
            p_b = spec_bh4.power[iw, ik]
            p_d = spec_dpss4.power[iw, ik]
            leakage_rows.append({
                "mach": m, "k": k, "omega_bin": float(spec_hann4.omega[iw]),
                "omega_target": float(target_omega),
                "P_hann": float(p_h), "P_BH": float(p_b), "P_DPSS": float(p_d),
                "P_BH_over_P_hann": float(p_b / p_h) if p_h > 0 else float("nan"),
                "P_DPSS_over_P_hann": float(p_d / p_h) if p_h > 0 else float("nan"),
            })

    # ================================================================
    # CHECK A.4 -- aliasing: subsampled (dt*2) vs full-rate, k=1, omega in [5,35]
    # ================================================================
    ik1 = k_index(spec_hann4, 1)
    ik1_sub = k_index(spec_sub, 1)
    band = (spec_sub.omega >= 5) & (spec_sub.omega <= 35)
    omega_band = spec_sub.omega[band]
    p_sub_band = spec_sub.power[band, ik1_sub]
    p_full_interp = np.interp(omega_band, spec_hann4.omega, spec_hann4.power[:, ik1])
    ratio = p_sub_band / np.clip(p_full_interp, 1e-300, None)
    max_dev_idx = int(np.argmax(np.abs(ratio - 1.0)))
    aliasing = {
        "omega_band": [5.0, 35.0],
        "n_points_compared": int(band.sum()),
        "max_abs_ratio_minus_1": float(np.max(np.abs(ratio - 1.0))),
        "ratio_at_max_dev": float(ratio[max_dev_idx]),
        "omega_at_max_dev": float(omega_band[max_dev_idx]),
        "new_nyquist_sub": float(np.pi / dt_sub),
        "full_nyquist": float(np.pi / dt),
    }

    # ================================================================
    # CHECK A.5 -- local log-slope at threshold, near Nyquist; floor onset
    # ================================================================
    slope_rows = []
    onset_by_k = {}
    for k in (1, 2, 3):
        ik = k_index(spec_hann4, k)
        power_k = spec_hann4.power[:, ik]
        peak_idx = int(np.argmax(power_k))
        onset_idx = floor_onset(power_k, start_idx=peak_idx)
        onset_by_k[k] = {
            "onset_idx": onset_idx,
            "onset_omega": float(spec_hann4.omega[onset_idx]) if onset_idx is not None else None,
            "onset_power": float(power_k[onset_idx]) if onset_idx is not None else None,
        }
        # near-Nyquist slope (last 7 bins, centred 3 from the end)
        nyq_centre = len(spec_hann4.omega) - 4
        slope_nyq = local_log_slope(spec_hann4.omega, power_k, nyq_centre, half_width=3)
        for m in MACHS:
            iw = nearest_idx(spec_hann4.omega, c0[m] * k)
            slope_thresh = local_log_slope(spec_hann4.omega, power_k, iw, half_width=3)
            slope_rows.append({
                "mach": m, "k": k,
                "omega_threshold_bin": float(spec_hann4.omega[iw]),
                "slope_at_threshold": slope_thresh,
                "slope_near_nyquist": slope_nyq,
                "still_decaying_at_threshold": bool(slope_thresh < 0),
            })

    # ================================================================
    # CHECK A.6 -- margins vs floor and vs onset; cone-energy-near-floor fraction
    # ================================================================
    margin_rows = []
    for m in MACHS:
        for k in (1, 2, 3):
            ik = k_index(spec_hann4, k)
            iw = nearest_idx(spec_hann4.omega, c0[m] * k)
            p_thresh = spec_hann4.power[iw, ik]
            p_floor = spec_floor.power[iw, ik]
            onset_idx = onset_by_k[k]["onset_idx"]
            p_onset = spec_hann4.power[onset_idx, ik] if onset_idx is not None else None
            margin_rows.append({
                "mach": m, "k": k,
                "P_at_threshold": float(p_thresh),
                "P_float32floor_at_threshold": float(p_floor),
                "margin_over_float32floor": float(p_thresh / p_floor) if p_floor > 0 else float("inf"),
                "P_at_onset": float(p_onset) if p_onset is not None else None,
                "margin_over_onset_level": (float(p_thresh / p_onset)
                                             if (p_onset is not None and p_onset > 0) else None),
                "beyond_onset": (bool(iw >= onset_idx) if onset_idx is not None else False),
            })

    cone_floor_fraction = {}
    for m in MACHS:
        mask_cone = spec_hann4.radiating_mask(c0[m])
        cone_energy = spec_hann4.power[mask_cone].sum()
        near_floor = spec_hann4.power <= FLOOR_MARGIN * spec_floor.power
        frac = (spec_hann4.power[mask_cone & near_floor].sum() / cone_energy
                if cone_energy > 0 else float("nan"))
        cone_floor_fraction[m] = float(frac)

    # ================================================================
    # CHECK A verdict, k=1 (headline), plus k=2,3 flagged
    # ================================================================
    verdicts = {}
    for k in (1, 2, 3):
        verdicts[k] = {}
        onset_idx = onset_by_k[k]["onset_idx"]
        for m in MACHS:
            row = next(r for r in margin_rows if r["mach"] == m and r["k"] == k)
            srow = next(r for r in slope_rows if r["mach"] == m and r["k"] == k)
            pass_floor = row["margin_over_float32floor"] >= FLOOR_MARGIN
            still_decaying = srow["still_decaying_at_threshold"] and not row["beyond_onset"]
            verdict = "PASS" if (pass_floor and still_decaying) else "FAIL"
            reasons = []
            if not pass_floor:
                reasons.append(f"margin_over_float32floor={row['margin_over_float32floor']:.3g} < {FLOOR_MARGIN}")
            if not still_decaying:
                reasons.append("spectrum has reached floor-onset / is not decaying at threshold")
            omega_threshold = c0[m] * k
            resolved = omega_threshold <= spec_hann4.omega.max()
            verdicts[k][m] = {
                "verdict": verdict if resolved else "UNRESOLVED",
                "reasons": reasons,
                "omega_threshold": float(omega_threshold),
                "resolved_within_nyquist": bool(resolved),
            }

    result = {
        "U": U,
        "c0": c0,
        "nt": nt, "dt": dt, "grid_n": grid.n, "grid_length": grid.length,
        "R": R,
        "R_plus_float32_delta": R_plusfloor,
        "R_relative_change_from_delta": R_rel_change,
        "leakage": leakage_rows,
        "aliasing": aliasing,
        "slopes": slope_rows,
        "floor_onset_by_k": onset_by_k,
        "margins": margin_rows,
        "cone_energy_fraction_near_float32_floor": cone_floor_fraction,
        "check_A_verdicts": verdicts,
        "figure": str(fig_path),
        "spectra_for_checkB": {  # stashed for check B below, not serialised
            "_spec_hann4": spec_hann4,
        },
    }
    timings[f"{name}_total_checkA"] = time.time() - t_start
    return result


def analyse_checkB(name: str, u, v, grid, dt, U, spec_hann4: KOmegaSpectrum,
                    checkA_verdicts: dict) -> dict:
    c0 = {m: sound_speed(U, m) for m in MACHS}
    k_bins = spec_hann4.k_bins
    total_all = spec_hann4.power.sum()

    table = []
    R_all_by_m, R_k2_by_m, R_k3_by_m = {}, {}, {}
    for m in MACHS:
        mask_cone = spec_hann4.radiating_mask(c0[m])
        kmask2 = k_bins >= 2
        kmask3 = k_bins >= 3
        total_k2 = spec_hann4.power[:, kmask2].sum()
        total_k3 = spec_hann4.power[:, kmask3].sum()
        cone_all = spec_hann4.power[mask_cone].sum()
        cone_k2 = spec_hann4.power[mask_cone & kmask2[None, :]].sum()
        cone_k3 = spec_hann4.power[mask_cone & kmask3[None, :]].sum()

        R_all = cone_all / total_all
        R_k2 = cone_k2 / total_all
        R_k2_given = cone_k2 / total_k2 if total_k2 > 0 else float("nan")
        R_k3 = cone_k3 / total_all
        R_k3_given = cone_k3 / total_k3 if total_k3 > 0 else float("nan")

        R_all_by_m[m] = R_all
        R_k2_by_m[m] = R_k2
        R_k3_by_m[m] = R_k3

        table.append({
            "mach": m,
            "R_all_k": R_all,
            "R_kge2": R_k2,
            "R_kge2_given_kge2": R_k2_given,
            "R_kge3": R_k3,
            "R_kge3_given_kge3": R_k3_given,
            "checkA_pass_k2": checkA_verdicts[2][m]["verdict"],
            "checkA_pass_k3": checkA_verdicts[3][m]["verdict"],
        })

    def exponents(vals_by_m):
        rows = []
        machs_sorted = sorted(MACHS)
        for i in range(len(machs_sorted) - 1):
            m0, m1 = machs_sorted[i], machs_sorted[i + 1]
            v0, v1 = vals_by_m[m0], vals_by_m[m1]
            if v0 > 0 and v1 > 0:
                slope = (np.log(v1) - np.log(v0)) / (np.log(m1) - np.log(m0))
            else:
                slope = float("nan")
            rows.append({"mach_low": m0, "mach_high": m1, "d_lnR_d_lnM": float(slope)})
        return rows

    exps = {
        "R_all_k": exponents(R_all_by_m),
        "R_kge2": exponents(R_k2_by_m),
        "R_kge3": exponents(R_k3_by_m),
    }

    out = {"table": table, "exponents": exps}

    if name == "baseline":
        nt = u.shape[0]
        half = nt // 2
        spec_first = vector_welch_komega_spectrum(u[:half], v[:half], grid, dt,
                                                    n_segments=2, window="hann")
        spec_second = vector_welch_komega_spectrum(u[half:], v[half:], grid, dt,
                                                     n_segments=2, window="hann")
        m = 0.1
        c0_m = sound_speed(U, m)

        def r_kge2(spec):
            mask_cone = spec.radiating_mask(c0_m)
            kmask2 = spec.k_bins >= 2
            cone_k2 = spec.power[mask_cone & kmask2[None, :]].sum()
            total_all = spec.power.sum()
            return float(cone_k2 / total_all) if total_all > 0 else float("nan")

        out["half_record_band_M0.1_R_kge2"] = {
            "first_half": r_kge2(spec_first),
            "second_half": r_kge2(spec_second),
        }

        # Lighthill source R, k>=2, baseline only, divergence form, frame-by-frame,
        # float32 storage
        t0 = time.time()
        S = np.empty((nt, grid.n, grid.n), dtype=np.float32)
        for t in range(nt):
            S[t] = lighthill_source_divergence_form(u[t], v[t], grid).astype(np.float32)
        spec_S = welch_komega_spectrum(S, grid, dt, n_segments=4, window="hann")
        t_lighthill = time.time() - t0
        timings["baseline_lighthill_S"] = t_lighthill

        total_S_all = spec_S.power.sum()
        kmask2_S = spec_S.k_bins >= 2
        total_S_k2 = spec_S.power[:, kmask2_S].sum()
        lighthill_rows = []
        for m2 in MACHS:
            mask_cone = spec_S.radiating_mask(c0[m2])
            cone_k2 = spec_S.power[mask_cone & kmask2_S[None, :]].sum()
            lighthill_rows.append({
                "mach": m2,
                "R_S_kge2_over_total_all": float(cone_k2 / total_S_all) if total_S_all > 0 else float("nan"),
                "R_S_kge2_over_total_kge2": float(cone_k2 / total_S_k2) if total_S_k2 > 0 else float("nan"),
            })
        out["lighthill_S_kge2"] = lighthill_rows

    return out


def main():
    t_all = time.time()
    full_results = {}
    for name, path in DATASETS.items():
        res = analyse_dataset(name, path)
        u, v, meta = load_dataset(path)
        grid = Grid(n=res["grid_n"], length=res["grid_length"])
        spec_hann4 = res["spectra_for_checkB"]["_spec_hann4"]
        checkB = analyse_checkB(name, u, v, grid, res["dt"], res["U"], spec_hann4,
                                 res["check_A_verdicts"])
        res.pop("spectra_for_checkB")
        res["check_B"] = checkB
        full_results[name] = res
        del u, v

    timings["total_runtime"] = time.time() - t_all
    full_results["_timings_seconds"] = timings

    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(full_results, indent=2, default=float))
    print(f"\nWrote {json_path}")
    print(f"Total runtime: {timings['total_runtime']:.1f} s")


if __name__ == "__main__":
    main()
