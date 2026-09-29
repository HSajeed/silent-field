"""Verification + analysis of the L=4pi, n=256 turbulence box.

Writes JSON/PNGs into this directory (runs/box_L4pi/) only. Does not modify
anything under the repo's src/, data/, scripts/ trees.

Run from the silent-field project root:
    python runs/box_L4pi/analysis.py
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import get_window

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.solvers.turb2d import acoustic_sampling_interval
from src.acoustics.komega import (
    KOmegaSpectrum,
    komega_spectrum,
    welch_komega_spectrum,
    sound_speed,
)
from src.acoustics.lighthill import lighthill_source_divergence_form
from src.diagnostics.spectra import time_averaged_energy_spectrum

OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "turb2d_L4pi.npz"
MANIFEST_PATH = REPO_ROOT / "data" / "turb2d_L4pi.manifest.json"
GEN_SCRIPT_PATH = REPO_ROOT / "scripts" / "01_generate_data.py"

MACHS = [0.05, 0.1, 0.2, 0.3]
MATCHED_C0 = 11.308
FLOOR_MARGIN_THRESHOLD = 10.0  # our chosen ">>floor" operationalisation
AGREEMENT_RATIO_THRESHOLD = 2.0
SEED = 0

timings: dict[str, float] = {}


def mem_report(label: str) -> None:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    print(f"--- free -m @ {label} ---\n{out}")


def peak_rss_mb() -> float:
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ============================================================================
# Vector (u,v) k-omega spectra without materialising a stacked float64 array:
# power(u) + power(v) computed and summed component-by-component.
# ============================================================================
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


def k_index(spec: KOmegaSpectrum, k: float) -> int:
    return nearest_idx(spec.k_bins, k)


# ============================================================================
# R metrics on a KOmegaSpectrum
# ============================================================================
def R_all(spec: KOmegaSpectrum, c0: float) -> float:
    total = spec.power.sum()
    if total <= 0:
        return float("nan")
    return float(spec.power[spec.radiating_mask(c0)].sum() / total)


def R_kge(spec: KOmegaSpectrum, c0: float, kmin: float):
    """Return (cone_kge/total_all, cone_kge/total_kge)."""
    mask = spec.radiating_mask(c0)
    kmask = spec.k_bins >= kmin
    total_all = spec.power.sum()
    total_kge = spec.power[:, kmask].sum()
    cone_kge = spec.power[mask & kmask[None, :]].sum()
    r_over_all = float(cone_kge / total_all) if total_all > 0 else float("nan")
    r_given = float(cone_kge / total_kge) if total_kge > 0 else float("nan")
    return r_over_all, r_given


def per_bin_cone_share(spec: KOmegaSpectrum, c0: float):
    mask = spec.radiating_mask(c0)
    cone_by_bin = spec.power * mask
    per_bin = cone_by_bin.sum(axis=0)
    total_cone = per_bin.sum()
    share = per_bin / total_cone if total_cone > 0 else np.zeros_like(per_bin)
    return share


def local_exponents(machs, values):
    machs = sorted(machs)
    rows = []
    for i in range(len(machs) - 1):
        m0, m1 = machs[i], machs[i + 1]
        v0, v1 = values[m0], values[m1]
        slope = (np.log(v1) - np.log(v0)) / (np.log(m1) - np.log(m0)) if v0 > 0 and v1 > 0 else float("nan")
        rows.append({"mach_low": m0, "mach_high": m1, "d_lnR_d_lnM": float(slope)})
    return rows


# ============================================================================
# Section 3 -- memory-efficient low-k (|k|<=3) mode extraction & manual Welch
# ============================================================================
def low_k_indices(grid: Grid, k_cut: float = 3.0):
    kx, ky = grid.wavenumbers()
    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    mask = k_mag <= k_cut
    rows, cols = np.nonzero(mask)
    return rows, cols, k_mag[rows, cols]


def extract_low_k_modes(u32: np.ndarray, v32: np.ndarray, rows, cols):
    """Exact low-k Fourier-coefficient time series, one fft2 per frame."""
    nt = u32.shape[0]
    n_modes = rows.size
    modes_u = np.empty((nt, n_modes), dtype=np.complex128)
    modes_v = np.empty((nt, n_modes), dtype=np.complex128)
    for t in range(nt):
        fu = np.fft.fft2(u32[t].astype(np.float64))
        fv = np.fft.fft2(v32[t].astype(np.float64))
        modes_u[t] = fu[rows, cols]
        modes_v[t] = fv[rows, cols]
    return modes_u, modes_v


def extract_low_k_floor_modes(u32: np.ndarray, v32: np.ndarray, rows, cols, rng):
    """float32-rounding-noise modes: delta = U(-0.5,0.5) * spacing(float32(field))."""
    nt = u32.shape[0]
    n_modes = rows.size
    modes_u = np.empty((nt, n_modes), dtype=np.complex128)
    modes_v = np.empty((nt, n_modes), dtype=np.complex128)
    for t in range(nt):
        eps_u = np.spacing(u32[t]).astype(np.float64)
        eps_v = np.spacing(v32[t]).astype(np.float64)
        du = rng.uniform(-0.5, 0.5, size=u32[t].shape) * eps_u
        dv = rng.uniform(-0.5, 0.5, size=v32[t].shape) * eps_v
        fu = np.fft.fft2(du)
        fv = np.fft.fft2(dv)
        modes_u[t] = fu[rows, cols]
        modes_v[t] = fv[rows, cols]
    return modes_u, modes_v


def welch_profile_from_modes(mode_components, k_mag_modes, grid_n, dt, window,
                              n_segments, overlap, dk, n_k_bins):
    """Manual Welch (k,omega) profile restricted to the modes provided.

    mode_components : list of (nt, n_modes) complex arrays (one per field
                       component; their |.|^2 is summed, matching
                       komega_spectrum's component summation).
    Returns (KOmegaSpectrum, domega) with domega = 2*pi/(seg*dt) the bin
    width of this estimator (constant across the omega axis).
    """
    nt = mode_components[0].shape[0]
    n_modes = mode_components[0].shape[1]
    if n_segments > 1:
        seg = int(nt / (1 + (n_segments - 1) * (1 - overlap)))
        step = max(1, int(seg * (1 - overlap)))
        starts = list(range(0, nt - seg + 1, step))
    else:
        seg = nt
        starts = [0]

    edges = (np.arange(n_k_bins + 1) - 0.5) * dk
    edges[0] = 0.0
    centres = np.arange(n_k_bins) * dk
    idx_bin = np.digitize(k_mag_modes, edges) - 1
    valid = (idx_bin >= 0) & (idx_bin < n_k_bins)
    idxv = idx_bin[valid]

    w = get_window(window, seg, fftbins=True).astype(np.float64)
    w_power = np.mean(w ** 2)

    n_pos = seg // 2 + 1
    omega_all = 2.0 * np.pi * np.fft.fftfreq(seg, d=dt)
    omega = np.abs(omega_all[:n_pos])
    omega[0] = 0.0

    acc = None
    count = 0
    for start in starts:
        power_modes = np.zeros((seg, n_modes))
        for comp in mode_components:
            seg_data = comp[start:start + seg].copy()
            seg_data -= seg_data.mean(axis=0, keepdims=True)
            seg_data *= w[:, None]
            spec = np.fft.fft(seg_data, axis=0)
            power_modes += np.abs(spec) ** 2
        n_total = seg * grid_n * grid_n
        power_modes /= (n_total ** 2) * w_power
        power_modes = power_modes[:n_pos]
        if seg > 1:
            power_modes[1: n_pos - (seg % 2 == 0)] *= 2.0

        flat = power_modes[:, valid]
        binned = np.zeros((n_pos, n_k_bins))
        for b in range(n_k_bins):
            sel = idxv == b
            if sel.any():
                binned[:, b] = flat[:, sel].sum(axis=1)
        acc = binned if acc is None else acc + binned
        count += 1

    domega = 2.0 * np.pi / (seg * dt)
    return KOmegaSpectrum(power=acc / count, k_bins=centres, omega=omega), domega


def density(spec: KOmegaSpectrum, domega: float) -> np.ndarray:
    return spec.power / domega


def main():
    t_all = time.time()
    result: dict = {}
    mem_report("start")

    # ========================================================================
    # 0. Verify the dataset
    # ========================================================================
    manifest = json.loads(MANIFEST_PATH.read_text())
    cfg = manifest["config"]
    extra = manifest["extra"]

    verify = {"manifest": manifest}
    checks = {
        "n_256": cfg["n"] == 256,
        "length_4pi": abs(cfg["length"] - 4 * np.pi) < 1e-9,
        "viscosity_5e-4": abs(cfg["viscosity"] - 5e-4) < 1e-12,
        "drag_0.08": abs(cfg["drag"] - 0.08) < 1e-12,
        "forcing_k_8": abs(cfg["forcing_k"] - 8.0) < 1e-12,
        "forcing_kind_kolmogorov": cfg["forcing_kind"] == "kolmogorov",
        "dt_2e-3": abs(cfg["dt"] - 2e-3) < 1e-12,
        "dt_save_0.04": abs(extra.get("dt_save", -1) - 0.04) < 1e-12,
    }
    verify["config_checks"] = checks
    verify["all_config_checks_pass"] = bool(all(checks.values()))

    # spin-up: not part of Turb2DConfig / meta schema -> cannot be recovered
    # from the manifest or dataset regardless of what was actually run.
    verify["spinup_steps_recorded"] = "spinup" in extra or "spinup_steps" in cfg
    verify["spinup_note"] = (
        "Turb2DConfig has no spinup field and scripts/01_generate_data.py's "
        "`meta` dict passed to save_dataset/write_manifest never includes "
        "args.spinup, so the number of spin-up steps actually used cannot be "
        "recovered from data/turb2d_L4pi.npz or its manifest under any "
        "circumstance -- this is a gap in the save_dataset/write_manifest "
        "schema, not something specific to this run. No process log "
        "analogous to rerun_log_2026-09-15.md (which covers only the L=2pi "
        "n=128 baseline and the kf16 run) was found for this L=4pi n=256 "
        "dataset. The generation script's --spinup default is 5000; the "
        "task brief states the intended value was 10000; which was actually "
        "used is UNVERIFIABLE from the artifacts alone."
    )

    # --- --length flag: additive, default 2pi, passed into Turb2DConfig?
    gen_src = GEN_SCRIPT_PATH.read_text()
    length_flag_additive = '"--length"' in gen_src and "default=2.0 * np.pi" in gen_src
    length_passed_to_cfg = "length=args.length" in gen_src
    verify["length_flag"] = {
        "flag_present": '"--length"' in gen_src,
        "default_is_2pi": "default=2.0 * np.pi" in gen_src,
        "passed_into_Turb2DConfig": length_passed_to_cfg,
        "other_flags_unchanged": True,  # confirmed by reading the file; see note
        "note": (
            "scripts/01_generate_data.py:23 adds `--length` with "
            "type=float, default=2.0*np.pi (additive, default unchanged from "
            "the implicit 2pi the script always used); scripts/01_generate_"
            "data.py:35-36 passes it straight through as "
            "Turb2DConfig(length=args.length, ...) alongside the other "
            "existing flags, none of which were altered. Read-only "
            "confirmation; file not modified by this analysis."
        ),
    }

    # --- load dataset (float64 transiently, per task instructions) then cast
    t0 = time.time()
    u64, v64, meta = load_dataset(DATA_PATH)
    u = u64.astype(np.float32)
    v = v64.astype(np.float32)
    del u64, v64
    timings["load_and_cast"] = time.time() - t0

    grid = Grid(n=meta["n"], length=meta["grid_length"])
    dt = meta["dt"]
    nt = u.shape[0]

    verify["dataset_shape"] = list(u.shape)
    verify["n_snapshots_1024"] = nt == 1024
    verify["grid_length_matches_manifest"] = abs(meta["grid_length"] - cfg["length"]) < 1e-9
    verify["grid_length_matches_4pi"] = abs(meta["grid_length"] - 4 * np.pi) < 1e-9

    nan_u = bool(np.isnan(u).any())
    nan_v = bool(np.isnan(v).any())
    inf_u = bool(np.isinf(u).any())
    inf_v = bool(np.isinf(v).any())
    verify["nan_inf"] = {"nan_u": nan_u, "nan_v": nan_v, "inf_u": inf_u, "inf_v": inf_v,
                          "any_nan_or_inf": nan_u or nan_v or inf_u or inf_v}

    U = float(np.sqrt(np.mean(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)))
    dt_max = acoustic_sampling_interval(U, grid, 0.1)
    verify["acoustic_sampling_bound"] = {
        "U_recomputed": U,
        "U_manifest": extra.get("u_rms"),
        "mach": 0.1,
        "dt_max": dt_max,
        "dt_save": dt,
        "oversampling_factor": dt_max / dt,
        "ok": bool(dt <= dt_max),
    }

    print(f"U={U:.6f}  nt={nt}  dt={dt}  grid.n={grid.n}  length={grid.length:.6f}")
    print(f"acoustic dt_max(M=0.1)={dt_max:.5f}  dt_save={dt:.5f}  oversampled by {dt_max/dt:.2f}x")
    result["verification"] = verify

    # ========================================================================
    # 1. Stationarity
    # ========================================================================
    t0 = time.time()
    ke = 0.5 * (u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2).mean(axis=(1, 2))
    ke_mean = float(ke.mean())
    ke_std = float(ke.std())
    cv = ke_std / ke_mean

    t_axis = np.arange(nt) * dt
    slope, intercept = np.polyfit(t_axis, ke, 1)
    drift_pct = float((slope * t_axis[-1]) / ke_mean * 100.0)

    ke_fluc = ke - ke_mean
    acf_raw = np.correlate(ke_fluc, ke_fluc, mode="full")[nt - 1:]
    acf = acf_raw / acf_raw[0]
    below = np.where(acf <= 1.0 / np.e)[0]
    if below.size > 0:
        lag0 = below[0]
        efold_lag = float(lag0)
    else:
        lag0 = None
        efold_lag = float("nan")
    tau_corr = efold_lag * dt if lag0 is not None else float("nan")
    n_corr_times = (nt * dt) / tau_corr if tau_corr and not np.isnan(tau_corr) else float("nan")

    k_avg, e_avg, n_used = time_averaged_energy_spectrum(u, v, grid, stride=16)
    total_e = e_avg.sum()
    share_below1 = float(e_avg[k_avg < 1.0].sum() / total_e)
    peak_k = float(k_avg[np.argmax(e_avg)])
    idx05 = nearest_idx(k_avg, 0.5)
    share_k05 = float(e_avg[idx05] / total_e)
    idx10 = nearest_idx(k_avg, 1.0)
    idx15 = nearest_idx(k_avg, 1.5)
    condensation_flag = bool(peak_k <= 0.5 or (e_avg[idx05] > e_avg[idx10] and e_avg[idx05] > e_avg[idx15]))

    stationarity = {
        "KE_mean": ke_mean, "KE_std": ke_std, "KE_CV": cv,
        "KE_min": float(ke.min()), "KE_max": float(ke.max()),
        "linear_drift_pct_of_mean": drift_pct,
        "autocorr_efold_lag_steps": lag0, "autocorr_efold_time": tau_corr,
        "record_length": nt * dt, "n_correlation_times": n_corr_times,
        "energy_spectrum_n_snapshots_averaged": n_used,
        "share_KE_k_below_1": share_below1,
        "share_KE_k_eq_0.5": share_k05,
        "peak_k": peak_k,
        "condensation_flag_at_k=0.5": condensation_flag,
        "E_k_first_bins": {"k": k_avg[:8].tolist(), "E": e_avg[:8].tolist()},
    }
    timings["stationarity"] = time.time() - t0
    result["stationarity"] = stationarity
    print(f"KE CV={cv:.4f} drift={drift_pct:.2f}% tau_corr={tau_corr:.3f} "
          f"n_corr_times={n_corr_times:.2f} peak_k={peak_k} share(k<1)={share_below1:.4f}")

    # figure: KE(t), ACF, E(k)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(t_axis, ke)
    axes[0].axhline(ke_mean, color="r", ls="--", lw=1)
    axes[0].set_xlabel("time"); axes[0].set_ylabel("KE"); axes[0].set_title("KE(t)")
    lags = np.arange(min(300, nt)) * dt
    axes[1].plot(lags, acf[:len(lags)])
    axes[1].axhline(1 / np.e, color="r", ls="--", lw=1)
    axes[1].set_xlabel("lag (time)"); axes[1].set_ylabel("ACF"); axes[1].set_title("KE autocorrelation")
    axes[2].loglog(k_avg[1:], e_avg[1:], marker="o", ms=3)
    axes[2].axvline(0.5, color="orange", ls=":", label="k=0.5")
    axes[2].axvline(peak_k, color="g", ls="--", label=f"peak k={peak_k}")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("E(k)"); axes[2].set_title("time-avg E(k)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    stat_fig_path = OUT_DIR / "box_L4pi_stationarity.png"
    fig.savefig(stat_fig_path, dpi=130)
    plt.close(fig)
    result["stationarity"]["figure"] = str(stat_fig_path)
    mem_report("after section 1")

    # ========================================================================
    # 2. R (production estimator)
    # ========================================================================
    t0 = time.time()
    spec_hann4 = vector_welch_komega_spectrum(u, v, grid, dt, n_segments=4, window="hann")
    timings["section2_velocity_welch4"] = time.time() - t0
    mem_report("after velocity welch4")

    c0_by_m = {m: sound_speed(U, m) for m in MACHS}

    rows = []
    for m in MACHS:
        c0 = c0_by_m[m]
        r_all = R_all(spec_hann4, c0)
        r1_all, r1_given = R_kge(spec_hann4, c0, 1.0)
        r2_all, r2_given = R_kge(spec_hann4, c0, 2.0)
        share = per_bin_cone_share(spec_hann4, c0)
        share_requested = {k: float(share[k_index(spec_hann4, k)]) for k in (0.5, 1.0, 1.5, 2.0)}
        top4_idx = np.argsort(share)[::-1][:4]
        top4 = [{"k": float(spec_hann4.k_bins[i]), "share": float(share[i])} for i in top4_idx]
        rows.append({
            "mach": m, "c0": c0,
            "R_all_k": r_all,
            "R_kge1_over_total_all": r1_all, "R_kge1_given_kge1": r1_given,
            "R_kge2_over_total_all": r2_all, "R_kge2_given_kge2": r2_given,
            "per_bin_cone_share_requested": share_requested,
            "top4_bins_by_cone_share": top4,
        })
    result_R = {"table": rows}

    # matched c0
    r_all_matched = R_all(spec_hann4, MATCHED_C0)
    _, r1_given_matched = R_kge(spec_hann4, MATCHED_C0, 1.0)
    _, r2_given_matched = R_kge(spec_hann4, MATCHED_C0, 2.0)
    result_R["matched_c0"] = {
        "c0": MATCHED_C0,
        "R_all_k": r_all_matched,
        "R_kge1_given_kge1": r1_given_matched,
        "R_kge2_given_kge2": r2_given_matched,
    }

    # local exponents
    R_all_by_m = {m: rows[i]["R_all_k"] for i, m in enumerate(MACHS)}
    R_kge1_given_by_m = {m: rows[i]["R_kge1_given_kge1"] for i, m in enumerate(MACHS)}
    result_R["local_exponents"] = {
        "R_all_k": local_exponents(MACHS, R_all_by_m),
        "R_kge1_given_kge1": local_exponents(MACHS, R_kge1_given_by_m),
    }

    # half-record split at M=0.1
    t0 = time.time()
    half = nt // 2
    spec_half1 = vector_welch_komega_spectrum(u[:half], v[:half], grid, dt, n_segments=2, window="hann")
    spec_half2 = vector_welch_komega_spectrum(u[half:], v[half:], grid, dt, n_segments=2, window="hann")
    timings["section2_half_record_welch2"] = time.time() - t0
    c0_01 = c0_by_m[0.1]

    def band_vals(spec_list, fn):
        vals = [fn(s) for s in spec_list]
        return {"values": vals, "band": [min(vals), max(vals)], "spread": (max(vals) / min(vals)) if min(vals) > 0 else float("inf")}

    result_R["half_record_split_M0.1"] = {
        "R_all_k": band_vals([spec_hann4, spec_half1, spec_half2], lambda s: R_all(s, c0_01)),
        "R_kge1_given_kge1": band_vals([spec_hann4, spec_half1, spec_half2], lambda s: R_kge(s, c0_01, 1.0)[1]),
    }

    # Lighthill source S
    t0 = time.time()
    S = np.empty((nt, grid.n, grid.n), dtype=np.float32)
    for t in range(nt):
        S[t] = lighthill_source_divergence_form(u[t], v[t], grid).astype(np.float32)
    timings["section2_lighthill_source_compute"] = time.time() - t0
    mem_report("after Lighthill source computed")

    t0 = time.time()
    spec_S = welch_komega_spectrum(S, grid, dt, n_segments=4, window="hann")
    timings["section2_lighthill_welch4"] = time.time() - t0
    del S
    mem_report("after Lighthill welch4")

    src_rows = []
    for m in MACHS:
        c0 = c0_by_m[m]
        r_all = R_all(spec_S, c0)
        r1_all, r1_given = R_kge(spec_S, c0, 1.0)
        src_rows.append({
            "mach": m, "c0": c0,
            "R_S_all_k": r_all,
            "R_S_kge1_over_total_all": r1_all,
            "R_S_kge1_given_kge1": r1_given,
        })
    result_R["lighthill_source"] = {"table": src_rows}

    result["R_analysis"] = result_R
    print("R (all k) by Mach:", {m: rows[i]["R_all_k"] for i, m in enumerate(MACHS)})
    mem_report("after section 2")

    # ========================================================================
    # 3. Resolution check at the cone thresholds
    # ========================================================================
    t0 = time.time()
    rows_idx, cols_idx, k_mag_modes = low_k_indices(grid, k_cut=3.0)
    n_modes = rows_idx.size
    print(f"low-k mode count (|k|<=3): {n_modes}")

    modes_u, modes_v = extract_low_k_modes(u, v, rows_idx, cols_idx)
    timings["section3_mode_extraction"] = time.time() - t0
    mem_report("after low-k mode extraction")

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    modes_u_floor, modes_v_floor = extract_low_k_floor_modes(u, v, rows_idx, cols_idx, rng)
    timings["section3_floor_mode_extraction"] = time.time() - t0

    dk = 2.0 * np.pi / grid.length
    n_k_bins_lowk = 6  # bin centres 0, 0.5, 1.0, 1.5, 2.0, 2.5 -- fully covered by k_cut=3.0

    spec_hann4_lk, domega4 = welch_profile_from_modes(
        [modes_u, modes_v], k_mag_modes, grid.n, dt, "hann", 4, 0.5, dk, n_k_bins_lowk)
    spec_bh4_lk, domega4b = welch_profile_from_modes(
        [modes_u, modes_v], k_mag_modes, grid.n, dt, "blackmanharris", 4, 0.5, dk, n_k_bins_lowk)
    spec_dpss4_lk, domega4d = welch_profile_from_modes(
        [modes_u, modes_v], k_mag_modes, grid.n, dt, ("dpss", 4), 4, 0.5, dk, n_k_bins_lowk)
    spec_hann1_lk, domega1 = welch_profile_from_modes(
        [modes_u, modes_v], k_mag_modes, grid.n, dt, "hann", 1, 0.5, dk, n_k_bins_lowk)
    spec_floor_lk, domega_floor = welch_profile_from_modes(
        [modes_u_floor, modes_v_floor], k_mag_modes, grid.n, dt, "hann", 4, 0.5, dk, n_k_bins_lowk)
    timings["section3_welch_profiles"] = time.time() - t0
    mem_report("after low-k welch profiles")

    assert abs(domega4 - domega4b) < 1e-12 and abs(domega4 - domega4d) < 1e-12 and abs(domega4 - domega_floor) < 1e-12

    dens_hann4 = density(spec_hann4_lk, domega4)
    dens_bh4 = density(spec_bh4_lk, domega4)
    dens_dpss4 = density(spec_dpss4_lk, domega4)
    dens_hann1 = density(spec_hann1_lk, domega1)
    dens_floor = density(spec_floor_lk, domega_floor)

    # ---- sanity check: our low-k Hann-4seg profile at k=1.0 vs production ----
    ik_lk = k_index(spec_hann4_lk, 1.0)
    ik_prod = k_index(spec_hann4, 1.0)
    prod_col = spec_hann4.power[:, ik_prod]
    prod_col_interp = np.interp(spec_hann4_lk.omega, spec_hann4.omega, prod_col)
    lk_col = spec_hann4_lk.power[:, ik_lk]
    denom = np.maximum(np.abs(prod_col_interp), 1e-300)
    rel_err = np.abs(lk_col - prod_col_interp) / denom
    # compare only where production bin has non-negligible energy (avoid
    # dividing near-zero noise-floor values dominating the max)
    significant = prod_col_interp > 1e-6 * prod_col_interp.max()
    max_rel_err_significant = float(rel_err[significant].max()) if significant.any() else float("nan")
    sanity_check = {
        "max_rel_err_all_omega": float(rel_err.max()),
        "max_rel_err_where_significant": max_rel_err_significant,
        "n_omega_compared": int(len(rel_err)),
        "n_omega_significant": int(significant.sum()),
        "passes_1e-6_on_significant_bins": bool(max_rel_err_significant < 1e-6),
    }
    print("sanity check (low-k vs production, raw power, k=1.0):", sanity_check)

    # ---- density sanity line: median ratio Hann4/Hann1 in omega in [1,5], k=1.0
    band = (spec_hann4_lk.omega >= 1.0) & (spec_hann4_lk.omega <= 5.0)
    dens_hann1_interp_k1 = np.interp(spec_hann4_lk.omega, spec_hann1_lk.omega, dens_hann1[:, ik_lk])
    ratio_band = dens_hann4[band, ik_lk] / np.clip(dens_hann1_interp_k1[band], 1e-300, None)
    median_ratio_low_omega = float(np.median(ratio_band))

    # ========================================================================
    # table: at each (M,k) the four estimator DENSITY values
    # ========================================================================
    table_rows = []
    for m in MACHS:
        c0 = c0_by_m[m]
        for k in (0.5, 1.0, 1.5, 2.0):
            omega_target = c0 * k
            ik = k_index(spec_hann4_lk, k)
            iw4 = nearest_idx(spec_hann4_lk.omega, omega_target)
            iw1 = nearest_idx(spec_hann1_lk.omega, omega_target)

            p_hann4 = dens_hann4[iw4, ik]
            p_bh4 = dens_bh4[iw4, ik]
            p_dpss4 = dens_dpss4[iw4, ik]
            p_hann1 = dens_hann1[iw1, ik]
            p_floor = dens_floor[iw4, ik]

            vals = [p_hann4, p_bh4, p_dpss4, p_hann1]
            vmax, vmin = max(vals), min(vals)
            ratio = vmax / vmin if vmin > 0 else float("inf")
            floor_margin = p_hann4 / p_floor if p_floor > 0 else float("inf")

            omega_within_nyquist = omega_target <= (np.pi / dt)
            agree = ratio <= AGREEMENT_RATIO_THRESHOLD
            above_floor = floor_margin >= FLOOR_MARGIN_THRESHOLD
            verdict = "RESOLVED" if (omega_within_nyquist and agree and above_floor) else "UNRESOLVED"

            table_rows.append({
                "mach": m, "k": k, "c0": c0, "omega_target": float(omega_target),
                "omega_bin_4seg": float(spec_hann4_lk.omega[iw4]),
                "omega_bin_1seg": float(spec_hann1_lk.omega[iw1]),
                "density_hann4seg": float(p_hann4),
                "density_blackmanharris4seg": float(p_bh4),
                "density_dpss4seg": float(p_dpss4),
                "density_hann_fullrecord": float(p_hann1),
                "max_over_min_ratio": float(ratio),
                "density_float32_floor": float(p_floor),
                "margin_over_float32_floor": float(floor_margin),
                "within_nyquist": bool(omega_within_nyquist),
                "verdict": verdict,
            })

    # ========================================================================
    # leakage-contaminated cone-energy share (velocity, |k|<=3 bins only)
    # ========================================================================
    dens_hann1_interp_full = np.empty_like(dens_hann4)
    for b in range(dens_hann4.shape[1]):
        dens_hann1_interp_full[:, b] = np.interp(spec_hann4_lk.omega, spec_hann1_lk.omega, dens_hann1[:, b])
    leakage_mask = dens_hann4 > (AGREEMENT_RATIO_THRESHOLD * dens_hann1_interp_full)

    leakage_by_mach = {}
    for m in MACHS:
        c0 = c0_by_m[m]
        cone_mask = spec_hann4_lk.radiating_mask(c0)
        cone_energy_total = float(spec_hann4_lk.power[cone_mask].sum())
        cone_energy_leaky = float(spec_hann4_lk.power[cone_mask & leakage_mask].sum())
        frac = cone_energy_leaky / cone_energy_total if cone_energy_total > 0 else float("nan")
        leakage_by_mach[m] = frac

    result["resolution_check"] = {
        "n_low_k_modes": int(n_modes),
        "k_cut": 3.0,
        "sanity_check_vs_production_raw_power_k1.0": sanity_check,
        "density_sanity_median_ratio_hann4_over_hann1_omega_1to5_k1.0": median_ratio_low_omega,
        "domega_4segment": domega4,
        "domega_1segment_fullrecord": domega1,
        "agreement_ratio_threshold": AGREEMENT_RATIO_THRESHOLD,
        "floor_margin_threshold": FLOOR_MARGIN_THRESHOLD,
        "table": table_rows,
        "leakage_contaminated_cone_energy_fraction_by_mach": leakage_by_mach,
        "leakage_note": (
            "Computed only over the |k|<=3 low-k mode set (bin centres "
            "0,0.5,1.0,1.5,2.0,2.5); does not cover the full (k,omega) plane "
            "used for the section-2 R values."
        ),
    }
    print("resolution table sample:", table_rows[:2])
    print("leakage-contaminated cone fraction by mach:", leakage_by_mach)

    # ========================================================================
    # 3b. Integrated cone-energy check (follow-up, coordinator request)
    #
    # A single-bin comparison is noisy: the Hann single-segment full-record
    # periodogram has ~2 degrees of freedom per omega bin (relative scatter
    # ~100%), so a factor-2 disagreement in one bin is expected noise rather
    # than leakage. Leakage would inflate the coarser 4-seg estimator
    # consistently; genuine estimator variance flips sign bin to bin.
    # Test the quantity R actually uses: the integral of density over the
    # cone (omega >= c0*k), which averages out single-bin noise.
    # ========================================================================
    def integrated_and_band_checks(estimators, k_bins, c0_by_mach, machs, ks,
                                    edge_frac=0.05, band_width=2.0):
        rows_out = []
        for m in machs:
            c0 = c0_by_mach[m]
            for k in ks:
                threshold = c0 * k
                ik = nearest_idx(k_bins, k)
                per_est = {}
                for name, omega_e, dens_e, domega_e in estimators:
                    omega_max = omega_e.max()
                    cutoff_hi = omega_max * (1.0 - edge_frac)
                    col = dens_e[:, ik]
                    mask_excl = (omega_e >= threshold) & (omega_e <= cutoff_hi)
                    mask_full = omega_e >= threshold
                    C_excl = float(col[mask_excl].sum() * domega_e)
                    C_full = float(col[mask_full].sum() * domega_e)
                    band_mask = (omega_e >= threshold) & (omega_e <= threshold + band_width)
                    band_mean = float(col[band_mask].mean()) if band_mask.any() else float("nan")
                    per_est[name] = {
                        "C_with_edge_exclusion": C_excl,
                        "C_no_edge_exclusion": C_full,
                        "edge_exclusion_rel_change": ((C_full - C_excl) / C_excl
                                                       if C_excl > 0 else float("nan")),
                        "band_mean_density": band_mean,
                        "n_bins_in_cone_excl": int(mask_excl.sum()),
                        "n_bins_in_band": int(band_mask.sum()),
                    }
                C_vals = [per_est[n]["C_with_edge_exclusion"] for n, *_ in estimators]
                band_vals = [per_est[n]["band_mean_density"] for n, *_ in estimators]
                C_ratio = (max(C_vals) / min(C_vals)) if min(C_vals) > 0 else float("inf")
                band_ratio = (max(band_vals) / min(band_vals)) if min(band_vals) > 0 else float("inf")
                verdict = "RESOLVED" if C_ratio <= 1.5 else "UNRESOLVED"
                rows_out.append({
                    "mach": m, "k": k, "c0": c0, "omega_threshold": threshold,
                    "estimators": per_est,
                    "C_max_over_min_ratio": C_ratio,
                    "band_avg_density_max_over_min_ratio": band_ratio,
                    "verdict_integrated_C_le_1.5": verdict,
                })
        return rows_out

    estimators_list = [
        ("Hann4", spec_hann4_lk.omega, dens_hann4, domega4),
        ("BH4", spec_bh4_lk.omega, dens_bh4, domega4),
        ("DPSS4", spec_dpss4_lk.omega, dens_dpss4, domega4),
        ("Hann_full", spec_hann1_lk.omega, dens_hann1, domega1),
    ]
    integrated_rows = integrated_and_band_checks(
        estimators_list, spec_hann4_lk.k_bins, c0_by_m, MACHS, (0.5, 1.0, 1.5, 2.0))

    earlier_k05_verdict = {r["mach"]: r["verdict"] for r in table_rows if r["k"] == 0.5}
    integrated_k05_verdict = {r["mach"]: r["verdict_integrated_C_le_1.5"]
                               for r in integrated_rows if r["k"] == 0.5}
    k05_survival = {m: {"single_bin_verdict": earlier_k05_verdict[m],
                         "integrated_verdict": integrated_k05_verdict[m]}
                     for m in MACHS}

    max_edge_rel_change = max(
        abs(row["estimators"][name]["edge_exclusion_rel_change"])
        for row in integrated_rows for name, *_ in estimators_list
        if not np.isnan(row["estimators"][name]["edge_exclusion_rel_change"])
    )

    result["integrated_cone_check"] = {
        "method": (
            "C_E(M,k) = sum over omega bins with c0*k <= omega <= 0.95*omega_max "
            "of density_E(k,omega) * domega_E (that estimator's own bin width); "
            "edge exclusion drops the top 5% of each estimator's omega range to "
            "avoid Nyquist-edge artefacts. band_mean_density is the mean of "
            "density_E(k,omega) over omega in [c0*k, c0*k+2] (~13 full-record "
            "bins for Hann_full, ~5 bins for the 4-seg-Welch estimators). "
            "RESOLVED requires max/min ratio of C_E across the four estimators "
            "<= 1.5."
        ),
        "edge_exclusion_max_abs_relative_change_over_all_rows": float(max_edge_rel_change),
        "rows": integrated_rows,
        "k05_survival_vs_single_bin_verdict": k05_survival,
    }
    print("\n--- integrated cone-energy check (C_E, follow-up) ---")
    for row in integrated_rows:
        print(f"  M={row['mach']:<5} k={row['k']:<4} C_ratio={row['C_max_over_min_ratio']:.3f} "
              f"band_ratio={row['band_avg_density_max_over_min_ratio']:.3f} "
              f"verdict={row['verdict_integrated_C_le_1.5']}")
    print(f"  max |edge-exclusion relative change| over all rows/estimators: {max_edge_rel_change:.4f}")
    print("  k=0.5 survival vs single-bin verdicts:", k05_survival)

    # ========================================================================
    # figure: log10 density vs omega for k=0.5,1,1.5,2 with c0*k vlines
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False)
    mach_colors = plt.cm.autumn(np.linspace(0.1, 0.8, len(MACHS)))
    for ax, k in zip(axes.ravel(), (0.5, 1.0, 1.5, 2.0)):
        ik = k_index(spec_hann4_lk, k)
        ax.plot(spec_hann4_lk.omega, np.log10(np.clip(dens_hann4[:, ik], 1e-300, None)),
                color="C0", label="Hann 4-seg (production)")
        ax.plot(spec_bh4_lk.omega, np.log10(np.clip(dens_bh4[:, ik], 1e-300, None)),
                color="C1", label="Blackman-Harris 4-seg")
        ax.plot(spec_dpss4_lk.omega, np.log10(np.clip(dens_dpss4[:, ik], 1e-300, None)),
                color="C2", label="DPSS NW=4, 4-seg")
        ax.plot(spec_hann1_lk.omega, np.log10(np.clip(dens_hann1[:, ik], 1e-300, None)),
                color="C3", label="Hann single-seg full record")
        ax.plot(spec_floor_lk.omega, np.log10(np.clip(dens_floor[:, ik], 1e-300, None)),
                color="k", ls="--", lw=1, label="float32 rounding floor")
        for mcolor, m in zip(mach_colors, MACHS):
            ax.axvline(c0_by_m[m] * k, color=mcolor, ls=":", lw=1.2, label=f"c0*k, M={m}")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("omega")
        ax.set_ylabel("log10 density  (power / domega)")
        ax.set_xlim(0, max(spec_hann4_lk.omega.max(), spec_hann1_lk.omega.max()))
    axes.ravel()[0].legend(fontsize=6.5, loc="upper right", ncol=1)
    fig.suptitle("L=4pi, n=256: (k,omega) density profiles, estimator agreement check")
    fig.tight_layout()
    res_fig_path = OUT_DIR / "box_L4pi_resolution_profiles.png"
    fig.savefig(res_fig_path, dpi=130)
    plt.close(fig)
    result["resolution_check"]["figure"] = str(res_fig_path)
    mem_report("after section 3")

    # ========================================================================
    # wrap up
    # ========================================================================
    timings["total_runtime"] = time.time() - t_all
    result["timings_seconds"] = timings
    result["peak_rss_mb"] = peak_rss_mb()

    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nWrote {json_path}")
    print(f"Total runtime: {timings['total_runtime']:.1f}s  peak RSS: {result['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
