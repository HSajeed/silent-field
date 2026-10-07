#!/usr/bin/env python3
"""STAGE 3 -- where in (k, omega) the surrogate's error lives, and whether it is jitter.

scripts/09 found the teacher-forced error ~4700x over-represented in the radiating cone,
20-130x below an equally large time-incoherent error.  Two explanations predict different
fingerprints:

  jitter    step-to-step incoherent error.  At the cone's low |k| the error's omega
            spectrum is flat (white) where the truth's falls steeply, and the error's
            lag-1 temporal correlation is far below the truth's.
  aliasing  high-k content folded back by the network's nonlinearity (McCabe et al. 2023).
            A deterministic function of a smooth-in-time input, so temporally coherent,
            and concentrated at high |k|.

Per |k| shell this reports the error's share, its lag-1 correlation (truth beside it), and
the omega-spectrum of error vs truth over the cone's k range -- for the model and for the
copy-input baseline (a coherent reference).  Writes a (k, omega) figure.

    python scripts/10_error_diagnostic.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01/diag
    python scripts/10_error_diagnostic.py --smoke --model tmp/smoke02/model.pt --out tmp/smoke10
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spectral import Grid
from src.solvers.io import load_dataset
from src.acoustics.komega import velocity_komega_spectrum, sound_speed


def shell_lag1(eu, ev, grid):
    """Per integer shell m^2: energy and lag-1 temporal correlation of a 2-component field.

    rho(m) = Re sum_t sum_{modes in shell} e_t conj(e_{t+1}) / sum_t sum |e_t|^2.
    White-in-time noise gives ~0; a slowly varying field gives ~1.
    """
    dk = 2.0 * np.pi / grid.length
    m2 = np.rint(grid.k_sq / dk**2).astype(np.int64).ravel()
    nbin = m2.max() + 1
    num, den = np.zeros(nbin), np.zeros(nbin)
    for comp in (eu, ev):
        prev = np.fft.fft2(comp[0]).ravel()
        den += np.bincount(m2, np.abs(prev) ** 2, nbin)
        for t in range(1, comp.shape[0]):
            cur = np.fft.fft2(comp[t]).ravel()
            num += np.bincount(m2, np.real(prev * np.conj(cur)), nbin)
            den += np.bincount(m2, np.abs(cur) ** 2, nbin)
            prev = cur
    present = den > 0
    return dk * np.sqrt(np.nonzero(present)[0]), den[present], num[present] / den[present]


def band(k, w, sel):
    """Energy-weighted mean of a per-shell quantity over a selection of shells."""
    return float(np.sum(w[sel]) and np.sum(w[sel] * k[sel]) / np.sum(w[sel]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/turb2d.npz")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mach", type=float, default=0.1)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="first 64 snapshots, 2 segments")
    args = ap.parse_args()
    t0 = time.time()

    from src.models.train import TrainConfig, make_pairs, predict_onestep, denormalise, load_model

    u, v, meta = load_dataset(args.data)
    if args.smoke:
        u, v, args.segments = u[:64], v[:64], 2
    grid = Grid(n=u.shape[-1], length=meta["grid_length"])
    dt = meta["dt"]
    model, ckpt = load_model(args.model, grid.n)
    stats = ckpt["stats"]
    x, _, _ = make_pairs(u, v, normalise=False)
    x = (x - stats["mean"]) / stats["std"]
    pred = denormalise(predict_onestep(model, x, TrainConfig()), stats).astype(np.float64)
    del x
    ut, vt = u[1:].astype(np.float64), v[1:].astype(np.float64)
    U = float(np.sqrt(np.mean(ut**2 + vt**2)))
    c0 = sound_speed(U, args.mach)
    errors = {"model": (pred[:, 0] - ut, pred[:, 1] - vt),
              "copy_input": (u[:-1] - ut, v[:-1] - vt)}
    del pred

    spec_t = velocity_komega_spectrum(ut, vt, grid, dt, n_segments=args.segments)
    omega = spec_t.omega
    k_cone = omega.max() / c0                       # the cone reaches |k| <= omega_max / c0
    k_alias = 2.0 / 3.0 * grid.k_max                # FNO/UNet nonlinearity folds content above this
    kt, wt, rho_t = shell_lag1(ut, vt, grid)
    low_t = spec_t.k_resolved <= k_cone
    psd_t = spec_t.power_resolved[:, low_t].sum(axis=1)

    out = {"mach": args.mach, "c0": c0, "k_cone_max": float(k_cone), "k_alias": float(k_alias),
           "truth": {"rho_lag1_cone_k": band(rho_t, wt, (kt > 0) & (kt <= k_cone)),
                     "rho_lag1_alias_k": band(rho_t, wt, kt > k_alias)},
           "omega": omega.tolist(), "psd_lowk_truth": psd_t.tolist()}
    specs = {}
    for name, (eu, ev) in errors.items():
        se = velocity_komega_spectrum(eu, ev, grid, dt, n_segments=args.segments)
        specs[name] = se
        ke, we, rho_e = shell_lag1(eu, ev, grid)
        cone = se.radiating_mask_resolved(c0)
        cone_pow = (se.power_resolved * cone).sum(axis=0)          # per resolved shell
        tot = cone_pow.sum()
        top = np.argsort(cone_pow)[::-1][:5]
        psd_e = se.power_resolved[:, se.k_resolved <= k_cone].sum(axis=1)
        hi = omega >= 0.5 * omega.max()
        mid = (omega >= 0.05 * omega.max()) & (omega < 0.15 * omega.max())
        out[name] = {
            "energy_share_alias_k": float(we[ke > k_alias].sum() / we.sum()),
            "energy_share_cone_k": float(we[(ke > 0) & (ke <= k_cone)].sum() / we.sum()),
            "rho_lag1_cone_k": band(rho_e, we, (ke > 0) & (ke <= k_cone)),
            "rho_lag1_alias_k": band(rho_e, we, ke > k_alias),
            "cone_share_by_shell_top5": [{"k": float(se.k_resolved[i]), "share": float(cone_pow[i] / tot)}
                                         for i in top] if tot > 0 else [],
            # how flat the low-k error spectrum is: high-omega / mid-omega mean PSD (truth beside it)
            "lowk_hi_over_mid": float(psd_e[hi].mean() / psd_e[mid].mean()),
            "lowk_hi_over_mid_truth": float(psd_t[hi].mean() / psd_t[mid].mean()),
            "lowk_err_over_truth_hi": float(psd_e[hi].sum() / psd_t[hi].sum()),
            "lowk_err_over_truth_mid": float(psd_e[mid].sum() / psd_t[mid].sum()),
            "psd_lowk_error": psd_e.tolist(),
        }
        print(f"[{name}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / "10_error_diagnostic.json").write_text(json.dumps(
        {**out, "model_path": str(args.model), "arch": ckpt["arch"], "loss": ckpt.get("loss"), "unroll": ckpt.get("unroll", 1),
         "smoke": args.smoke}, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        se = specs["model"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        for a, P, title in ((ax[0], spec_t.power, "truth E(k, ω)"), (ax[1], se.power, "model error E(k, ω)")):
            P = np.maximum(P, P.max() * 1e-14)
            pc = a.pcolormesh(spec_t.k_bins, omega, P, norm=LogNorm(), shading="auto")
            a.plot(spec_t.k_bins, c0 * spec_t.k_bins, "r-", lw=1.5, label="sonic line")
            a.plot(spec_t.k_bins, U * spec_t.k_bins, "w--", lw=1.2, label="convective ridge")
            a.set_xlim(spec_t.k_bins[1], spec_t.k_bins.max()); a.set_ylim(omega[1], omega.max())
            a.set_xlabel("|k|"); a.set_ylabel("ω"); a.set_title(title); fig.colorbar(pc, ax=a)
        ax[0].legend(loc="upper right", fontsize=8)
        ratio = (se.power / se.power.sum()) / np.maximum(spec_t.power / spec_t.power.sum(), 1e-300)
        ratio = np.where(spec_t.power > spec_t.power.max() * 1e-12, ratio, np.nan)  # empty (dealiased) bins
        pc = ax[2].pcolormesh(spec_t.k_bins, omega, np.clip(ratio, 1e-3, 1e7), norm=LogNorm(), shading="auto")
        ax[2].plot(spec_t.k_bins, c0 * spec_t.k_bins, "r-", lw=1.5)
        ax[2].set_xlim(spec_t.k_bins[1], spec_t.k_bins.max()); ax[2].set_ylim(omega[1], omega.max())
        ax[2].set_title("error share / truth share (local B)"); ax[2].set_xlabel("|k|")
        fig.colorbar(pc, ax=ax[2]); fig.tight_layout(); fig.savefig(o / "10_kw_error.png", dpi=130)
        fig, a = plt.subplots(figsize=(6.5, 4.2))
        a.loglog(omega[1:], psd_t[1:], "k-", label="truth")
        for name in errors:
            a.loglog(omega[1:], np.array(out[name]["psd_lowk_error"])[1:], label=f"error: {name}")
        a.axvline(c0 * 2 * np.pi / grid.length, color="r", ls=":", label="c0·dk")
        a.set_xlabel("ω"); a.set_ylabel(f"PSD summed over |k| ≤ {k_cone:.1f}"); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(o / "10_lowk_omega_spectrum.png", dpi=130)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\n" + "=" * 78)
    print(f"SUMMARY  scripts/10_error_diagnostic.py  ({ckpt['arch'].upper()} {ckpt.get('loss', '')} unroll {ckpt.get('unroll', 1)}, "
          f"M={args.mach}{', SMOKE' if args.smoke else ''})")
    print(f"  cone |k| range <= {k_cone:.2f};  aliasing band |k| > {k_alias:.1f}")
    print(f"  {'':<34}{'model':>12}{'copy-input':>12}{'truth':>10}")
    m, c, tr = out["model"], out["copy_input"], out["truth"]
    print(f"  {'lag-1 corr, cone |k|':<34}{m['rho_lag1_cone_k']:>12.3f}{c['rho_lag1_cone_k']:>12.3f}"
          f"{tr['rho_lag1_cone_k']:>10.3f}")
    print(f"  {'lag-1 corr, aliasing band':<34}{m['rho_lag1_alias_k']:>12.3f}{c['rho_lag1_alias_k']:>12.3f}"
          f"{tr['rho_lag1_alias_k']:>10.3f}")
    print(f"  {'error energy share, cone |k|':<34}{m['energy_share_cone_k']:>12.2e}{c['energy_share_cone_k']:>12.2e}")
    print(f"  {'error energy share, aliasing band':<34}{m['energy_share_alias_k']:>12.2e}{c['energy_share_alias_k']:>12.2e}")
    print(f"  {'low-k PSD, high-ω / mid-ω':<34}{m['lowk_hi_over_mid']:>12.2e}{c['lowk_hi_over_mid']:>12.2e}"
          f"{m['lowk_hi_over_mid_truth']:>10.2e}")
    print(f"  {'low-k error/truth, high ω':<34}{m['lowk_err_over_truth_hi']:>12.2e}{c['lowk_err_over_truth_hi']:>12.2e}")
    print(f"  {'low-k error/truth, mid ω':<34}{m['lowk_err_over_truth_mid']:>12.2e}{c['lowk_err_over_truth_mid']:>12.2e}")
    tops = ", ".join(f"|k|={d['k']:.2f}: {d['share']:.0%}" for d in m["cone_share_by_shell_top5"][:3])
    print(f"  model cone error by shell        {tops}")
    print(f"  wall time {(time.time() - t0) / 60:.1f} min   results {o}")
    print("=" * 78)


if __name__ == "__main__":
    main()
