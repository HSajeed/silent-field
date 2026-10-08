#!/usr/bin/env python3
"""STAGE 3 -- does an error-floor crossover predict the Mach below which a surrogate fails acoustically?

Hypothesis (2026-10-08 review).  At low |k| the true temporal spectrum falls far faster in omega
(~omega^-13 near the top of the band) than a trained surrogate's error (~omega^-5..-6).  So there
is a crossover frequency omega_x above which the error exceeds the signal (or exceeds the copy-input
error).  The radiating band on the lowest shell |k| = dk starts at omega = c0 * dk = U * dk / M, so
the whole lowest-shell radiating band sits above the crossover once

        M  <=  M*_pred = U * dk / omega_x .

This script measures omega_x from the low-k omega spectra that scripts/10 already wrote, predicts
M*, and compares it with the M* observed in scripts/03's far field (the Mach at which the model's
radiating-band far-field error equals the copy-input error, or 100%), interpolated in log-log
between the Machs that were evaluated.  No model, no torch, no dataset: JSON in, JSON + figure out.

Caveats built into the method (read before quoting numbers):
  * psd_lowk_* in scripts/10 is summed over |k| <= omega_max / c0 at the Mach scripts/10 ran at
    (6.93 baseline, 9.93 kf16 at M=0.1), not the |k| = dk shell alone.  omega_x is a proxy.
  * dk = 2*pi/L; both trained datasets have L = 2*pi, so --dk 1 (default).
  * Observed M* below the lowest evaluated Mach (0.05) is extrapolated and flagged.

    python scripts/11_crossover.py --runs-root /content/drive/MyDrive/silent-field/runs --out /content/drive/MyDrive/silent-field/analysis/11_crossover
    python scripts/11_crossover.py --runs-root runs --out figures/11_crossover
"""
import argparse, json, math
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------ discovery

def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def find_files(run_dir):
    """Pick this run's scripts/10, Mach-swept scripts/03 and scripts/09 outputs (searched recursively)."""
    run_dir = Path(run_dir)
    diag = sorted(run_dir.rglob("10_error_diagnostic.json"), key=lambda p: ("diag" not in p.parts, len(p.parts)))
    f03 = None
    for p in sorted(run_dir.rglob("03_errors.json"), key=lambda p: ("machs" not in p.parts, len(p.parts))):
        d = _load(p)
        if d and len(d.get("far_field_by_mach", {})) >= 2:
            f03 = p
            break
    f09 = sorted(run_dir.rglob("09_blindness_machs.json"), key=lambda p: len(p.parts))
    return (diag[0] if diag else None), f03, (f09[0] if f09 else None)


def family(name, diag):
    loss = (diag or {}).get("loss") or ""
    arch = (diag or {}).get("arch") or ""
    unroll = (diag or {}).get("unroll", 1) or 1
    if loss == "cone" or name.startswith("cone"):
        return "cone"
    if loss == "h1" or "h1" in name:
        return "h1"
    if unroll and unroll > 1 or "unroll" in name:
        return "unroll"
    if arch == "unet" or "unet" in name:
        return "unet"
    return "mse"


# ------------------------------------------------------------------ crossings

def sustained_crossing(x, r, start=1, hold=3):
    """First x at which r rises through 1 and stays >= 1 for `hold` consecutive points.

    Log-log interpolation between the bracketing points.  Returns (x_cross, flag) with flag in
    {"ok", "already_above", "never"}.
    """
    x, r = np.asarray(x, float), np.asarray(r, float)
    n = len(x)
    if np.all(r[start:start + hold] >= 1):
        return float(x[start]), "already_above"
    for i in range(start + 1, n - hold + 1):
        if np.all(r[i:i + hold] >= 1) and r[i - 1] < 1:
            lx0, lx1 = math.log(x[i - 1]), math.log(x[i])
            lr0, lr1 = math.log(max(r[i - 1], 1e-300)), math.log(r[i])
            t = (0.0 - lr0) / (lr1 - lr0) if lr1 != lr0 else 0.0
            return float(math.exp(lx0 + t * (lx1 - lx0))), "ok"
    return None, "never"


def mach_crossing(machs, ratio):
    """Mach at which ratio falls through 1 as M increases (log-log interpolation).

    Outside the evaluated range, extrapolates from the two nearest points and says so.
    """
    m, r = np.asarray(machs, float), np.asarray(ratio, float)
    o = np.argsort(m); m, r = m[o], r[o]
    lm, lr = np.log(m), np.log(np.maximum(r, 1e-300))
    for i in range(len(m) - 1):
        if r[i] >= 1 > r[i + 1]:
            t = (0 - lr[i]) / (lr[i + 1] - lr[i])
            return float(np.exp(lm[i] + t * (lm[i + 1] - lm[i]))), "interpolated"
    if np.all(r < 1):              # fine everywhere evaluated -> M* below the lowest Mach
        i, j = 0, 1
        tag = "extrapolated_below"
    elif np.all(r >= 1):           # fails everywhere evaluated -> M* above the highest Mach
        i, j = len(m) - 2, len(m) - 1
        tag = "extrapolated_above"
    else:                          # non-monotone; report the first upward violation
        return None, "non_monotone"
    slope = (lr[j] - lr[i]) / (lm[j] - lm[i])
    if slope >= 0:
        return None, tag + "_bad_slope"
    mx = float(np.exp(lm[i] - lr[i] / slope))
    if not (m[0] / 3 <= mx <= m[-1] * 3):   # refuse to extrapolate more than 3x outside the data
        return None, tag + "_too_far"
    return mx, tag


def loglog_slope(x, y, lo, hi):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = (x >= lo) & (x <= hi) & (y > 0)
    if s.sum() < 4:
        return None
    return float(np.polyfit(np.log10(x[s]), np.log10(y[s]), 1)[0])


# ------------------------------------------------------------------ per run

def analyse_run(run_dir, dk, fit_lo_frac):
    name = Path(run_dir).name
    f10, f03, f09 = find_files(run_dir)
    rec = {"run": name, "files": {"10": str(f10) if f10 else None, "03": str(f03) if f03 else None,
                                  "09": str(f09) if f09 else None}}
    d10 = _load(f10) if f10 else None
    rec["family"] = family(name, d10)
    if d10:
        om = np.array(d10["omega"])
        pt = np.array(d10["psd_lowk_truth"])
        pe = np.array(d10["model"]["psd_lowk_error"])
        pc = np.array(d10["copy_input"]["psd_lowk_error"])
        mach10, c0 = d10["mach"], d10["c0"]
        U = c0 * mach10
        wx_t, ft = sustained_crossing(om, pe / np.maximum(pt, 1e-300))
        wx_c, fc = sustained_crossing(om, pe / np.maximum(pc, 1e-300))
        lo, hi = fit_lo_frac * om.max(), om.max()
        p_t, p_e, p_c = loglog_slope(om, pt, lo, hi), loglog_slope(om, pe, lo, hi), loglog_slope(om, pc, lo, hi)
        rec["lowk"] = {
            "mach_of_diag": mach10, "c0": c0, "U": U, "dk": dk, "k_sum_max": d10.get("k_cone_max"),
            "omega_x_err_eq_truth": wx_t, "flag_truth": ft,
            "omega_x_err_eq_copy": wx_c, "flag_copy": fc,
            "fit_range": [lo, hi], "slope_truth": p_t, "slope_err": p_e, "slope_copy": p_c,
            "Mstar_pred_truth": (U * dk / wx_t) if wx_t else None,
            "Mstar_pred_copy": (U * dk / wx_c) if wx_c else None,
            "err_over_truth_at": {f"{w:g}": float(np.interp(w, om, pe / np.maximum(pt, 1e-300)))
                                  for w in (11.3, 19.2, 38.4, float(om.max()))},
        }
        rec["_curves"] = {"omega": om.tolist(), "err_over_truth": (pe / np.maximum(pt, 1e-300)).tolist(),
                          "err_over_copy": (pe / np.maximum(pc, 1e-300)).tolist()}
    d03 = _load(f03) if f03 else None
    if d03:
        fbm = d03["far_field_by_mach"]
        ms = sorted(float(m) for m in fbm)
        mod = [fbm[_key(fbm, m)]["model"]["far_field_complex_above_c0dk"] for m in ms]
        cop = [fbm[_key(fbm, m)]["copy_input"]["far_field_complex_above_c0dk"] for m in ms]
        rc = np.array(mod) / np.array(cop)
        m_copy, tag_c = mach_crossing(ms, rc)
        m_100, tag_1 = mach_crossing(ms, np.array(mod))
        rec["farfield"] = {"machs": ms, "model_above": mod, "copy_above": cop, "model_over_copy": rc.tolist(),
                           "Mstar_obs_copy": m_copy, "tag_copy": tag_c, "Mstar_obs_100pct": m_100, "tag_100pct": tag_1}
        pf = d03.get("per_frequency", [])
        if pf:
            w = np.array([q["omega"] for q in pf]); em = np.array([q["err_model"] for q in pf])
            ec = np.array([q["err_copy"] for q in pf])
            a = w > d03.get("c0", 0)          # radiating part only
            wf, ff = sustained_crossing(w[a], (em / ec)[a], start=0, hold=2) if a.sum() > 2 else (None, "n/a")
            rec["farfield"]["perfreq_ref_mach"] = d03.get("mach")
            rec["farfield"]["omega_model_worse_than_copy"] = wf
            rec["farfield"]["flag_perfreq"] = ff
            rec["farfield"]["perfreq_top2_model_over_copy"] = (em[-2:] / ec[-2:]).tolist()
    d09 = _load(f09) if f09 else None
    if d09:
        v = d09["variants"]
        bm = v["model_tf"]["by_mach"]; bw = v["white_matched"]["by_mach"]
        rec["B_over_Bwhite"] = {k: bm[k]["B"] / bw[k]["B"] for k in sorted(bm, key=float)}
        rec["B"] = {k: bm[k]["B"] for k in sorted(bm, key=float)}
        ro = v.get("model_rollout", {}).get("by_mach", {})
        if ro:
            rec["rollout_P_ratio"] = {k: ro[k]["P_pred_over_truth"] for k in sorted(ro, key=float)}
    return rec


def _key(d, m):
    for k in d:
        if abs(float(k) - m) < 1e-12:
            return k
    raise KeyError(m)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="directory whose subdirectories are runs")
    ap.add_argument("--runs", nargs="*", help="restrict to these run names")
    ap.add_argument("--dk", type=float, default=1.0, help="2*pi/L of the training data (1 for L=2*pi)")
    ap.add_argument("--fit-lo-frac", type=float, default=0.25, help="slope fit over [frac*omega_max, omega_max]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.runs_root)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (not args.runs or p.name in args.runs))
    recs = [analyse_run(p, args.dk, args.fit_lo_frac) for p in runs]
    recs = [r for r in recs if "lowk" in r or "farfield" in r]

    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / "11_crossover.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "_curves"} for r in recs], indent=2))

    # ---- table
    f = lambda x, s="{:.3g}": "—" if x is None else s.format(x)
    print("=" * 128)
    print(f"{'run':<26}{'fam':<7}{'ωx(err=truth)':>14}{'ωx(err=copy)':>13}{'p_truth':>9}{'p_err':>8}"
          f"{'M*pred(t)':>11}{'M*pred(c)':>11}{'M*obs(copy)':>13}{'tag':>20}{'M*obs(100%)':>13}")
    for r in recs:
        lk, ff = r.get("lowk", {}), r.get("farfield", {})
        print(f"{r['run']:<26}{r['family']:<7}{f(lk.get('omega_x_err_eq_truth')):>14}{f(lk.get('omega_x_err_eq_copy')):>13}"
              f"{f(lk.get('slope_truth'), '{:.1f}'):>9}{f(lk.get('slope_err'), '{:.1f}'):>8}"
              f"{f(lk.get('Mstar_pred_truth')):>11}{f(lk.get('Mstar_pred_copy')):>11}"
              f"{f(ff.get('Mstar_obs_copy')):>13}{ff.get('tag_copy', '—'):>20}{f(ff.get('Mstar_obs_100pct')):>13}")
    print("=" * 128)
    pairs = [(r["lowk"]["Mstar_pred_truth"], r["farfield"]["Mstar_obs_copy"]) for r in recs
             if r.get("lowk", {}).get("Mstar_pred_truth") and r.get("farfield", {}).get("Mstar_obs_copy")]
    if len(pairs) >= 3:
        a = np.log(np.array(pairs))
        rho = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        ratio = np.exp(a[:, 1] - a[:, 0])
        print(f"log-log corr(M*pred(truth), M*obs(copy)) = {rho:.3f} over {len(pairs)} runs;"
              f"  obs/pred median {np.median(ratio):.2f}, range [{ratio.min():.2f}, {ratio.max():.2f}]")

    # ---- figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"mse": "#4C72B0", "unet": "#55A868", "unroll": "#8172B2", "h1": "#C44E52", "cone": "#DD8452"}
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        for r in recs:
            c = col.get(r["family"], "gray")
            if "_curves" in r:
                cv = r["_curves"]; w = np.array(cv["omega"][1:])
                ax[0].loglog(w, np.array(cv["err_over_truth"][1:]), color=c, lw=1, alpha=.8)
                ax[1].loglog(w, np.array(cv["err_over_copy"][1:]), color=c, lw=1, alpha=.8)
            lk, ff = r.get("lowk", {}), r.get("farfield", {})
            for key, mk in (("Mstar_pred_truth", "o"), ("Mstar_pred_copy", "s")):
                if lk.get(key) and ff.get("Mstar_obs_copy"):
                    hollow = ff["tag_copy"] != "interpolated"
                    ax[2].scatter(lk[key], ff["Mstar_obs_copy"], marker=mk, s=46,
                                  facecolors="none" if hollow else c, edgecolors=c)
                    ax[2].annotate(r["run"], (lk[key], ff["Mstar_obs_copy"]), fontsize=6, alpha=.7) if key == "Mstar_pred_truth" else None
        for a, t in ((ax[0], "low-k error PSD / truth PSD"), (ax[1], "low-k error PSD / copy-input error PSD")):
            a.axhline(1, color="k", lw=.8, ls=":"); a.set_xlabel("ω"); a.set_title(t)
        lim = [0.01, 0.2]
        ax[2].plot(lim, lim, "k:", lw=.8); ax[2].set_xscale("log"); ax[2].set_yscale("log")
        ax[2].set_xlabel("M* predicted = U·dk/ω_x   (○ err=truth, □ err=copy)")
        ax[2].set_ylabel("M* observed: far field above c0·dk = copy-input")
        ax[2].set_title("hollow = extrapolated beyond evaluated Machs")
        for fam, c in col.items():
            ax[0].plot([], [], color=c, label=fam)
        ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(o / "11_crossover.png", dpi=140)
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"results: {o}")


if __name__ == "__main__":
    main()
