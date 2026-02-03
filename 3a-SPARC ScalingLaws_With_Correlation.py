#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3a-SPARC_ScalingLawsParametersGenerator.py

Extrait des lois d'échelle (scaling laws) depuis:
  SPARC_fit_summary_optimized_5p_occam.csv

Fitte des lois de puissance:
  y = C * x^beta   <=>  log10(y) = a + beta*log10(x), C=10^a

Méthodes:
- OLS (Ordinary Least Squares = moindres carrés ordinaires)
- Theil-Sen (robuste aux outliers)

Sorties dans:
  ./2-SPARC Scaling Laws Results/
    - scaling_laws_summary.csv
    - PNG (1 plot par relation)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import theilslopes
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -----------------------------
# utilitaires
# -----------------------------

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

def fit_powerlaw_ols(x, y):
    """
    Fit log10(y) = a + b log10(x) via OLS.
    Retourne: a, b, C=10^a, R2, rmse_log
    """
    lx = np.log10(x)
    ly = np.log10(y)
    b, a = np.polyfit(lx, ly, 1)  # ly ≈ b*lx + a
    ly_hat = a + b*lx
    rmse_log = np.sqrt(np.mean((ly - ly_hat)**2))
    r2 = r2_score(ly, ly_hat)
    return a, b, 10**a, r2, rmse_log

def fit_powerlaw_theilsen(x, y):
    """
    Theil-Sen sur log10(y) vs log10(x).
    """
    if not HAVE_SCIPY:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept, lo_slope, hi_slope = theilslopes(ly, lx, 0.95)
    ly_hat = intercept + slope*lx
    rmse_log = np.sqrt(np.mean((ly - ly_hat)**2))
    r2 = r2_score(ly, ly_hat)
    return intercept, slope, 10**intercept, r2, rmse_log

def plot_relation(x, y, xlab, ylab, title, out_png, fit_ols, fit_ts=None):
    a, b, C, r2, rmse_log = fit_ols

    # courbe fit sur domaine x
    xs = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    ys_ols = C * (xs**b)

    plt.figure(figsize=(6.8, 4.8))
    plt.scatter(x, y, s=18)
    plt.plot(xs, ys_ols, linewidth=2, label=f"OLS: y={C:.3g} x^{b:.3g} | R²log={r2:.3f}")

    if fit_ts is not None and np.isfinite(fit_ts[0]):
        a2, b2, C2, r22, _ = fit_ts
        ys_ts = C2 * (xs**b2)
        plt.plot(xs, ys_ts, linewidth=2, linestyle="--",
                 label=f"Theil–Sen: y={C2:.3g} x^{b2:.3g} | R²log={r22:.3f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# main
# -----------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_csv = os.path.join(script_dir, "2-SPARC Params Parsing Results Optimized", "SPARC_fit_summary_optimized_5p_occam.csv")
    out_dir = os.path.join(script_dir, "3a-SPARC Scaling Laws Results")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"CSV introuvable: {in_csv}")

    df = pd.read_csv(in_csv)

    # filtre standard
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    # Colonnes minimales attendues
    needed = ["Rmax_kpc", "lambda0_kpc", "Rc_kpc", "A", "zeta", "V_max_fixed"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # nettoyages
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)

    # logs: positifs
    df = df[(df["Rmax_kpc"] > 0) & (df["lambda0_kpc"] > 0) & (df["Rc_kpc"] > 0) & (df["V_max_fixed"] > 0)]
    # A peut être 0, zeta peut être proche de 0; on évite log dessus.
    # Ici on ne log-fit pas A ni zeta directement (sauf si tu veux une autre forme).

    relations = [
        # (xcol, ycol, xlabel, ylabel, title, filename)
        ("Rmax_kpc", "lambda0_kpc", "Rmax (kpc)", "lambda0 (kpc)",
         "Loi d'échelle: lambda0 = C * Rmax^beta", "lambda0_vs_Rmax.png"),

        ("Rmax_kpc", "Rc_kpc", "Rmax (kpc)", "Rc (kpc)",
         "Loi d'échelle: Rc = C * Rmax^beta", "Rc_vs_Rmax.png"),

        ("V_max_fixed", "lambda0_kpc", "Vmax fixé (km/s)", "lambda0 (kpc)",
         "Loi d'échelle: lambda0 = C * Vmax^beta", "lambda0_vs_Vmax.png"),

        ("V_max_fixed", "Rc_kpc", "Vmax fixé (km/s)", "Rc (kpc)",
         "Loi d'échelle: Rc = C * Vmax^beta", "Rc_vs_Vmax.png"),
    ]

    summary_rows = []

    for xcol, ycol, xlab, ylab, title, png in relations:
        x = df[xcol].to_numpy(float)
        y = df[ycol].to_numpy(float)

        # Fit OLS (log-log)
        a, b, C, r2, rmse_log = fit_powerlaw_ols(x, y)

        # Fit Theil–Sen si dispo
        ts = None
        if HAVE_SCIPY:
            ts = fit_powerlaw_theilsen(x, y)

        # plot
        out_png = os.path.join(out_dir, png)
        plot_relation(x, y, xlab, ylab, title, out_png, (a, b, C, r2, rmse_log), ts)

        summary_rows.append({
            "relation": f"{ycol} vs {xcol}",
            "method": "OLS (moindres carres ordinaires)",
            "a_log10": a,
            "beta": b,
            "C": C,
            "R2_log": r2,
            "RMSE_log10": rmse_log,
            "N": len(x),
        })

        if ts is not None and np.isfinite(ts[0]):
            a2, b2, C2, r22, rmse2 = ts
            summary_rows.append({
                "relation": f"{ycol} vs {xcol}",
                "method": "Theil–Sen (robuste)",
                "a_log10": a2,
                "beta": b2,
                "C": C2,
                "R2_log": r22,
                "RMSE_log10": rmse2,
                "N": len(x),
            })

    out_summary = os.path.join(out_dir, "scaling_laws_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    print("=" * 80)
    print("SCALING LAWS - TERMINE")
    print("=" * 80)
    print("Input :", in_csv)
    print("Output:", out_dir)
    print("Résumé:", out_summary)
    print("Note: les fits sont en log-log (lois de puissance).")


if __name__ == "__main__":
    main()
