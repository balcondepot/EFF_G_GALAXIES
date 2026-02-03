#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPARC_ParseGalaxies_4P_FixedLambda.py

Modele 4 parametres avec loi d'echelle:
    lambda0 = 1.3 * RHI   (RHI = rayon HI a 1 Msun/pc2, Table1 SPARC)

Parametres libres (4):
    A, zeta, phi0, Rc
Parametres fixes:
    Vmax fixe par plateau (mediane des derniers points)
    alpha = 0 (phase simple)
    lambda0 = 1.3 * RHI

Sorties:
  ./4-SPARC Params Parsing Results 4P FixedLambda/
    - SPARC_fit_results_4p_fixedlambda.json
    - SPARC_fit_summary_4p_fixedlambda.csv
    - PNG par galaxie
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# =========================
# CONFIG
# =========================
LAMBDA_SCALE = 1.3
MIN_POINTS_FIT = 10

PLATEAU_FRAC_LAST = 0.25
PLATEAU_MIN_POINTS = 4


# =========================
# Parse Table1 (pour RHI)
# =========================
def parse_table1_rhi(filepath):
    """Extrait RHI (bytes 82-86) par galaxie depuis Table1."""
    rhi_dict = {}
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
    for line in lines[data_start:]:
        if len(line) < 86:
            continue
        try:
            name = line[0:11].strip()
            rhi = float(line[81:86].strip())
            if rhi > 0:
                rhi_dict[name] = rhi
        except (ValueError, IndexError):
            continue
    return rhi_dict


# =========================
# Parse Table2
# =========================
def parse_table2(filepath):
    galaxies = {}
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        try:
            name = parts[0]
            D = float(parts[1])
            R = float(parts[2])
            Vobs = float(parts[3])
            e = float(parts[4])
            Vgas = float(parts[5])
            Vdisk = float(parts[6])
            Vbul = float(parts[7])

            if name not in galaxies:
                galaxies[name] = {
                    "D": D,
                    "R": [], "Vobs": [], "e_Vobs": [],
                    "Vgas": [], "Vdisk": [], "Vbul": []
                }

            galaxies[name]["R"].append(R)
            galaxies[name]["Vobs"].append(Vobs)
            galaxies[name]["e_Vobs"].append(e)
            galaxies[name]["Vgas"].append(Vgas)
            galaxies[name]["Vdisk"].append(Vdisk)
            galaxies[name]["Vbul"].append(Vbul)

        except (ValueError, IndexError):
            continue

    # numpy + tri
    for g in galaxies:
        R = np.array(galaxies[g]["R"], dtype=float)
        idx = np.argsort(R)
        for k in ["R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul"]:
            galaxies[g][k] = np.array(galaxies[g][k], dtype=float)[idx]

    return galaxies


# =========================
# Vmax plateau robuste
# =========================
def estimate_Vmax_plateau(R, Vobs, frac_last=PLATEAU_FRAC_LAST, min_points=PLATEAU_MIN_POINTS):
    n = len(R)
    k = max(min_points, int(np.ceil(frac_last * n)))
    k = min(k, n)
    tail = Vobs[-k:]
    vmax = float(np.median(tail))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(Vobs))
    return vmax


# =========================
# Modele 4P (lambda0 fixe)
# =========================
def model_V_4p(R, Vmax, A, zeta, phi0, Rc, lambda0):
    R = np.asarray(R, dtype=float)
    R_safe = np.maximum(R, 1e-6)

    Rc = max(float(Rc), 1e-8)
    lambda0 = max(float(lambda0), 1e-8)

    omega0 = 2.0 * np.pi / lambda0

    # enveloppe
    f_R = R_safe / np.sqrt(R_safe**2 + Rc**2)

    # phase (alpha=0)
    phi_R = 2.0 * np.pi * R_safe / lambda0

    zeta = np.clip(zeta, 1e-6, 0.9999)
    sqrt_term = np.sqrt(1.0 - zeta**2)

    osc = (A / sqrt_term) * np.exp(-zeta * omega0 * R_safe) * np.sin(phi_R + phi0)

    return Vmax * f_R * (1.0 - osc)


# =========================
# Cout
# =========================
def chi2_reduced_4p(params, R, Vobs, e, Vmax, lambda0):
    A, zeta, phi0, Rc = params
    V_model = model_V_4p(R, Vmax, A, zeta, phi0, Rc, lambda0)

    e_safe = np.maximum(e, 1.0)
    res = (Vobs - V_model) / e_safe
    chi2 = float(np.sum(res**2))

    dof = len(R) - 4
    return chi2 / dof if dof > 0 else chi2


# =========================
# Fit galaxie
# =========================
def fit_galaxy_4p(R, Vobs, e, Vmax, lambda0, seed=42):
    Rmax = float(np.max(R))

    # bounds
    bounds = [
        (0.0, 2.0),                 # A
        (0.01, 0.99),               # zeta
        (0.0, 2.0 * np.pi),         # phi0
        (0.05, max(0.7 * Rmax, 0.06))  # Rc
    ]

    # adaptatif leger
    n = len(R)
    if n < 18:
        maxiter, popsize = 140, 14
    elif n < 35:
        maxiter, popsize = 200, 16
    else:
        maxiter, popsize = 240, 18

    result = differential_evolution(
        chi2_reduced_4p,
        bounds=bounds,
        args=(R, Vobs, e, Vmax, lambda0),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        seed=seed,
        polish=True,
        updating="deferred",
        workers=-1
    )

    A, zeta, phi0, Rc = result.x
    V_fit = model_V_4p(R, Vmax, A, zeta, phi0, Rc, lambda0)

    rmse = float(np.sqrt(np.mean((Vobs - V_fit) ** 2)))

    ss_res = float(np.sum((Vobs - V_fit) ** 2))
    ss_tot = float(np.sum((Vobs - np.mean(Vobs)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "V_max_fixed": float(Vmax),
        "lambda0_fixed": float(lambda0),
        "A": float(A),
        "zeta": float(zeta),
        "phi0": float(phi0),
        "Rc": float(Rc),
        "chi2_reduced": float(result.fun),
        "R_squared": float(R2),
        "rmse": rmse,
        "n_points": int(len(R)),
        "success": bool(result.success),
        "message": str(result.message),
        "Rmax_kpc": float(Rmax),
        "Nosc_Rmax_over_lambda0": float(Rmax / max(lambda0, 1e-9)),
    }


# =========================
# Plot
# =========================
def plot_fit(name, R, Vobs, e, fit, RHI, outdir):
    Vmax = fit["V_max_fixed"]
    lambda0 = fit["lambda0_fixed"]

    R_dense = np.linspace(float(np.min(R)), float(np.max(R)), 500)
    V_dense = model_V_4p(
        R_dense, Vmax,
        fit["A"], fit["zeta"], fit["phi0"], fit["Rc"],
        lambda0
    )

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(R, Vobs, yerr=e, fmt="o", markersize=4, capsize=2, label="Observations")
    plt.plot(R_dense, V_dense, linewidth=2, label="Modele 4P (lambda0=1.3*RHI)")

    plt.title(
        f"{name} | RMSE={fit['rmse']:.2f} km/s | R2={fit['R_squared']:.4f}\n"
        f"RHI={RHI:.2f} kpc | lambda0={lambda0:.2f} kpc | zeta={fit['zeta']:.3f} | A={fit['A']:.3f}"
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("V (km/s)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=140)
    plt.close()


# =========================
# Main
# =========================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input
    table1_path = os.path.join(script_dir, "Table1 SPARC Galaxy Sample.mrt")
    table2_path = os.path.join(script_dir, "Table2 SPARC Masse Models.mrt")

    for p in [table1_path, table2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    out_dir = os.path.join(script_dir, "4-SPARC Params Parsing Results 4P FixedLambda")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "SPARC_fit_results_4p_fixedlambda.json")
    out_csv  = os.path.join(out_dir, "SPARC_fit_summary_4p_fixedlambda.csv")

    # Parse
    rhi_dict = parse_table1_rhi(table1_path)
    galaxies = parse_table2(table2_path)
    names = sorted(galaxies.keys())

    print("=" * 90)
    print("SPARC FIT 4 PARAMETRES: lambda0 = 1.3 * RHI (fixe) | Vmax fixe plateau")
    print(f"Galaxies Table2: {len(names)} | RHI disponible: {len(rhi_dict)}")
    print("=" * 90)

    results = {}
    rows = []
    n_fit = 0

    for i, name in enumerate(names, 1):
        data = galaxies[name]
        R = data["R"]
        Vobs = data["Vobs"]
        e = data["e_Vobs"]

        if len(R) < MIN_POINTS_FIT:
            print(f"[{i:3d}/{len(names)}] {name:12s} - SKIP (n={len(R)} < {MIN_POINTS_FIT})")
            continue

        if name not in rhi_dict:
            print(f"[{i:3d}/{len(names)}] {name:12s} - SKIP (RHI non disponible)")
            continue

        Vmax = estimate_Vmax_plateau(R, Vobs)
        RHI = rhi_dict[name]
        lambda0 = LAMBDA_SCALE * RHI

        try:
            fit = fit_galaxy_4p(R, Vobs, e, Vmax, lambda0, seed=42)
            fit["D"] = float(data["D"])
            fit["RHI_kpc"] = float(RHI)
            results[name] = fit
            n_fit += 1

            status = "OK" if fit["success"] else "WARN"
            print(f"[{i:3d}/{len(names)}] {name:12s} - {status}  "
                  f"chi2={fit['chi2_reduced']:.4f}  R2={fit['R_squared']:.4f}  "
                  f"RMSE={fit['rmse']:.2f}  RHI={RHI:.2f}  lambda0={fit['lambda0_fixed']:.2f}  "
                  f"Nosc~{fit['Nosc_Rmax_over_lambda0']:.2f}")

            plot_fit(name, R, Vobs, e, fit, RHI, out_dir)

            rows.append({
                "name": name,
                "D_Mpc": fit["D"],
                "n_points": fit["n_points"],
                "Rmax_kpc": fit["Rmax_kpc"],
                "RHI_kpc": fit["RHI_kpc"],
                "lambda0_kpc_fixed": fit["lambda0_fixed"],
                "Nosc_Rmax_over_lambda0": fit["Nosc_Rmax_over_lambda0"],
                "chi2_reduced": fit["chi2_reduced"],
                "R_squared": fit["R_squared"],
                "RMSE_km_s": fit["rmse"],
                "V_max_fixed": fit["V_max_fixed"],
                "A": fit["A"],
                "zeta": fit["zeta"],
                "phi0_rad": fit["phi0"],
                "Rc_kpc": fit["Rc"],
                "success": fit["success"],
            })

        except Exception as ex:
            print(f"[{i:3d}/{len(names)}] {name:12s} - ERROR: {ex}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # stats
    print("\n" + "=" * 90)
    print("STATISTIQUES GLOBALES (4P fixed lambda0 = 1.3*RHI)")
    print("=" * 90)

    if rows:
        rmse = np.array([r["RMSE_km_s"] for r in rows], float)
        r2 = np.array([r["R_squared"] for r in rows], float)
        print(f"Galaxies fittees: {n_fit}")
        print(f"RMSE moyen:   {rmse.mean():.2f} +/- {rmse.std():.2f} km/s")
        print(f"RMSE median:  {np.median(rmse):.2f} km/s")
        print(f"R2 moyen:     {r2.mean():.4f} +/- {r2.std():.4f}")
        print(f"R2 median:    {np.median(r2):.4f}")
    else:
        print("Aucun fit realise.")

    print("\nSorties:", out_dir)
    print(" JSON:", out_json)
    print(" CSV :", out_csv)
    print(" PNG : 1 par galaxie")


if __name__ == "__main__":
    main()