#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPARC_ParseGalaxies_Optimized5P_Occam.py

Fit SPARC Table2 avec modèle second-ordre à 5 paramètres, contraint contre la sur-oscillation.

Choix structurants (d’après tes corrélations):
- alpha fixé à 0   -> phase simple phi(R)=2*pi*R/lambda0
- V_max fixé depuis les données (plateau robuste)

Paramètres libres (5):
  A        : amplitude
  zeta     : facteur d’amortissement (0<zeta<1)
  lambda0  : longueur d’onde spatiale (kpc)
  phi0     : phase initiale (rad)
  Rc       : rayon caractéristique de montée (kpc)

Anti-sur-oscillation:
- Pénalité douce: beta*(Rmax/lambda0)^2  (Occam)
- Borne: lambda0 >= frac_lambda0_min*Rmax

Sorties dans:
  ./SPARC Params Parsing Results Optimized/
    - SPARC_fit_results_optimized_5p_occam.json
    - SPARC_fit_summary_optimized_5p_occam.csv
    - Plots PNG (1 par galaxie)
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# =============================================================================
# CONFIG (ajuste ici)
# =============================================================================

# pénalité Occam: plus beta est grand, plus on favorise lambda0 grand
BETA_OCCAM = 0.15          # tester: 0.05, 0.10, 0.15, 0.25

# borne minimale sur lambda0: lambda0 >= frac * Rmax
FRAC_LAMBDA0_MIN = 0.25    # 0.25 => max ~4 oscillations sur le disque
LAMBDA0_MIN_FLOOR = 0.5    # plancher absolu en kpc

# estimation Vmax: fraction des derniers points pris pour plateau
PLATEAU_FRAC_LAST = 0.25
PLATEAU_MIN_POINTS = 4

# critère points min
MIN_POINTS_FIT = 10


# =============================================================================
# Parsing Table2
# =============================================================================

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
            e_Vobs = float(parts[4])
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
            galaxies[name]["e_Vobs"].append(e_Vobs)
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


# =============================================================================
# Vmax fixé: estimation plateau robuste
# =============================================================================

def estimate_Vmax_plateau(R, Vobs, frac_last=PLATEAU_FRAC_LAST, min_points=PLATEAU_MIN_POINTS):
    n = len(R)
    k = max(min_points, int(np.ceil(frac_last * n)))
    k = min(k, n)
    tail = Vobs[-k:]
    vmax = float(np.median(tail))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(Vobs))
    return vmax


# =============================================================================
# Modèle 5P (alpha=0, Vmax fixé)
# =============================================================================

def model_V_5p(R, Vmax_fixed, A, zeta, lambda0, phi0, Rc):
    R = np.asarray(R, dtype=float)
    R_safe = np.maximum(R, 1e-6)

    lambda0 = max(float(lambda0), 1e-8)
    Rc = max(float(Rc), 1e-8)

    omega0 = 2.0 * np.pi / lambda0
    f_R = R_safe / np.sqrt(R_safe**2 + Rc**2)

    # alpha=0 -> phase simple
    phi_R = 2.0 * np.pi * R_safe / lambda0

    zeta_safe = np.clip(zeta, 1e-6, 0.9999)
    sqrt_term = np.sqrt(1.0 - zeta_safe**2)

    osc = (A / sqrt_term) * np.exp(-zeta_safe * omega0 * R_safe) * np.sin(phi_R + phi0)

    return Vmax_fixed * f_R * (1.0 - osc)


# =============================================================================
# Coût: chi² réduit + pénalité Occam (priorise lambda0 grand)
# =============================================================================

def chi2_reduced_5p_occam(params_log, R, Vobs, e_Vobs, Vmax_fixed):
    """
    params_log = [A, zeta, ln(lambda0), phi0, ln(Rc)]
    """
    A, zeta, ln_lambda0, phi0, ln_Rc = params_log
    lambda0 = np.exp(ln_lambda0)
    Rc = np.exp(ln_Rc)

    V_model = model_V_5p(R, Vmax_fixed, A, zeta, lambda0, phi0, Rc)

    e_safe = np.maximum(e_Vobs, 1.0)
    res = (Vobs - V_model) / e_safe
    chi2 = float(np.sum(res**2))

    dof = len(R) - 5
    chi2_red = chi2 / dof if dof > 0 else chi2

    # pénalité anti haute fréquence: favorise lambda0 grand
    Rmax = float(np.max(R))
    Nosc = Rmax / max(lambda0, 1e-9)  # ~ nombre d'oscillations
    penalty = BETA_OCCAM * (Nosc ** 2)

    return chi2_red + penalty


# =============================================================================
# Fit (DE)
# =============================================================================

def fit_galaxy_5p_occam(R, Vobs, e_Vobs, Vmax_fixed, seed=42):
    n = len(R)
    R_max = float(np.max(R))

    # itérations/population adaptatives
    if n < 18:
        maxiter, popsize = 140, 14
    elif n < 35:
        maxiter, popsize = 200, 16
    else:
        maxiter, popsize = 240, 18

    # bornes
    lambda0_min = max(FRAC_LAMBDA0_MIN * R_max, LAMBDA0_MIN_FLOOR)
    lambda0_max = max(2.0 * R_max, lambda0_min * 1.05)

    Rc_min = 0.05
    Rc_max = max(0.7 * R_max, 0.06)

    bounds = [
        (0.0, 2.0),                              # A
        (0.01, 0.99),                            # zeta
        (np.log(lambda0_min), np.log(lambda0_max)),  # ln(lambda0)
        (0.0, 2.0 * np.pi),                      # phi0
        (np.log(Rc_min), np.log(Rc_max)),        # ln(Rc)
    ]

    result = differential_evolution(
        chi2_reduced_5p_occam,
        bounds=bounds,
        args=(R, Vobs, e_Vobs, Vmax_fixed),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        seed=seed,
        polish=True,
        updating="deferred",
        workers=-1
    )

    A, zeta, ln_lambda0, phi0, ln_Rc = result.x
    lambda0 = float(np.exp(ln_lambda0))
    Rc = float(np.exp(ln_Rc))

    V_fit = model_V_5p(R, Vmax_fixed, A, zeta, lambda0, phi0, Rc)
    rmse = float(np.sqrt(np.mean((Vobs - V_fit) ** 2)))

    ss_res = float(np.sum((Vobs - V_fit) ** 2))
    ss_tot = float(np.sum((Vobs - np.mean(Vobs)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "V_max_fixed": float(Vmax_fixed),
        "A": float(A),
        "zeta": float(zeta),
        "lambda0": lambda0,
        "phi0": float(phi0),
        "Rc": Rc,
        "chi2_reduced": float(result.fun),
        "R_squared": float(R2),
        "rmse": rmse,
        "n_points": int(len(R)),
        "success": bool(result.success),
        "message": str(result.message),
        "Rmax_kpc": float(R_max),
        "Nosc_Rmax_over_lambda0": float(R_max / max(lambda0, 1e-9)),
    }


# =============================================================================
# Plot
# =============================================================================

def plot_fit_5p(galaxy_name, R, Vobs, e_Vobs, fit_result, out_dir):
    Vmax = fit_result["V_max_fixed"]
    R_dense = np.linspace(float(np.min(R)), float(np.max(R)), 500)

    V_dense = model_V_5p(
        R_dense, Vmax,
        fit_result["A"], fit_result["zeta"], fit_result["lambda0"],
        fit_result["phi0"], fit_result["Rc"]
    )

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(R, Vobs, yerr=e_Vobs, fmt="o", markersize=4, capsize=2, label="Observations")
    plt.plot(R_dense, V_dense, linewidth=2, label="Modèle 5P (Occam λ0)")

    title = (f"{galaxy_name} | RMSE={fit_result['rmse']:.2f} km/s | R²={fit_result['R_squared']:.4f}\n"
             f"lambda0={fit_result['lambda0']:.2f} kpc | zeta={fit_result['zeta']:.3f} | "
             f"Nosc~Rmax/lambda0={fit_result['Nosc_Rmax_over_lambda0']:.2f}")
    plt.title(title)

    plt.xlabel("R (kpc)")
    plt.ylabel("V (km/s)")
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{galaxy_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input: même dossier que le script
    table2_path = os.path.join(script_dir, "Table2 SPARC Masse Models.mrt")

    # Output dir demandé
    out_dir = os.path.join(script_dir, "2-SPARC Params Parsing Results Optimized")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "SPARC_fit_results_optimized_5p_occam.json")
    out_csv  = os.path.join(out_dir, "SPARC_fit_summary_optimized_5p_occam.csv")

    print("=" * 90)
    print("SPARC FIT OPTIMISÉ 5 PARAMÈTRES (alpha=0, Vmax fixé) + Occam (priorise lambda0 grand)")
    print(f"Occam beta={BETA_OCCAM} | lambda0_min >= {FRAC_LAMBDA0_MIN}*Rmax, floor={LAMBDA0_MIN_FLOOR} kpc")
    print("=" * 90)

    if not os.path.exists(table2_path):
        raise FileNotFoundError(f"Fichier introuvable: {table2_path}")

    galaxies = parse_table2(table2_path)
    names = sorted(galaxies.keys())
    print(f"Galaxies trouvées: {len(names)}")

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

        Vmax_fixed = estimate_Vmax_plateau(R, Vobs)

        try:
            fit = fit_galaxy_5p_occam(R, Vobs, e, Vmax_fixed, seed=42)
            fit["D"] = float(data["D"])
            results[name] = fit
            n_fit += 1

            status = "OK" if fit["success"] else "WARN"
            print(f"[{i:3d}/{len(names)}] {name:12s} - {status}  "
                  f"chi2={fit['chi2_reduced']:.4f}  R²={fit['R_squared']:.4f}  "
                  f"RMSE={fit['rmse']:.2f}  lambda0={fit['lambda0']:.2f}  zeta={fit['zeta']:.3f}  "
                  f"Nosc~{fit['Nosc_Rmax_over_lambda0']:.2f}")

            plot_fit_5p(name, R, Vobs, e, fit, out_dir)

            rows.append({
                "name": name,
                "D_Mpc": fit["D"],
                "n_points": fit["n_points"],
                "Rmax_kpc": fit["Rmax_kpc"],
                "Nosc_Rmax_over_lambda0": fit["Nosc_Rmax_over_lambda0"],
                "chi2_reduced": fit["chi2_reduced"],
                "R_squared": fit["R_squared"],
                "RMSE_km_s": fit["rmse"],
                "V_max_fixed": fit["V_max_fixed"],
                "A": fit["A"],
                "zeta": fit["zeta"],
                "lambda0_kpc": fit["lambda0"],
                "phi0_rad": fit["phi0"],
                "Rc_kpc": fit["Rc"],
                "success": fit["success"],
            })

        except Exception as ex:
            print(f"[{i:3d}/{len(names)}] {name:12s} - ERROR: {ex}")

    # Sauvegarde
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Stats globales
    print("\n" + "=" * 90)
    print("STATISTIQUES GLOBALES (5P + Occam)")
    print("=" * 90)

    if rows:
        chi2 = np.array([r["chi2_reduced"] for r in rows], float)
        r2   = np.array([r["R_squared"] for r in rows], float)
        rmse = np.array([r["RMSE_km_s"] for r in rows], float)
        nosc = np.array([r["Nosc_Rmax_over_lambda0"] for r in rows], float)

        print(f"Galaxies fittées: {n_fit}")
        print(f"chi2 réduit moyen: {chi2.mean():.4f} ± {chi2.std():.4f}")
        print(f"R² moyen:          {r2.mean():.4f} ± {r2.std():.4f}")
        print(f"R² médian:         {np.median(r2):.4f}")
        print(f"RMSE moyen:        {rmse.mean():.2f} ± {rmse.std():.2f} km/s")
        print(f"RMSE médian:       {np.median(rmse):.2f} km/s")
        print(f"Nosc médian (Rmax/lambda0): {np.median(nosc):.2f}")

        best = sorted(rows, key=lambda x: x["R_squared"], reverse=True)[:10]
        print("\nTop 10 meilleurs fits (R²):")
        for b in best:
            print(f"  {b['name']:12s}  R²={b['R_squared']:.4f}  RMSE={b['RMSE_km_s']:.2f}  "
                  f"lambda0={b['lambda0_kpc']:.2f}  Nosc~{b['Nosc_Rmax_over_lambda0']:.2f}")
    else:
        print("Aucun fit réalisé.")

    print("\nSorties:")
    print(f"  Dossier: {out_dir}")
    print(f"  JSON   : {out_json}")
    print(f"  CSV    : {out_csv}")
    print("  PNG    : 1 par galaxie")


if __name__ == "__main__":
    main()
