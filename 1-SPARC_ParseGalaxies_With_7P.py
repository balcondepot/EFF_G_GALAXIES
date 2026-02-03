#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-SPARC_ParseGalaxies_With_7P.py

Classification :
- REJECTED si n < 10 points de données
- OK sinon
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# =============================================================================
# Parsing Table2
# =============================================================================

def parse_table2(filepath):
    galaxies = {}
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) < 8: continue

        try:
            galaxy = parts[0]
            D = float(parts[1])
            R = float(parts[2])
            Vobs = float(parts[3])
            e_Vobs = float(parts[4])
            Vgas = float(parts[5])
            Vdisk = float(parts[6])
            Vbul = float(parts[7])

            if galaxy not in galaxies:
                galaxies[galaxy] = {
                    "D": D, "R": [], "Vobs": [], "e_Vobs": [],
                    "Vgas": [], "Vdisk": [], "Vbul": []
                }
            galaxies[galaxy]["R"].append(R)
            galaxies[galaxy]["Vobs"].append(Vobs)
            galaxies[galaxy]["e_Vobs"].append(e_Vobs)
            galaxies[galaxy]["Vgas"].append(Vgas)
            galaxies[galaxy]["Vdisk"].append(Vdisk)
            galaxies[galaxy]["Vbul"].append(Vbul)
        except:
            continue

    for g in galaxies:
        R = np.array(galaxies[g]["R"], dtype=float)
        idx = np.argsort(R)
        for k in ["R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul"]:
            galaxies[g][k] = np.array(galaxies[g][k], dtype=float)[idx]
    return galaxies

# =============================================================================
# Modèle V(R) second-ordre
# =============================================================================

def model_V(R, V_max, A, zeta, lambda0, phi0, Rc, alpha):
    R = np.asarray(R, dtype=float)
    R_safe = np.maximum(R, 1e-6)

    lambda0 = max(float(lambda0), 1e-8)
    Rc      = max(float(Rc),      1e-8)
    alpha   = max(float(alpha),   1e-8)

    omega0 = 2.0 * np.pi / lambda0
    f_R = R_safe / np.sqrt(R_safe**2 + Rc**2)
    phi_R = (2.0 * np.pi / alpha) * np.log((lambda0 + alpha * R_safe) / lambda0)

    zeta_safe = np.clip(zeta, 1e-6, 0.9999)
    sqrt_term = np.sqrt(1.0 - zeta_safe**2)

    osc = (A / sqrt_term) * np.exp(-zeta_safe * omega0 * R_safe) * np.sin(phi_R + phi0)
    V = V_max * f_R * (1.0 - osc)
    return V

# =============================================================================
# Coût + Fit
# =============================================================================

def chi2_reduced(params_log, R, Vobs, e_Vobs):
    V_max, A, zeta, ln_lambda0, phi0, ln_Rc, ln_alpha = params_log
    lambda0 = np.exp(ln_lambda0)
    Rc      = np.exp(ln_Rc)
    alpha   = np.exp(ln_alpha)

    V_model = model_V(R, V_max, A, zeta, lambda0, phi0, Rc, alpha)
    e_safe = np.maximum(e_Vobs, 1.0)
    res = (Vobs - V_model) / e_safe
    chi2 = float(np.sum(res**2))

    dof = len(R) - 7
    return chi2 / dof if dof > 0 else chi2

def fit_galaxy(R, Vobs, e_Vobs, seed=42):
    V_max_est = float(np.max(Vobs) * 1.10)
    R_max = float(np.max(R))
    n = len(R)

    if n < 18:
        maxiter, popsize = 160, 14
    elif n < 35:
        maxiter, popsize = 220, 16
    else:
        maxiter, popsize = 280, 18

    bounds = [
        (0.50 * V_max_est, 1.60 * V_max_est),
        (0.0, 2.0),
        (0.01, 0.99),
        (np.log(0.2), np.log(max(2.0 * R_max, 0.25))),
        (0.0, 2.0 * np.pi),
        (np.log(0.05), np.log(max(0.7 * R_max, 0.06))),
        (np.log(0.05), np.log(max(3.0 * R_max, 0.06))),
    ]

    result = differential_evolution(
        chi2_reduced, bounds=bounds, args=(R, Vobs, e_Vobs),
        strategy="best1bin", maxiter=maxiter, popsize=popsize,
        tol=1e-6, seed=seed, polish=True, updating="deferred", workers=-1
    )

    V_max, A, zeta, ln_lambda0, phi0, ln_Rc, ln_alpha = result.x
    lambda0 = float(np.exp(ln_lambda0))
    Rc      = float(np.exp(ln_Rc))
    alpha   = float(np.exp(ln_alpha))

    V_fit = model_V(R, V_max, A, zeta, lambda0, phi0, Rc, alpha)
    rmse = float(np.sqrt(np.mean((Vobs - V_fit)**2)))
    ss_res = float(np.sum((Vobs - V_fit)**2))
    ss_tot = float(np.sum((Vobs - np.mean(Vobs))**2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "V_max": float(V_max), "A": float(A), "zeta": float(zeta),
        "lambda0": lambda0, "phi0": float(phi0), "Rc": Rc, "alpha": alpha,
        "chi2_reduced": float(result.fun), "R_squared": float(R2),
        "rmse": rmse, "n_points": int(len(R)),
        "success": bool(result.success), "message": str(result.message),
    }

# =============================================================================
# Plot avec Classification (LOGIQUE MISE A JOUR)
# =============================================================================

def plot_fit(galaxy_name, R, Vobs, e_Vobs, fit_result, out_dir):
    """
    Trace et sauvegarde la courbe.
    Logique : 
    1. REJECT si RMSE > 15 km/s (QUOI QU'IL ARRIVE)
    2. OK si R2 > 0.90 et Chi2 < 10
    3. WARNING si R2 > 0.75
    4. REJECT sinon
    """
    
    n_pts = fit_result["n_points"]

    # Classification unique : n < 10 -> REJECTED
    if n_pts < 10:
        category = "REJECTED"
        color_title = "red"
        suffix = "(REJECTED)"
    else:
        category = "OK"
        color_title = "green"
        suffix = ""

    # Modèle dense pour l'affichage
    R_dense = np.linspace(float(np.min(R)), float(np.max(R)), 500)
    V_dense = model_V(
        R_dense, fit_result["V_max"], fit_result["A"], fit_result["zeta"],
        fit_result["lambda0"], fit_result["phi0"], fit_result["Rc"], fit_result["alpha"]
    )

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(R, Vobs, yerr=e_Vobs, fmt="o", markersize=4, capsize=2, label="Observations", color='black', alpha=0.6)
    plt.plot(R_dense, V_dense, linewidth=2, label="Modèle", color='blue')

    title_str = (f"{galaxy_name} [{category}] | RMSE={fit_result['rmse']:.2f} | "
                 f"R²={fit_result['R_squared']:.3f} | Chi2={fit_result['chi2_reduced']:.2f}")
    
    plt.title(title_str, color=color_title, fontweight='bold')
    plt.xlabel("R (kpc)")
    plt.ylabel("V (km/s)")
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{suffix}{galaxy_name}.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    
    return category

# =============================================================================
# Main
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    table2_path = os.path.join(script_dir, "Table2 SPARC Masse Models.mrt")
    out_json = os.path.join(script_dir, "SPARC_fit_results.json")
    out_csv  = os.path.join(script_dir, "SPARC_fit_summary.csv")
    plot_dir = os.path.join(script_dir, "1-SPARC Params Parsing Results")

    print("=" * 70)
    print("SPARC Analysis: CLASSIFICATION (OK / REJECTED)")
    print("REJECT Criteria: n < 10 data points")
    print("=" * 70)

    if not os.path.exists(table2_path):
        raise FileNotFoundError(f"Fichier introuvable: {table2_path}")

    galaxies = parse_table2(table2_path)
    names = sorted(galaxies.keys())
    print(f"Galaxies trouvées: {len(names)}")

    results = {}
    rows = []
    n_total = len(names)

    count_ok = 0
    count_rej = 0

    for i, name in enumerate(names, 1):
        data = galaxies[name]
        R = data["R"]
        Vobs = data["Vobs"]
        e = data["e_Vobs"]

        try:
            fit = fit_galaxy(R, Vobs, e, seed=42)
            fit["D"] = float(data["D"])
            results[name] = fit

            category = plot_fit(name, R, Vobs, e, fit, plot_dir)

            if category == "OK": count_ok += 1
            else: count_rej += 1

            print(f"[{i:3d}/{n_total}] {name:12s} - {category:8s} "
                  f"chi2={fit['chi2_reduced']:.4f}  R²={fit['R_squared']:.4f}  "
                  f"RMSE={fit['rmse']:.2f}  pts={len(R)}")

            rows.append({
                "name": name,
                "D_Mpc": fit["D"],
                "n_points": fit["n_points"],
                "chi2_reduced": fit["chi2_reduced"],
                "R_squared": fit["R_squared"],
                "RMSE_km_s": fit["rmse"],
                "V_max": fit["V_max"],
                "A": fit["A"],
                "zeta": fit["zeta"],
                "lambda0_kpc": fit["lambda0"],
                "phi0_rad": fit["phi0"],
                "Rc_kpc": fit["Rc"],
                "alpha_kpc": fit["alpha"],
                "success": fit["success"],
                "category": category 
            })

        except Exception as ex:
            print(f"[{i:3d}/{n_total}] {name:12s} - ERROR: {ex}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"Total traité : {len(rows)}")
    print(f"  - OK       : {count_ok}")
    print(f"  - REJECTED : {count_rej} (n < 10 points)")
    print(f"\nRésultats sauvegardés dans {out_csv}")
    print(f"Graphiques sauvegardés dans {plot_dir}")

if __name__ == "__main__":
    main()