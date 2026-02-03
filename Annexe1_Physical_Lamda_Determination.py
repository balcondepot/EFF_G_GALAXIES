import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# Constante de Gravitation (kpc (km/s)^2 Msun^-1)
G = 4.302e-6  

def parse_table2(filepath):
    """Parse le fichier MRT de SPARC pour extraire les profils de vitesse."""
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
            name, D, R, Vobs, eVobs, Vgas, Vdisk, Vbul = parts[0], *map(float, parts[1:8])
            if name not in galaxies:
                galaxies[name] = {"D": D, "R": [], "Vobs": [], "e_Vobs": [], "Vgas": [], "Vdisk": [], "Vbul": []}
            for k, v in zip(["R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul"], [R, Vobs, eVobs, Vgas, Vdisk, Vbul]):
                galaxies[name][k].append(v)
        except: continue
    for g in galaxies:
        idx = np.argsort(galaxies[g]["R"])
        for k in galaxies[g]:
            if k != "D": galaxies[g][k] = np.array(galaxies[g][k])[idx]
    return galaxies

def get_Mc(R_arr, Vgas, Vdisk, Vbul, Rc):
    """Calcule la masse baryonique à Rc par interpolation des profils SPARC."""
    f_gas = interp1d(R_arr, Vgas, fill_value="extrapolate")
    f_disk = interp1d(R_arr, Vdisk, fill_value="extrapolate")
    f_bul = interp1d(R_arr, Vbul, fill_value="extrapolate")
    # M_bar(R) = R * (Vgas^2 + Vdisk^2 + Vbul^2) / G
    Vbar2 = f_gas(Rc)**2 + f_disk(Rc)**2 + f_bul(Rc)**2
    return max((Rc * Vbar2) / G, 1e-3)

def model_V_dynamic(R, A, zeta, phi0, Rc, k, Mc, Vmax):
    """Modèle de gravité effective avec Lambda Dynamique."""
    lam = k * np.sqrt(Rc**3 / (G * Mc))
    omega0 = 2.0 * np.pi / lam
    f_R = R / np.sqrt(R**2 + Rc**2)
    phi_R = 2.0 * np.pi * R / lam
    
    zeta_safe = np.clip(zeta, 1e-6, 0.99)
    sqrt_term = np.sqrt(1.0 - zeta_safe**2)
    osc = (A / sqrt_term) * np.exp(-zeta_safe * omega0 * R) * np.sin(phi_R + phi0)
    return Vmax * f_R * (1.0 - osc)

def chi2_reduced(params, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, Vmax):
    A, zeta, phi0, Rc, k = params
    Mc = get_Mc(R, Vgas, Vdisk, Vbul, Rc)
    V_mod = model_V_dynamic(R, A, zeta, phi0, Rc, k, Mc, Vmax)
    chi2 = np.sum(((Vobs - V_mod) / np.maximum(e_Vobs, 1.0))**2)
    dof = len(R) - 5
    return chi2 / dof if dof > 0 else chi2

def fit_galaxy(name, data):
    R, Vobs, eVobs = data["R"], data["Vobs"], data["e_Vobs"]
    Vgas, Vdisk, Vbul = data["Vgas"], data["Vdisk"], data["Vbul"]
    Vmax_fixed = float(np.median(Vobs[-3:])) # Fixé sur le plateau
    
    bounds = [(0.01, 0.8), (0.01, 0.95), (0, 2*np.pi), (0.1, np.max(R)), (0.1, 5.0)]
    
    res = differential_evolution(chi2_reduced, bounds, 
                                 args=(R, Vobs, eVobs, Vgas, Vdisk, Vbul, Vmax_fixed),
                                 tol=1e-5, strategy='best1bin')
    
    A, zeta, phi0, Rc, k = res.x
    Mc = get_Mc(R, Vgas, Vdisk, Vbul, Rc)
    return {"A": A, "zeta": zeta, "phi0": phi0, "Rc": Rc, "k": k, "Mc": Mc, 
            "Vmax": Vmax_fixed, "chi2_red": res.fun, "success": res.success}

# Script principal d'exécution
if __name__ == "__main__":
    galaxies = parse_table2("Table2 SPARC Masse Models.mrt")
    results = []
    for name in sorted(galaxies.keys()):
        print(f"Fitting {name}...")
        try:
            fit = fit_galaxy(name, galaxies[name])
            fit["name"] = name
            results.append(fit)
        except Exception as e: print(f"Error {name}: {e}")
    
    pd.DataFrame(results).to_csv("SPARC_DynamicLambda_Results.csv", index=False)
    print("Fini. Résultats dans SPARC_DynamicLambda_Results.csv")