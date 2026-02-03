"""
5-KiDS WeakLens Test_Optimized.py

OBJECTIF :
  Valider le modèle réduit (4 paramètres) sur les données de lentillage faible (KiDS).
  Utilise les lois d'échelle (C, beta) validées à l'étape 3a (SPARC).

DONNÉES :
  Attend des fichiers "Fig-3_Lensing-rotation-curves_Massbin-{b}.txt"
  dans le dossier "./Data_KiDS_Brouwer2021/".
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- CONSTANTES PHYSIQUES (Source: README Brouwer et al. 2021) ---
G_B21 = 4.52e-30     # pc^3/(Msun*s^2)
MPC_TO_PC = 1e6      # 1 Mpc = 10^6 pc
PC_TO_KM = 3.086e13  # 1 pc = 3.086e13 km

def load_data_strict(filepath):
    """Charge les données et calcule V_circ selon le protocole B21."""
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath, sep=r'\s+', comment="#", header=None)
    R_Mpc = df.iloc[:, 0].values
    ESD_t = df.iloc[:, 1].values
    err   = df.iloc[:, 3].values
    bias  = df.iloc[:, 4].values

    # Correction du biais
    ESD_corr = ESD_t / bias
    err_corr = err / bias

    # Vitesse circulaire (Eq.23)
    v_sq = 4 * G_B21 * ESD_corr * R_Mpc * MPC_TO_PC * (PC_TO_KM**2)
    v_obs = np.sqrt(np.maximum(v_sq, 0))
    
    # Propagation d'Erreur (STRICTE : pas de division par 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_err = v_obs * (err_corr / ESD_corr)
        v_err = np.where(np.isfinite(v_err), v_err, v_obs * 0.1)
    
    return R_Mpc * 1000.0, v_obs, v_err

def V_model_4p(r, A, zeta, phi0, Rc, Vmax_fixed, Rmax_fixed):
    """Modèle 4 paramètres avec Vmax et Rmax fixés."""
    r = np.maximum(np.asarray(r, float), 1e-6)
    lambda0 = 1.3 * Rmax_fixed
    omega0 = 2 * np.pi / lambda0
    
    # Rc peut être très petit (0.01 kpc) -> profil quasi plat dès le centre
    Rc = max(Rc, 0.01) 
    f = r / np.sqrt(r**2 + Rc**2)
    
    phi = 2 * np.pi * r / lambda0
    z = np.clip(zeta, 1e-6, 0.9999)
    s = np.sqrt(1 - z**2)
    
    osc = (A / s) * np.exp(-z * omega0 * r) * np.sin(phi + phi0)
    return Vmax_fixed * f * (1 - osc)

def cost_function(params, r, v_obs, v_err, Vmax_fixed, Rmax_fixed):
    """Somme des carrés des résidus."""
    v_mod = V_model_4p(r, *params, Vmax_fixed, Rmax_fixed)
    # Plancher d'erreur à 1.0 km/s
    weights = 1.0 / np.maximum(v_err, 1.0) 
    return np.sum(((v_obs - v_mod) * weights)**2)

def plot_result(R, v_obs, v_err, params, Vmax, Rmax, chi2_red, bin_num, output_dir):
    """Génère et sauvegarde le graphique du fit."""
    A, zeta, phi0, Rc = params
    
    # Création de la figure
    plt.figure(figsize=(10, 6))
    
    # Données
    plt.errorbar(R, v_obs, yerr=v_err, fmt='ko', capsize=3, label='KiDS Data (Brouwer+21)', alpha=0.7)
    
    # Modèle (courbe lisse)
    r_smooth = np.linspace(0, R.max() * 1.1, 500)
    v_fit = V_model_4p(r_smooth, *params, Vmax, Rmax)
    plt.plot(r_smooth, v_fit, 'r-', linewidth=2, label=f'Fit Modèle 4P')
    
    # Esthétique
    plt.xlabel("Rayon [kpc]", fontsize=12)
    plt.ylabel("Vitesse Circulaire [km/s]", fontsize=12)
    plt.title(f"Mass Bin {bin_num} : Fit Oscillation (Chi2_red = {chi2_red:.2f})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    # Boîte d'info paramètres
    info_text = (
        f"Vmax = {Vmax:.1f} km/s\n"
        f"A = {A:.3f}\n"
        f"$\\zeta$ = {zeta:.3f}\n"   # Double backslash pour \zeta
        f"$R_c$ = {Rc:.3f} kpc\n"
        f"$\\phi_0$ = {phi0:.2f}"    # Double backslash pour \phi
    )
    plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Sauvegarde
    filename = f"Fit_MassBin_{bin_num}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  > Graphique sauvegardé : {filename}")
    
if __name__ == "__main__":
    # --- Configuration des chemins ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "KiDS Brouwer Weak Lensing")
    
    # Dossier de sortie
    results_dir = os.path.join(script_dir, "5-KiDS WeakLens Results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"--- DÉBUT DE L'ANALYSE ---")
    print(f"Données : {data_dir}")
    print(f"Sortie  : {results_dir}\n")

    results_list = []

    # Boucle sur les 4 Mass Bins
    for b in range(1, 5):
        filename = f"Fig-3_Lensing-rotation-curves_Massbin-{b}.txt"
        filepath = os.path.join(data_dir, filename)
        
        data = load_data_strict(filepath)
        
        if data is None:
            print(f"[ERREUR] Fichier introuvable : {filename}")
            continue
            
        R, v_obs, v_err = data
        
        # Fixation Vmax / Rmax
        Vmax_fixed = np.mean(v_obs[-3:])
        Rmax_fixed = np.max(R)
        
        print(f"BIN {b} | Vmax: {Vmax_fixed:.1f} km/s | Points: {len(R)}")

        # Optimisation avec bornes Rc élargies (0.01 à 50 kpc)
        bounds = [(0.0, 2.0), (0.01, 0.99), (0.0, 2*np.pi), (0.01, 50.0)]
        
        res = differential_evolution(
            cost_function, bounds, 
            args=(R, v_obs, v_err, Vmax_fixed, Rmax_fixed),
            strategy='best1bin', maxiter=1000, seed=42
        )
        
        # Calcul Chi2 réduit
        dof = len(R) - 4
        chi2_red = res.fun / dof if dof > 0 else np.inf
        
        # Sauvegarde Graphique
        plot_result(R, v_obs, v_err, res.x, Vmax_fixed, Rmax_fixed, chi2_red, b, results_dir)
        
        # Stockage résultats
        results_list.append({
            "Bin": b,
            "Vmax": Vmax_fixed,
            "Chi2_red": chi2_red,
            "A": res.x[0],
            "Zeta": res.x[1],
            "Phi0": res.x[2],
            "Rc": res.x[3]
        })
        print(f"  > RÉSULTAT : Chi2_red = {chi2_red:.4f}")
        print(f"  > Params   : A={res.x[0]:.3f} | Rc={res.x[3]:.3f}\n")

    # Sauvegarde CSV récapitulatif
    df_res = pd.DataFrame(results_list)
    csv_path = os.path.join(results_dir, "Summary_Results.csv")
    df_res.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"--- ANALYSE TERMINÉE ---")
    print(f"Résumé sauvegardé dans : {csv_path}")