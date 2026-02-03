import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution
from numpy.linalg import inv, pinv

# --- CONSTANTES ---
# G en (km/s)^2 * kpc / M_sun
# G ~ 4.301e-6
G_GRAV = 4.301e-6 
G_B21 = 4.52e-30     # Constante conversion Brouwer
MPC_TO_PC = 1e6
PC_TO_KM = 3.086e13

# Masses stellaires moyennes approximatives par bin (Log M_sun)
# Bin 1: 8.5-10.3 -> ~9.4
# Bin 2: 10.3-10.6 -> ~10.45
# Bin 3: 10.6-10.8 -> ~10.7
# Bin 4: 10.8-11.0 -> ~10.9
MEAN_LOG_MSTAR = [9.4, 10.45, 10.7, 10.9]

# --- CHARGEMENT DONNÉES ---
def load_all_bins(data_dir):
    list_R, list_V, list_ESD = [], [], []
    for b in range(1, 5):
        fname = os.path.join(data_dir, f"Fig-3_Lensing-rotation-curves_Massbin-{b}.txt")
        if not os.path.exists(fname): continue
        df = pd.read_csv(fname, sep=r'\s+', comment="#", header=None)
        R = df.iloc[:, 0].values
        ESD = df.iloc[:, 1].values
        bias = df.iloc[:, 4].values
        v_obs = np.sqrt(np.maximum(4 * G_B21 * (ESD/bias) * R * MPC_TO_PC * PC_TO_KM**2, 0))
        list_R.append(R * 1000.0) # kpc
        list_V.append(v_obs)
        list_ESD.append(ESD/bias)
    return list_R, list_V, list_ESD

def build_full_covariance(cov_path, list_R, list_V, list_ESD):
    try: df_cov = pd.read_csv(cov_path, sep=r'\s+', comment="#", header=None)
    except: return None
    
    vals = df_cov.iloc[:, 4].values
    bias_sq = np.where(df_cov.iloc[:, 6].values==0, 1.0, df_cov.iloc[:, 6].values)
    cov_corr = vals / bias_sq
    
    bin_labels = np.sort(np.unique(df_cov.iloc[:, 0].values))
    bin_map = {val: i for i, val in enumerate(bin_labels)}
    m_idx = np.vectorize(bin_map.get)(df_cov.iloc[:, 0].values)
    n_idx = np.vectorize(bin_map.get)(df_cov.iloc[:, 1].values)
    
    ns = [len(x) for x in list_R]
    N_total = sum(ns)
    Cov_ESD = np.zeros((N_total, N_total))
    starts = [0] + list(np.cumsum(ns))
    
    for b1 in range(4):
        for b2 in range(4):
            mask = (m_idx == b1) & (n_idx == b2)
            sub = cov_corr[mask]
            if len(sub) == ns[b1]*ns[b2]:
                Cov_ESD[starts[b1]:starts[b1]+ns[b1], starts[b2]:starts[b2]+ns[b2]] = sub.reshape(ns[b1], ns[b2])
                
    J_diag = np.concatenate([np.where(np.isfinite(V/E), V/E, 0) for V, E in zip(list_V, list_ESD)])
    return Cov_ESD * np.outer(J_diag, J_diag)

# --- MODÈLE NFW + BARYONS ---
def nfw_velocity_sq(r, M200, c):
    """Vitesse au carré due au Halo NFW"""
    h = 0.7
    # Densité critique en M_sun / kpc^3
    rho_crit = 2.775e11 * h**2 * 1e-9 
    
    # Rayon Viriel R200
    R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1/3)
    Rs = R200 / c
    x = r / Rs
    
    # Masse incluse NFW
    func = np.log(1+x) - x/(1+x)
    func_c = np.log(1+c) - c/(1+c)
    M_r = M200 * func / func_c
    
    return G_GRAV * M_r / r

def nfw_cost_function(params, R_concat, V_obs_concat, Inv_Cov, starts, ns):
    model_parts = []
    
    for b in range(4):
        idx0 = starts[b]
        idx1 = idx0 + ns[b]
        r_bin = R_concat[idx0:idx1]
        
        # 2 paramètres par bin : log10(M200) et Concentration c
        logM200 = params[2*b]
        c = params[2*b + 1]
        
        M200 = 10**logM200
        M_star = 10**MEAN_LOG_MSTAR[b]
        
        # V_tot = sqrt( V_NFW^2 + V_star^2 )
        v2_nfw = nfw_velocity_sq(r_bin, M200, c)
        v2_star = G_GRAV * M_star / r_bin
        
        v_mod = np.sqrt(v2_nfw + v2_star)
        model_parts.append(v_mod)
        
    res = V_obs_concat - np.concatenate(model_parts)
    return res.T @ Inv_Cov @ res

# --- MAIN ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "KiDS Brouwer Weak Lensing")
    
    # Recherche Covariance
    candidates = [f for f in os.listdir(data_dir) if "covmatrix.txt" in f]
    cov_file = None
    for f in candidates:
        if "Massbins_covmatrix.txt" in f: cov_file = os.path.join(data_dir, f); break
    if cov_file is None and candidates: cov_file = os.path.join(data_dir, candidates[0])

    print("--- COMPARAISON ΛCDM (NFW) vs TON MODÈLE ---")
    list_R, list_V, list_ESD = load_all_bins(data_dir)
    Cov_V = build_full_covariance(cov_file, list_R, list_V, list_ESD)

    try: Inv_Cov = inv(Cov_V)
    except: Inv_Cov = pinv(Cov_V)

    V_total = np.concatenate(list_V)
    R_total = np.concatenate(list_R)
    ns = [len(x) for x in list_R]
    starts = [0] + list(np.cumsum(ns))
    
    # Optimisation NFW (8 paramètres : 4 masses, 4 concentrations)
    # Bornes : logM200 [10, 15], c [1, 20]
    bounds = [(10.0, 15.0), (1.0, 20.0)] * 4
    
    print("Fitting Standard NFW Model (8 parameters)...")
    res_nfw = differential_evolution(
        nfw_cost_function, bounds,
        args=(R_total, V_total, Inv_Cov, starts, ns),
        strategy='best1bin', maxiter=200, popsize=15, seed=42
    )
    
    # --- RÉSULTATS COMPARATIFS ---
    chi2_nfw = res_nfw.fun
    k_nfw = 8  # 2 params * 4 bins
    
    # TES RÉSULTATS (Modèle Universel)
    chi2_univ = 15.71
    k_univ = 13
    
    n = 60 # Nombre de points
    
    # Calcul AIC
    aic_nfw = chi2_nfw + 2 * k_nfw
    aic_univ = chi2_univ + 2 * k_univ
    
    print("\n" + "="*60)
    print(f"{'MODÈLE':<20} | {'Chi2':<10} | {'Params (k)':<10} | {'AIC':<10}")
    print("-" * 60)
    print(f"{'ΛCDM (NFW)':<20} | {chi2_nfw:<10.2f} | {k_nfw:<10} | {aic_nfw:<10.2f}")
    print(f"{'Universal A':<20} | {chi2_univ:<10.2f} | {k_univ:<10} | {aic_univ:<10.2f}")
    print("-" * 60)
    
    delta_aic = aic_nfw - aic_univ
    print(f"Delta AIC (NFW - Univ) : {delta_aic:.2f}")
    
    if delta_aic > 0:
        print(">>> TON MODÈLE EST MEILLEUR QUE NFW (AIC préférentiel)")
        print(f">>> Preuve statistique : {np.abs(delta_aic):.2f}")
    else:
        print(">>> NFW est meilleur statistiquement (plus simple).")
        print(">>> Argumentez sur la physique (pas de matière noire).")

    print("\nParamètres NFW ajustés :")
    for b in range(4):
        m = res_nfw.x[2*b]
        c = res_nfw.x[2*b+1]
        print(f"Bin {b+1}: logM200={m:.2f}, c={c:.2f}")