import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution
from scipy.stats import chi2 as chi2_dist
from numpy.linalg import inv, pinv
import scipy.stats as st

# --- CONSTANTES ---
G_B21 = 4.52e-30
MPC_TO_PC = 1e6
PC_TO_KM = 3.086e13

# --- FONCTIONS UTILITAIRES ---
def load_all_bins(data_dir):
    """Charge les données R, V, ESD pour les 4 bins."""
    list_R, list_V, list_ESD = [], [], []
    for b in range(1, 5):
        fname = os.path.join(data_dir, f"Fig-3_Lensing-rotation-curves_Massbin-{b}.txt")
        if not os.path.exists(fname): continue
        df = pd.read_csv(fname, sep=r'\s+', comment="#", header=None)
        R = df.iloc[:, 0].values
        ESD = df.iloc[:, 1].values
        bias = df.iloc[:, 4].values
        
        # Calcul Vitesse (Eq. 23 Brouwer+21)
        v_obs = np.sqrt(np.maximum(4 * G_B21 * (ESD/bias) * R * MPC_TO_PC * PC_TO_KM**2, 0))
        list_R.append(R * 1000.0)
        list_V.append(v_obs)
        list_ESD.append(ESD/bias)
    return list_R, list_V, list_ESD

def build_full_covariance(cov_path, list_R, list_V, list_ESD):
    """Construit la matrice de covariance globale."""
    print(f"  > Lecture de : {os.path.basename(cov_path)}")
    try:
        df_cov = pd.read_csv(cov_path, sep=r'\s+', comment="#", header=None)
    except Exception as e:
        print(f"  [ERREUR] Lecture impossible : {e}")
        return None
    
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
            if len(sub) == ns[b1] * ns[b2]:
                block = sub.reshape(ns[b1], ns[b2])
                Cov_ESD[starts[b1]:starts[b1]+ns[b1], starts[b2]:starts[b2]+ns[b2]] = block

    J_diag = np.concatenate([np.where(np.isfinite(V/E), V/E, 0) for V, E in zip(list_V, list_ESD)])
    Cov_V = Cov_ESD * np.outer(J_diag, J_diag)
    return Cov_V

def null_cost_function(params_Rc, R_concat, V_obs_concat, Inv_Cov, starts, ns, fixed_params):
    """Calcule Chi2 pour Modèle Nul (A=0, seul Rc varie)."""
    model_parts = []
    for b in range(4):
        idx0 = starts[b]
        idx1 = idx0 + ns[b]
        r_bin = R_concat[idx0:idx1]
        Rc = params_Rc[b]
        Vmax, _ = fixed_params[b]
        
        # Modèle Nul : A=0 forcé
        Rc = max(Rc, 0.01)
        v_mod = Vmax * (r_bin / np.sqrt(r_bin**2 + Rc**2))
        model_parts.append(v_mod)
        
    res = V_obs_concat - np.concatenate(model_parts)
    return res.T @ Inv_Cov @ res

# --- MAIN ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "KiDS Brouwer Weak Lensing")
    
    # Recherche fichier covmatrix
    candidates = [f for f in os.listdir(data_dir) if "covmatrix.txt" in f]
    cov_file = None
    for f in candidates:
        if "Massbins_covmatrix.txt" in f: cov_file = os.path.join(data_dir, f); break
    if cov_file is None and candidates: cov_file = os.path.join(data_dir, candidates[0])

    if cov_file is None: raise FileNotFoundError("Pas de fichier covariance !")

    print(f"--- NULL TEST (A=0) & MODEL SELECTION ---")
    list_R, list_V, list_ESD = load_all_bins(data_dir)
    Cov_V = build_full_covariance(cov_file, list_R, list_V, list_ESD)

    # Inversion Robuste
    try:
        Inv_Cov = inv(Cov_V)
        print("  > Inversion Matrice : OK")
    except np.linalg.LinAlgError:
        print("  > Matrice singulière -> Utilisation Pseudo-inverse (pinv)")
        Inv_Cov = pinv(Cov_V)

    # Setup Optimisation
    V_total = np.concatenate(list_V)
    R_total = np.concatenate(list_R)
    ns = [len(x) for x in list_R]
    starts = [0] + list(np.cumsum(ns))
    fixed_params = [(np.mean(v[-3:]), np.max(r)) for v, r in zip(list_V, list_R)]

    print("\nOptimisation Modèle Nul (Rc libre)...")
    res_null = differential_evolution(
        null_cost_function, [(0.01, 50.0)] * 4,
        args=(R_total, V_total, Inv_Cov, starts, ns, fixed_params),
        strategy='best1bin', maxiter=500, popsize=15, seed=42
    )
    
    # --- RÉSULTATS ---
    chi2_null = res_null.fun

    # Lecture chi2 depuis scripts 6 et 7
    import json
    json6_path = os.path.join(script_dir, "results_script6.json")
    json7_path = os.path.join(script_dir, "results_script7.json")

    missing = []
    if os.path.exists(json6_path):
        with open(json6_path) as f:
            chi2_afree = json.load(f)["chi2"]
    else:
        missing.append("results_script6.json")
        chi2_afree = None

    if os.path.exists(json7_path):
        with open(json7_path) as f:
            chi2_univ = json.load(f)["chi2"]
    else:
        missing.append("results_script7.json")
        chi2_univ = None

    if missing:
        print(f"[ATTENTION] Fichier(s) manquant(s): {', '.join(missing)}")
        print("Executez les scripts 6 et 7 d'abord.")
        if chi2_afree is None and chi2_univ is None:
            print("Impossible de faire la comparaison. Arret.")
            exit(1)

    n_points = len(V_total)
    k_null  = 4    # 4 paramètres : Rc × 4 bins
    k_univ  = 13   # 13 paramètres : A + (zeta, phi0, Rc) × 4 bins
    k_afree = 16   # 16 paramètres : (A, zeta, phi0, Rc) × 4 bins

    print("\n" + "="*50)
    print("RESULTATS STATISTIQUES FINAUX")
    print("="*50)
    print(f"Chi2 Modele Nul (A=0)       : {chi2_null:.4f}  (k={k_null})")
    if chi2_univ is not None:
        print(f"Chi2 Modele Universal A     : {chi2_univ:.4f}  (k={k_univ})")
    if chi2_afree is not None:
        print(f"Chi2 Modele A-free          : {chi2_afree:.4f}  (k={k_afree})")

    # --- Test Null vs Universal A ---
    if chi2_univ is not None:
        delta_chi2_univ = chi2_null - chi2_univ
        delta_k_univ = k_univ - k_null
        p_univ = chi2_dist.sf(delta_chi2_univ, delta_k_univ)
        sigma_univ = st.norm.isf(p_univ / 2) if delta_chi2_univ > 0 else 0.0
        print(f"\n--- Null vs Universal A (dk={delta_k_univ}) ---")
        print(f"  dChi2 = {delta_chi2_univ:.2f}  ->  {sigma_univ:.2f} sigma")
    else:
        sigma_univ = None

    # --- Test Null vs A-free ---
    if chi2_afree is not None:
        delta_chi2_afree = chi2_null - chi2_afree
        delta_k_afree = k_afree - k_null
        p_afree = chi2_dist.sf(delta_chi2_afree, delta_k_afree)
        sigma_afree = st.norm.isf(p_afree / 2) if delta_chi2_afree > 0 else 0.0
        print(f"\n--- Null vs A-free (dk={delta_k_afree}) ---")
        print(f"  dChi2 = {delta_chi2_afree:.2f}  ->  {sigma_afree:.2f} sigma")

    if sigma_univ is not None and sigma_univ > 3:
        print("\n>>> INDICATION FORTE / DECOUVERTE <<<")

    # --- AIC / BIC ---
    print("\n" + "-"*50)
    print("CRITERES D'INFORMATION (AIC / BIC)")
    print("-"*50)

    aic_null  = chi2_null + 2 * k_null
    bic_null  = chi2_null + k_null * np.log(n_points)

    print(f"{'Modele':<20s} {'AIC':>8s} {'BIC':>8s}")
    print(f"{'Nul (A=0)':<20s} {aic_null:8.2f} {bic_null:8.2f}")

    if chi2_univ is not None:
        aic_univ = chi2_univ + 2 * k_univ
        bic_univ = chi2_univ + k_univ * np.log(n_points)
        print(f"{'Universal A':<20s} {aic_univ:8.2f} {bic_univ:8.2f}")

    if chi2_afree is not None:
        aic_afree = chi2_afree + 2 * k_afree
        bic_afree = chi2_afree + k_afree * np.log(n_points)
        print(f"{'A-free':<20s} {aic_afree:8.2f} {bic_afree:8.2f}")

    if chi2_univ is not None:
        delta_aic_univ = aic_null - aic_univ
        delta_bic_univ = bic_null - bic_univ
        print(f"\ndAIC (Nul - Universal)   : {delta_aic_univ:.2f}  (>0 = faveur oscillant)")
        print(f"dBIC (Nul - Universal)   : {delta_bic_univ:.2f}  (>0 = faveur oscillant)")

    if chi2_univ is not None and chi2_afree is not None:
        delta_aic_afree = aic_afree - aic_univ
        print(f"dAIC (A-free - Universal) : {delta_aic_afree:.2f}  (>0 = Universal prefere)")

    print("\n" + "="*50)
    print("Paramètres Rc (Modèle Nul) :")
    for b, rc in enumerate(res_null.x):
        print(f"Bin {b+1} : {rc:.2f} kpc")