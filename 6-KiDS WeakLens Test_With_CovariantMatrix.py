import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution
from numpy.linalg import inv, pinv

# --- CONSTANTES ---
G_B21 = 4.52e-30
MPC_TO_PC = 1e6
PC_TO_KM = 3.086e13

def load_all_bins(data_dir):
    """Charge les profils (R, V, ESD) pour les 4 bins."""
    list_R, list_V, list_ESD = [], [], []
    
    # On suppose que les fichiers profils s'appellent Massbin-1, -2, etc.
    # Si vos noms sont différents, adaptez ici.
    for b in range(1, 5):
        fname = os.path.join(data_dir, f"Fig-3_Lensing-rotation-curves_Massbin-{b}.txt")
        if not os.path.exists(fname):
            print(f"[ERREUR] Profil manquant : {fname}")
            continue
            
        df = pd.read_csv(fname, sep=r'\s+', comment="#", header=None)
        R_Mpc = df.iloc[:, 0].values
        ESD_t = df.iloc[:, 1].values
        bias  = df.iloc[:, 4].values
        
        # Calcul Vitesse
        ESD_corr = ESD_t / bias
        v_sq = 4 * G_B21 * ESD_corr * R_Mpc * MPC_TO_PC * (PC_TO_KM**2)
        v_obs = np.sqrt(np.maximum(v_sq, 0))
        
        list_R.append(R_Mpc * 1000.0)
        list_V.append(v_obs)
        list_ESD.append(ESD_corr)
        
    return list_R, list_V, list_ESD

def build_full_covariance(cov_path, list_R, list_V, list_ESD):
    """
    Lit le fichier UNIQUE et construit la matrice par blocs.
    Gère le mapping des labels de bins (ex: 8.5 -> 0, 10.3 -> 1).
    """
    print(f"Chargement covariance : {os.path.basename(cov_path)}")
    
    # 1. Lecture
    # Colonnes : m_min, n_min, R_i, R_j, cov, corr, bias_prod
    try:
        df_cov = pd.read_csv(cov_path, sep=r'\s+', comment="#", header=None)
    except Exception as e:
        print(f"Erreur lecture : {e}")
        return None

    vals = df_cov.iloc[:, 4].values # Col 5 : Covariance
    bias_sq = df_cov.iloc[:, 6].values # Col 7 : Bias product
    
    # Correction Bias (cov_corr = cov / bias_pair)
    bias_sq = np.where(bias_sq==0, 1.0, bias_sq)
    cov_vals = vals / bias_sq
    
    # 2. Mapping des Bins
    # Le fichier utilise des floats (8.5, 10.3...) pour identifier les bins
    m_raw = df_cov.iloc[:, 0].values
    n_raw = df_cov.iloc[:, 1].values
    
    unique_bins = np.sort(np.unique(m_raw))
    print(f"Labels de bins trouvés dans covmatrix : {unique_bins}")
    
    if len(unique_bins) != 4:
        print("ATTENTION : Le nombre de bins dans la matrice != 4")
    
    # Dictionnaire de mapping : val -> index (0, 1, 2, 3)
    bin_map = {val: i for i, val in enumerate(unique_bins)}
    
    # 3. Remplissage Matrice Géante
    ns = [len(x) for x in list_R] # [15, 15, 15, 15]
    N_total = sum(ns)
    Cov_ESD = np.zeros((N_total, N_total))
    
    # Indices de démarrage pour chaque bloc 0..3
    starts = [0] + list(np.cumsum(ns))
    
    # Itération sur les lignes du fichier de cov est trop lente ?
    # Non, Pandas est rapide. On va utiliser le mapping vectorisé.
    
    # On assigne à chaque ligne du fichier son bloc (i_blk, j_blk)
    m_idx = np.vectorize(bin_map.get)(m_raw)
    n_idx = np.vectorize(bin_map.get)(n_raw)
    
    # On suppose que dans chaque bloc (m, n), les données sont triées par R_i, R_j
    # C'est la structure standard. On va remplir bloc par bloc.
    
    for b1 in range(4):
        for b2 in range(4):
            # Filtrer le DataFrame pour ce couple de bins
            mask = (m_idx == b1) & (n_idx == b2)
            sub_cov = cov_vals[mask]
            
            # Taille attendue
            n_rows = ns[b1]
            n_cols = ns[b2]
            
            if len(sub_cov) == n_rows * n_cols:
                # C'est un bloc complet -> on reshape
                block = sub_cov.reshape((n_rows, n_cols))
                
                # Placement dans la grande matrice
                istart = starts[b1]
                jstart = starts[b2]
                Cov_ESD[istart:istart+n_rows, jstart:jstart+n_cols] = block
            else:
                # Parfois la matrice ne contient que les blocs diagonaux ou triangulaires sup?
                # Si c'est vide et b1 != b2, on assume 0 (pas de corrélation)
                pass

    # 4. Conversion ESD -> Vitesse
    # Jacobienne J (Diagonale)
    J_diag = []
    for b in range(4):
        # dV/dESD = V / ESD
        ratio = list_V[b] / list_ESD[b]
        J_diag.extend(ratio)
        
    J_diag = np.where(np.isfinite(J_diag), J_diag, 0.0)
    
    # C_V = J C_ESD J^T
    Cov_V = Cov_ESD * np.outer(J_diag, J_diag)
    
    return Cov_V

def joint_cost_function(params_flat, R_concat, V_obs_concat, Inv_Cov, starts, ns, fixed_params):
    """Calcule le Chi2 global matriciel"""
    model_parts = []
    
    # Reconstruction du modèle concaténé
    for b in range(4):
        idx0 = starts[b]
        idx1 = idx0 + ns[b]
        r_bin = R_concat[idx0:idx1]
        
        # 4 params par bin
        p = params_flat[4*b : 4*b+4]
        A, zeta, phi0, Rc = p
        
        Vmax, Rmax = fixed_params[b]
        
        # Modèle local
        lambda0 = 1.3 * Rmax
        omega0 = 2 * np.pi / lambda0
        Rc = max(Rc, 0.01)
        
        f = r_bin / np.sqrt(r_bin**2 + Rc**2)
        phi = 2 * np.pi * r_bin / lambda0
        z = np.clip(zeta, 1e-6, 0.9999)
        s = np.sqrt(1 - z**2)
        
        osc = (A / s) * np.exp(-z * omega0 * r_bin) * np.sin(phi + phi0)
        v_mod = Vmax * f * (1 - osc)
        
        model_parts.append(v_mod)
        
    V_model = np.concatenate(model_parts)
    res = V_obs_concat - V_model
    
    # Chi2 = R.T * C^-1 * R
    return res.T @ Inv_Cov @ res

if __name__ == "__main__":
    # --- CHEMINS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "KiDS Brouwer Weak Lensing")
    
    # Nom du fichier covariance : AJUSTEZ SI BESOIN
    # On cherche un fichier finissant par "covmatrix.txt"
    files = [f for f in os.listdir(data_dir) if f.endswith("covmatrix.txt")]
    if len(files) == 1:
        cov_file = os.path.join(data_dir, files[0])
    elif "Fig-3_Lensing-rotation-curves_Massbins_covmatrix.txt" in files:
        cov_file = os.path.join(data_dir, "Fig-3_Lensing-rotation-curves_Massbins_covmatrix.txt")
    else:
        # Fallback ou erreur
        cov_file = "CHEMIN_A_VERIFIER"
        print(f"Fichiers trouvés : {files}")

    print(f"--- JOINT ANALYSIS ---")
    print(f"Covariance : {os.path.basename(cov_file)}")
    
    # 1. Chargement Données
    list_R, list_V, list_ESD = load_all_bins(data_dir)
    if not list_R:
        raise ValueError("Aucune donnée chargée.")

    # 2. Construction Covariance Globale
    Cov_V = build_full_covariance(cov_file, list_R, list_V, list_ESD)
    if Cov_V is None:
        raise ValueError("Echec construction covariance.")
        
    # Inversion
    try:
        Inv_Cov = inv(Cov_V)
        print("Inversion matrice : OK")
    except:
        print("Matrice singulière -> Pseudo-inverse")
        Inv_Cov = pinv(Cov_V)
        
    # 3. Préparation Optimisation
    V_total = np.concatenate(list_V)
    R_total = np.concatenate(list_R)
    
    ns = [len(x) for x in list_R]
    starts = [0] + list(np.cumsum(ns))
    
    fixed_params = [(np.mean(v[-3:]), np.max(r)) for v, r in zip(list_V, list_R)]
    
    # 16 paramètres (4 bins * 4 params)
    bounds = [(0.0, 2.0), (0.01, 0.99), (0.0, 2*np.pi), (0.01, 50.0)] * 4
    
    print("Démarrage optimisation jointe (peut prendre 1-2 min)...")
    res = differential_evolution(
        joint_cost_function,
        bounds,
        args=(R_total, V_total, Inv_Cov, starts, ns, fixed_params),
        strategy='best1bin',
        maxiter=1000,
        popsize=10, # Réduit pour vitesse, augmenter pour précision
        disp=True
    )
    
    # 4. Résultats
    print(f"\nRÉSULTATS GLOBAUX")
    chi2_tot = res.fun
    dof = len(V_total) - 16
    print(f"Chi2 Total = {chi2_tot:.4f}")
    print(f"DoF = {dof}")
    print(f"Chi2 Reduced = {chi2_tot/dof:.4f}")
    
    p = res.x
    for b in range(4):
        pb = p[4*b : 4*b+4]
        print(f"BIN {b+1} | A={pb[0]:.3f} | zeta={pb[1]:.3f} | Rc={pb[3]:.2f}")

    # 5. Sauvegarde pour script 8 (null test)
    import json
    results_6 = {
        "chi2": float(chi2_tot),
        "k": 16,
        "dof": int(dof),
        "params": {f"bin{b+1}": {"A": float(p[4*b]), "zeta": float(p[4*b+1]),
                                  "phi0": float(p[4*b+2]), "Rc": float(p[4*b+3])}
                   for b in range(4)}
    }
    out_path = os.path.join(script_dir, "results_script6.json")
    with open(out_path, "w") as f:
        json.dump(results_6, f, indent=2)
    print(f"\nResultats sauvegardes : {out_path}")