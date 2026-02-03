import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ESTHÉTIQUE ---
# On utilise un style plus "scientifique"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# --- 2. CONSTANTES & PARAMÈTRES ---
G_B21 = 4.52e-30
MPC_TO_PC = 1e6
PC_TO_KM = 3.086e13
A_GLOBAL = 0.25  # Valeur arrondie pour l'affichage

# Paramètres du MODÈLE NUL (A=0)
params_null = {
    1: {'Rc': 0.01},
    2: {'Rc': 0.01},
    3: {'Rc': 25.97},
    4: {'Rc': 16.42}
}

# Paramètres de la LOI UNIVERSELLE (A=0.25)
# Note : Pour le Bin 2, zeta = 0.989 (arrondi à 0.99)
params_univ = {
    1: {'zeta': 0.010, 'Rc': 2.84,  'phi0': 4.0},
    2: {'zeta': 0.989, 'Rc': 6.80,  'phi0': 4.0}, 
    3: {'zeta': 0.453, 'Rc': 19.14, 'phi0': 4.0},
    4: {'zeta': 0.327, 'Rc': 26.28, 'phi0': 4.0},
}

# --- 3. FONCTIONS MODÈLES ---
def V_model_Universal(r, Vmax, Rmax, Rc, zeta, phi0):
    r = np.maximum(r, 1e-6)
    lambda0 = 1.3 * Rmax
    omega0 = 2 * np.pi / lambda0
    f = r / np.sqrt(r**2 + Rc**2)
    phi = 2 * np.pi * r / lambda0
    z = np.clip(zeta, 1e-6, 0.9999)
    s = np.sqrt(1 - z**2)
    osc = (A_GLOBAL / s) * np.exp(-z * omega0 * r) * np.sin(phi + phi0)
    return Vmax * f * (1 - osc)

def V_model_Null(r, Vmax, Rc):
    r = np.maximum(r, 1e-6)
    f = r / np.sqrt(r**2 + Rc**2)
    return Vmax * f

def load_data(filepath):
    if not os.path.exists(filepath): return None, None, None
    df = pd.read_csv(filepath, sep=r'\s+', comment="#", header=None)
    R_kpc = df.iloc[:, 0].values * 1000.0
    ESD = df.iloc[:, 1].values / df.iloc[:, 4].values # ESD / bias
    ERR = df.iloc[:, 3].values / df.iloc[:, 4].values
    
    # Conversion Vitesse
    V = np.sqrt(np.maximum(4 * G_B21 * ESD * df.iloc[:, 0].values * MPC_TO_PC * PC_TO_KM**2, 0))
    # Approximation erreur Vitesse
    with np.errstate(divide='ignore', invalid='ignore'):
        V_err = V * (ERR / ESD)
        V_err = np.where(np.isfinite(V_err), V_err, V*0.1)
    return R_kpc, V, V_err

# --- 4. GÉNÉRATION DU GRAPHIQUE ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "KiDS Brouwer Weak Lensing")
    
    # Création Figure (Taille ajustée pour A4/Article)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=False)
    
    # Titre Global (Optionnel, souvent retiré pour les papiers et mis en légende LaTeX)
    # fig.suptitle(r"Evidence for Universal Coupling: $A \approx 0.25$", fontsize=16, y=0.96)
    
    axes = axes.flatten()
    bins_titles = [
        r"Bin 1: $\log(M_*) \in [8.5, 10.3]$",
        r"Bin 2: $\log(M_*) \in [10.3, 10.6]$",
        r"Bin 3: $\log(M_*) \in [10.6, 10.8]$",
        r"Bin 4: $\log(M_*) \in [10.8, 11.0]$"
    ]

    for b_idx, ax in enumerate(axes):
        bin_num = b_idx + 1
        fname = os.path.join(data_dir, f"Fig-3_Lensing-rotation-curves_Massbin-{bin_num}.txt")
        R, V, Verr = load_data(fname)
        
        if R is None: 
            ax.text(0.5, 0.5, "Data Not Found", ha='center', transform=ax.transAxes)
            continue

        Vmax = np.mean(V[-3:])
        Rmax = np.max(R)
        
        # A. Données (Points noirs)
        ax.errorbar(R, V, yerr=Verr, fmt='o', color='black', ecolor='gray', 
                    markersize=5, capsize=3, elinewidth=1, alpha=0.8,
                    label='KiDS Data' if b_idx==0 else "")

        # B. Modèles (Courbes)
        r_smooth = np.linspace(0.1, Rmax*1.15, 300)
        
        # Modèle Nul (Gris tiretés)
        vn = V_model_Null(r_smooth, Vmax, params_null[bin_num]['Rc'])
        ax.plot(r_smooth, vn, color='gray', linestyle='--', linewidth=2, 
                label=r'Null ($A=0$)' if b_idx==0 else "")
        
        # Modèle Universel (Bleu plein)
        pu = params_univ[bin_num]
        vu = V_model_Universal(r_smooth, Vmax, Rmax, pu['Rc'], pu['zeta'], pu['phi0'])
        ax.plot(r_smooth, vu, color='#1f77b4', linestyle='-', linewidth=2.5, 
                label=r'Universal ($A=0.25$)' if b_idx==0 else "")

        # C. Esthétique
        ax.set_title(bins_titles[b_idx], fontsize=13, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim(0, Rmax*1.15)
        ax.set_ylim(0, np.max(V)*1.4) # Un peu de marge en haut
        
        # D. Annotation Spéciale Bin 2 (Avec boîte blanche)
        if bin_num == 2:
            zeta_val = pu['zeta']
            # Utilisation de raw string (r"") pour éviter les soucis LaTeX
            text_label = r"Critical Damping" + "\n" + r"$\zeta \approx " + f"{zeta_val:.2f}$"
            
            ax.text(0.05, 0.82, text_label, transform=ax.transAxes, 
                    fontsize=10, color='#0b4675', fontweight='bold', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9))

        # E. Labels Axes
        if b_idx >= 2: ax.set_xlabel("Radius [kpc]", fontsize=12)
        if b_idx % 2 == 0: ax.set_ylabel(r"Circular Velocity $V_{circ}$ [km/s]", fontsize=12)

    # Légende unique en haut
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
               ncol=3, frameon=False, fontsize=12)

    plt.tight_layout()
    # Ajustement pour laisser la place à la légende
    plt.subplots_adjust(top=0.88, hspace=0.25, wspace=0.15)
    
    # Sauvegarde Haute Qualité
    save_path = os.path.join(script_dir, "Figure_Final_Publication_300dpi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image générée : {save_path}")
    plt.show()