import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def parse_table2_properties(filepath):
    """
    Lit Table2 pour extraire les propriétés baryoniques au dernier point mesuré.
    Retourne un dict: {galaxy_name: {'Vbary': float, 'Rmax': float}}
    """
    properties = {}
    
    if not os.path.exists(filepath):
        print(f"Fichier introuvable : {filepath}")
        return {}

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1
            break
            
    # On doit accumuler les points pour chaque galaxie pour trouver le dernier (R_max)
    current_gal = ""
    gal_data = []

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 8: continue
        
        name = parts[0]
        try:
            R = float(parts[2])
            Vgas = float(parts[5])
            Vdisk = float(parts[6])
            Vbul = float(parts[7])
            
            # Calcul de Vbary pour ce point
            # Vbary = sqrt(|Vgas|Vgas + |Vdisk|Vdisk + |Vbul|Vbul) (somme des carrés signés)
            Vbary_sq = np.abs(Vgas)*Vgas + np.abs(Vdisk)*Vdisk + np.abs(Vbul)*Vbul
            Vbary = np.sqrt(Vbary_sq) if Vbary_sq > 0 else 0.0
            
            if name != current_gal:
                # Sauvegarder la précédente
                if current_gal != "" and gal_data:
                    # On prend le point avec le R le plus grand
                    last_point = max(gal_data, key=lambda x: x['R'])
                    properties[current_gal] = last_point
                
                current_gal = name
                gal_data = []
            
            gal_data.append({'R': R, 'Vbary': Vbary})
            
        except ValueError:
            continue
            
    # La dernière
    if current_gal != "" and gal_data:
        last_point = max(gal_data, key=lambda x: x['R'])
        properties[current_gal] = last_point

    return properties

def main():
    script_dir = os.getcwd()
    table2_path = os.path.join(script_dir, "Table2 SPARC Masse Models.mrt")
    summary_path = os.path.join(script_dir, "SPARC_fit_summary.csv")
    
    # 1. Charger les résultats du fit (V_max)
    if not os.path.exists(summary_path):
        print("Erreur: SPARC_fit_summary.csv manquant.")
        return
        
    df_fit = pd.read_csv(summary_path)
    
    # Filtrer les mauvais fits si la colonne existe
    if "category" in df_fit.columns:
        df_fit = df_fit[df_fit["category"].isin(["OK", "WARNING"])]
        print(f"Analyse sur {len(df_fit)} galaxies valides (OK+WARNING).")
    else:
        print(f"Analyse sur {len(df_fit)} galaxies (pas de filtre catégorie trouvé).")

    # 2. Charger les propriétés physiques (Vbary, Rmax)
    props = parse_table2_properties(table2_path)
    
    # 3. Fusionner
    df_fit["Vbary_obs"] = df_fit["name"].map(lambda x: props.get(x, {}).get("Vbary", np.nan))
    df_fit["Rmax_obs"] = df_fit["name"].map(lambda x: props.get(x, {}).get("R", np.nan))
    
    df_clean = df_fit.dropna(subset=["Vbary_obs", "V_max"])
    
    # 4. Analyse 1 : Relation V_max (Fit) vs V_bary (Propriété Baryonique)
    # C'est la relation de Tully-Fisher Baryonique "reconstruite"
    
    x = df_clean["Vbary_obs"]
    y = df_clean["V_max"]
    
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: V_max vs V_bary
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.6, c='blue', edgecolors='k')
    plt.plot(x, slope*x + intercept, 'r--', label=f"y = {slope:.2f}x + {intercept:.1f}")
    plt.xlabel("V_baryonique (déduite de la masse) [km/s]")
    plt.ylabel("V_max (Modèle Effectif) [km/s]")
    plt.title(f"Déduction de V_max par la Masse\nCorrélation R = {r_value:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Analyse 2 : Relation V_max (Fit) vs R_max (Taille)
    x2 = df_clean["Rmax_obs"]
    # Log-Log pour la loi de puissance
    # On évite les logs de 0
    mask = (x2 > 0) & (y > 0)
    x2_log = np.log10(x2[mask])
    y_log = np.log10(y[mask])
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2_log, y_log)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x2[mask], y[mask], alpha=0.6, c='green', edgecolors='k')
    # Fit en log space -> y = 10^b * x^a
    x_line = np.linspace(x2.min(), x2.max(), 100)
    y_line = (10**intercept2) * (x_line**slope2)
    
    plt.plot(x_line, y_line, 'r--', label=f"Vmax ~ R^{slope2:.2f}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("R_max (Taille) [kpc]")
    plt.ylabel("V_max (Modèle Effectif) [km/s]")
    plt.title(f"Déduction de V_max par la Taille\nCorrélation R = {r_value2:.3f}")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig("Vmax_Deduction_Analysis.png")
    print("Graphique généré : Vmax_Deduction_Analysis.png")
    print("-" * 50)
    print(f"Corrélation V_max vs V_bary (Masse) : {r_value:.4f} (R² = {r_value**2:.4f})")
    print(f"Corrélation V_max vs R_max (Taille) : {r_value2:.4f} (R² = {r_value2**2:.4f})")

if __name__ == "__main__":
    main()