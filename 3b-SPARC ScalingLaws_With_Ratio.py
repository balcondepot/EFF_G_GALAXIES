#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def spearman_corr(df):
    # Spearman = corrélation sur les rangs (robuste aux non-linéarités)
    return df.rank().corr(method="pearson")

def top_pairs(corr, k=20, min_abs=0.3):
    # récupère les plus fortes corrélations hors diagonale
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    pairs = (
        c.stack()
         .reset_index()
         .rename(columns={"level_0":"x", "level_1":"y", 0:"corr"})
    )
    # enlever doublons x-y / y-x
    pairs["key"] = pairs.apply(lambda r: "||".join(sorted([r["x"], r["y"]])), axis=1)
    pairs = pairs.drop_duplicates("key").drop(columns="key")
    pairs = pairs[np.abs(pairs["corr"]) >= min_abs].sort_values("corr", key=np.abs, ascending=False)
    return pairs.head(k)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_csv = os.path.join(script_dir, "2-SPARC Params Parsing Results Optimized", "SPARC_fit_summary_optimized_5p_occam.csv")
    out_dir = os.path.join(script_dir, "3b-SPARC Ratio Correlations Results")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)

    # filtre standard
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    # colonnes attendues
    cols = ["Rmax_kpc", "lambda0_kpc", "Rc_kpc", "A", "zeta", "V_max_fixed", "RMSE_km_s", "R_squared"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Rmax_kpc", "lambda0_kpc", "Rc_kpc", "A", "zeta", "V_max_fixed"])
    df = df[(df["Rmax_kpc"] > 0) & (df["lambda0_kpc"] > 0) & (df["Rc_kpc"] > 0) & (df["V_max_fixed"] > 0)]
    df = df[(df["zeta"] > 0) & (df["zeta"] < 1)]  # zeta dans (0,1)

    # -------- ratios / invariants --------
    df["q_lambda"] = df["lambda0_kpc"] / df["Rmax_kpc"]                # lambda0/Rmax
    df["q_Rc"]     = df["Rc_kpc"] / df["Rmax_kpc"]                     # Rc/Rmax
    df["Nosc"]     = df["Rmax_kpc"] / df["lambda0_kpc"]                # Rmax/lambda0
    df["L_damp"]   = df["lambda0_kpc"] / (2*np.pi*df["zeta"])          # lambda0/(2pi zeta)
    df["q_Ldamp"]  = df["L_damp"] / df["Rmax_kpc"]                     # Ldamp/Rmax
    df["lambda_over_zeta"] = df["lambda0_kpc"] / df["zeta"]            # (proche de L_damp * 2pi)

    # variables à corréler
    feats = [
        "Rmax_kpc", "V_max_fixed", "A", "zeta", "lambda0_kpc", "Rc_kpc",
        "q_lambda", "q_Rc", "Nosc", "L_damp", "q_Ldamp", "lambda_over_zeta"
    ]
    feats = [f for f in feats if f in df.columns]

    X = df[feats].copy()

    # corrélations
    pear = X.corr(method="pearson")
    spear = spearman_corr(X)

    pear.to_csv(os.path.join(out_dir, "corr_pearson.csv"))
    spear.to_csv(os.path.join(out_dir, "corr_spearman.csv"))

    top_pear = top_pairs(pear, k=30, min_abs=0.35)
    top_spear = top_pairs(spear, k=30, min_abs=0.35)
    top_pear.to_csv(os.path.join(out_dir, "top_pairs_pearson.csv"), index=False)
    top_spear.to_csv(os.path.join(out_dir, "top_pairs_spearman.csv"), index=False)

    print("OK. Fichiers écrits dans:", out_dir)
    print("\nTop Pearson:")
    print(top_pear.head(10).to_string(index=False))
    print("\nTop Spearman:")
    print(top_spear.head(10).to_string(index=False))

    # -------- plots ciblés (les plus pertinents) --------
    plots = [
        ("Rmax_kpc", "q_lambda", "Rmax (kpc)", "lambda0/Rmax", "Invariant q_lambda vs Rmax", "q_lambda_vs_Rmax.png"),
        ("V_max_fixed", "q_lambda", "Vmax (km/s)", "lambda0/Rmax", "q_lambda vs Vmax", "q_lambda_vs_Vmax.png"),
        ("Rmax_kpc", "q_Ldamp", "Rmax (kpc)", "Ldamp/Rmax", "q_Ldamp vs Rmax", "q_Ldamp_vs_Rmax.png"),
        ("q_lambda", "zeta", "lambda0/Rmax", "zeta", "zeta vs q_lambda", "zeta_vs_q_lambda.png"),
        ("q_lambda", "A", "lambda0/Rmax", "A", "A vs q_lambda", "A_vs_q_lambda.png"),
        ("Nosc", "A", "Rmax/lambda0", "A", "A vs Nosc", "A_vs_Nosc.png"),
    ]

    for x, y, xl, yl, title, fn in plots:
        if x not in df.columns or y not in df.columns:
            continue
        plt.figure(figsize=(6.6, 4.8))
        plt.scatter(df[x], df[y], s=18)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(title)
        plt.grid(alpha=0.25)
        # axes log seulement si positif partout et si ça a du sens
        if (df[x] > 0).all() and x in ["Rmax_kpc", "V_max_fixed", "Nosc"]:
            plt.xscale("log")
        # y en log rarement utile pour ratios bornés, donc on évite
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fn), dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
