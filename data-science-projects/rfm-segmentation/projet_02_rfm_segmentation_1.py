"""
================================================================================
PROJET 2 : Segmentation Client RFM — Superstore Dataset
================================================================================
Auteur  : Mafez Bouzaiene
Outils  : Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Source   : kaggle.com/datasets/vivek468/superstore-dataset-final
Objectif: Segmenter les clients du Superstore selon leur comportement d'achat
          (Recency, Frequency, Monetary) et proposer des stratégies marketing
          personnalisées par segment.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Base directory pour les fichiers d'entrée/sortie
BASE_DIR = Path(__file__).resolve().parent

# ── Style global ──
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
})
PALETTE = ["#0f3460", "#e94560", "#533483", "#0ea5e9", "#f97316",
           "#10b981", "#8b5cf6", "#ef4444"]
sns.set_palette(PALETTE)

# Segment colors
SEG_COLORS = {
    "Champions": "#10b981",
    "Clients Fidèles": "#0ea5e9",
    "Potentiel Élevé": "#8b5cf6",
    "À Risque": "#f97316",
    "Perdus": "#e94560",
}

print("=" * 70)
print("  PROJET 2 : Segmentation Client RFM — Superstore")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 1 : Chargement & Préparation
# ══════════════════════════════════════════════════════════════════════
print("\n📦 ÉTAPE 1 : Chargement des données...")

df = pd.read_csv(BASE_DIR / "Superstore.csv", encoding="latin-1")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")

print(f"✅ Dataset chargé : {len(df)} transactions")
print(f"   Clients uniques : {df['Customer ID'].nunique()}")
print(f"   Période : {df['Order Date'].min().strftime('%d/%m/%Y')} → {df['Order Date'].max().strftime('%d/%m/%Y')}")

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 2 : Calcul des scores RFM
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📐 ÉTAPE 2 : Calcul des scores RFM")
print("=" * 70)

# Date de référence = jour après la dernière commande
reference_date = df["Order Date"].max() + pd.Timedelta(days=1)
print(f"\n   Date de référence : {reference_date.strftime('%d/%m/%Y')}")

# Calcul RFM par client
rfm = df.groupby("Customer ID").agg(
    Recency=("Order Date", lambda x: (reference_date - x.max()).days),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum"),
).reset_index()

# Ajouter le nom du client et le segment
customer_info = df.groupby("Customer ID").agg(
    Customer_Name=("Customer Name", "first"),
    Segment=("Segment", "first"),
    Region=("Region", "first"),
    Avg_Discount=("Discount", "mean"),
    Total_Profit=("Profit", "sum"),
    Total_Quantity=("Quantity", "sum"),
).reset_index()

rfm = rfm.merge(customer_info, on="Customer ID")

print(f"\n    RFM calculé pour {len(rfm)} clients")
print(f"\n   Statistiques RFM :")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 3 : Scoring RFM (Quintiles)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🏷️  ÉTAPE 3 : Attribution des scores RFM")
print("=" * 70)

# Score de 1 à 5 par quintile
# Pour Recency : INVERSE — plus c'est bas (récent), mieux c'est
rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Score RFM combiné
rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

print(f"\n    Scores attribués (1 = faible, 5 = élevé)")
print(f"\n   Distribution des scores totaux :")
print(rfm["RFM_Total"].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 4 : Segmentation basée sur les scores
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("👥 ÉTAPE 4 : Segmentation des clients")
print("=" * 70)

def assign_segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Clients Fidèles"
    elif r >= 3 and (f >= 2 or m >= 3):
        return "Potentiel Élevé"
    elif r <= 2 and f >= 2:
        return "À Risque"
    else:
        return "Perdus"

rfm["RFM_Segment"] = rfm.apply(assign_segment, axis=1)

segment_summary = rfm.groupby("RFM_Segment").agg(
    Nb_Clients=("Customer ID", "count"),
    Recency_Moy=("Recency", "mean"),
    Frequency_Moy=("Frequency", "mean"),
    Monetary_Moy=("Monetary", "mean"),
    Profit_Moy=("Total_Profit", "mean"),
    Discount_Moy=("Avg_Discount", "mean"),
).round(1)

segment_summary["% Clients"] = (segment_summary["Nb_Clients"] / len(rfm) * 100).round(1)
segment_summary = segment_summary.sort_values("Monetary_Moy", ascending=False)

print(f"\n   📊 Résumé par segment :")
print(segment_summary.to_string())

# ══════════════════════════════════════════════════════════════════════
#  VIZ 1 : Distribution des segments (Donut + Barres)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 ÉTAPE 5 : Visualisations")
print("=" * 70)

print("   📈 Graphique 1 : Répartition des segments...")

seg_counts = rfm["RFM_Segment"].value_counts()
seg_order = ["Champions", "Clients Fidèles", "Potentiel Élevé", "À Risque", "Perdus"]
seg_counts = seg_counts.reindex(seg_order)
colors = [SEG_COLORS[s] for s in seg_counts.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Donut
wedges, texts, autotexts = ax1.pie(
    seg_counts.values, labels=seg_counts.index,
    autopct="%1.1f%%", startangle=90, pctdistance=0.78,
    colors=colors,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
    textprops={"fontsize": 10}
)
for t in autotexts:
    t.set_fontweight("bold")
    t.set_fontsize(10)
centre = plt.Circle((0, 0), 0.55, fc="white")
ax1.add_artist(centre)
ax1.text(0, 0, f"{len(rfm)}\nClients", ha="center", va="center",
         fontsize=16, fontweight="bold", color=PALETTE[0])
ax1.set_title("Répartition des Segments", fontsize=13, fontweight="bold")

# Bar chart with client count
ax2.barh(seg_counts.index[::-1], seg_counts.values[::-1],
         color=[SEG_COLORS[s] for s in seg_counts.index[::-1]],
         edgecolor="white", height=0.6)
for i, (val, seg) in enumerate(zip(seg_counts.values[::-1], seg_counts.index[::-1])):
    pct = val / len(rfm) * 100
    ax2.text(val + 3, i, f"{val} clients ({pct:.1f}%)", va="center",
             fontsize=10, fontweight="bold")
ax2.set_title("Nombre de Clients par Segment", fontsize=13, fontweight="bold")
ax2.set_xlabel("Nombre de clients")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Segmentation RFM — Vue d'Ensemble", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_01_segments_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_01_segments_overview.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 2 : Comparaison RFM par segment (Radar-like bar chart)
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 2 : Profil RFM par segment...")

seg_profile = rfm.groupby("RFM_Segment").agg(
    R=("R_Score", "mean"),
    F=("F_Score", "mean"),
    M=("M_Score", "mean"),
).reindex(seg_order).round(2)

fig, axes = plt.subplots(1, 5, figsize=(18, 4), subplot_kw=dict(polar=True))

categories = ["Recency", "Frequency", "Monetary"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for idx, (seg, ax) in enumerate(zip(seg_order, axes)):
    values = seg_profile.loc[seg].values.tolist()
    values += values[:1]

    ax.fill(angles, values, alpha=0.25, color=SEG_COLORS[seg])
    ax.plot(angles, values, linewidth=2.5, color=SEG_COLORS[seg])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["R", "F", "M"], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="gray")
    ax.set_title(seg, fontsize=11, fontweight="bold", color=SEG_COLORS[seg], pad=15)

plt.suptitle("Profil RFM par Segment", fontsize=15, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_02_radar_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_02_radar_profiles.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 3 : Scatter plot Recency vs Monetary (coloré par segment)
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 3 : Scatter Recency vs Monetary...")

fig, ax = plt.subplots(figsize=(12, 7))
for seg in seg_order:
    mask = rfm["RFM_Segment"] == seg
    ax.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Monetary"],
               c=SEG_COLORS[seg], label=seg, alpha=0.6, s=50, edgecolors="white",
               linewidths=0.5)

ax.set_xlabel("Recency (jours depuis dernier achat)", fontweight="bold", fontsize=12)
ax.set_ylabel("Monetary (CA total $)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.set_title("Segmentation Client : Recency vs. Monetary Value",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(title="Segment", fontsize=10, title_fontsize=11,
          loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_03_scatter_recency_monetary.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_03_scatter_recency_monetary.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 4 : Scatter Frequency vs Monetary
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 4 : Scatter Frequency vs Monetary...")

fig, ax = plt.subplots(figsize=(12, 7))
for seg in seg_order:
    mask = rfm["RFM_Segment"] == seg
    ax.scatter(rfm.loc[mask, "Frequency"], rfm.loc[mask, "Monetary"],
               c=SEG_COLORS[seg], label=seg, alpha=0.6, s=50, edgecolors="white",
               linewidths=0.5)

ax.set_xlabel("Frequency (nombre de commandes)", fontweight="bold", fontsize=12)
ax.set_ylabel("Monetary (CA total $)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.set_title("Segmentation Client : Frequency vs. Monetary Value",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(title="Segment", fontsize=10, title_fontsize=11,
          loc="upper left", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_04_scatter_freq_monetary.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_04_scatter_freq_monetary.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 5 : Boxplots comparatifs R, F, M par segment
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 5 : Boxplots RFM par segment...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
metrics = [("Recency", "Recency (jours)", True),
           ("Frequency", "Fréquence (commandes)", False),
           ("Monetary", "Valeur monétaire ($)", False)]

for ax, (col, label, invert) in zip(axes, metrics):
    data = [rfm[rfm["RFM_Segment"] == seg][col].values for seg in seg_order]
    bp = ax.boxplot(data, labels=[s.split()[0] if len(s.split()) > 1 else s for s in seg_order],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    for i, (box, seg) in enumerate(zip(bp["boxes"], seg_order)):
        box.set(facecolor=SEG_COLORS[seg], alpha=0.75, edgecolor="white", linewidth=1.5)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=30)
    if col == "Monetary":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Distribution R, F, M par Segment Client",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_05_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_05_boxplots.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 6 : Heatmap — Segment × Région
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 6 : Heatmap Segment × Région...")

seg_region = pd.crosstab(rfm["RFM_Segment"], rfm["Region"])
seg_region = seg_region.reindex(seg_order)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(seg_region, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=1.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Nombre de clients", "shrink": 0.8})
ax.set_title("Distribution des Segments par Région",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_06_heatmap_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_06_heatmap_region.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 7 : KMeans Clustering pour validation
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 7 : KMeans Clustering (validation)...")

# Standardiser les données RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Méthode du coude (Elbow Method)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Elbow
ax1.plot(list(K_range), inertias, marker="o", linewidth=2.5, color=PALETTE[0],
         markersize=8)
ax1.axvline(5, color=PALETTE[1], linewidth=2, linestyle="--",
            label="K optimal = 5")
ax1.set_xlabel("Nombre de clusters (K)", fontweight="bold")
ax1.set_ylabel("Inertie", fontweight="bold")
ax1.set_title("Méthode du Coude (Elbow Method)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# KMeans avec K=5
kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm["KMeans_Cluster"] = kmeans_final.fit_predict(rfm_scaled)

cluster_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[5]]
for c in range(5):
    mask = rfm["KMeans_Cluster"] == c
    ax2.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Monetary"],
                c=cluster_colors[c], label=f"Cluster {c}", alpha=0.6, s=50,
                edgecolors="white", linewidths=0.5)

ax2.set_xlabel("Recency (jours)", fontweight="bold")
ax2.set_ylabel("Monetary ($)", fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax2.set_title("KMeans Clustering (K=5)", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Validation par Machine Learning — KMeans",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_07_kmeans.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_07_kmeans.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 8 : Revenue & Profit contribution par segment
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 8 : Contribution CA & Profit par segment...")

seg_financial = rfm.groupby("RFM_Segment").agg(
    Total_Revenue=("Monetary", "sum"),
    Total_Profit=("Total_Profit", "sum"),
    Nb_Clients=("Customer ID", "count"),
).reindex(seg_order)

seg_financial["Rev_Pct"] = (seg_financial["Total_Revenue"] / seg_financial["Total_Revenue"].sum() * 100).round(1)
seg_financial["Profit_Pct"] = (seg_financial["Total_Profit"] / seg_financial["Total_Profit"].sum() * 100).round(1)
seg_financial["Client_Pct"] = (seg_financial["Nb_Clients"] / seg_financial["Nb_Clients"].sum() * 100).round(1)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(seg_order))
w = 0.25

bars1 = ax.bar(x - w, seg_financial["Client_Pct"], w, label="% Clients",
               color=PALETTE[3], edgecolor="white", alpha=0.85)
bars2 = ax.bar(x, seg_financial["Rev_Pct"], w, label="% CA",
               color=PALETTE[0], edgecolor="white", alpha=0.85)
bars3 = ax.bar(x + w, seg_financial["Profit_Pct"], w, label="% Profit",
               color=PALETTE[5], edgecolor="white", alpha=0.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.0f}%", ha="center", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(seg_order, fontsize=10)
ax.set_ylabel("Pourcentage (%)", fontweight="bold")
ax.set_title("Contribution de Chaque Segment : Clients, CA & Profit",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(BASE_DIR / "rfm_08_contribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Sauvegardé → rfm_08_contribution.png")

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 6 : Recommandations Stratégiques par Segment
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("💡 ÉTAPE 6 : Recommandations Stratégiques par Segment")
print("=" * 70)

# Top 5 clients Champions
top_champions = rfm[rfm["RFM_Segment"] == "Champions"].nlargest(5, "Monetary")

print(f"""
   ┌──────────────────────────────────────────────────────────────────┐
   │  SEGMENT         │ STRATÉGIE RECOMMANDÉE                        │
   ├──────────────────────────────────────────────────────────────────┤
   │                  │                                               │
   │  🏆 CHAMPIONS    │ Programme de fidélité VIP                     │
   │  ({segment_summary.loc['Champions', 'Nb_Clients']} clients)     │ Accès anticipé aux nouveaux produits          │
   │                  │ Programme de parrainage avec récompenses       │
   │                  │ Communication personnalisée & exclusive        │
   │                  │                                               │
   │  💙 FIDÈLES      │ Offres de montée en gamme (upsell)            │
   │  ({segment_summary.loc['Clients Fidèles', 'Nb_Clients']} clients)│ Remises sur le prochain achat               │
   │                  │ Demander des avis/témoignages                  │
   │                  │ Invitations à des événements exclusifs         │
   │                  │                                               │
   │  🌟 POTENTIEL    │ Offres d'essai sur nouvelles catégories        │
   │  ({segment_summary.loc['Potentiel Élevé', 'Nb_Clients']} clients)│ Email de bienvenue avec recommandations     │
   │                  │ Inciter à augmenter la fréquence d'achat       │
   │                  │ Programme de points progressif                 │
   │                  │                                               │
   │  ⚠️ À RISQUE     │ Campagne de réactivation urgente               │
   │  ({segment_summary.loc['À Risque', 'Nb_Clients']} clients)      │ Enquête de satisfaction                     │
   │                  │ Offre "Vous nous manquez" avec remise           │
   │                  │ Rappel panier abandonné                        │
   │                  │                                               │
   │  ❌ PERDUS       │ Campagne de reconquête avec offre forte         │
   │  ({segment_summary.loc['Perdus', 'Nb_Clients']} clients)        │ Email de "dernière chance"                  │
   │                  │ Si pas de réponse → réduire les coûts marketing │
   │                  │ Réallouer le budget vers les Champions          │
   └──────────────────────────────────────────────────────────────────┘
""")

print(f"   🏆 Top 5 Champions (par CA total) :")
for _, row in top_champions.iterrows():
    print(f"      {row['Customer_Name']:25s} | ${row['Monetary']:>10,.2f} | {row['Frequency']} commandes | Région: {row['Region']}")

# Save RFM table to CSV
rfm_export = rfm[["Customer ID", "Customer_Name", "Region", "Segment",
                   "Recency", "Frequency", "Monetary", "Total_Profit",
                   "R_Score", "F_Score", "M_Score", "RFM_Score",
                   "RFM_Total", "RFM_Segment", "KMeans_Cluster"]]
rfm_export.to_csv(BASE_DIR / "rfm_customer_segments.csv", index=False)
print(f"\n    Table RFM exportée → rfm_customer_segments.csv ({len(rfm_export)} clients)")

print("\n" + "=" * 70)
print(" PROJET 2 TERMINÉ — 8 visualisations + 1 fichier CSV")
print("=" * 70)

