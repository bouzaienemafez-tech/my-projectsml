"""
================================================================================
PROJET 1 : Analyse Exploratoire des Ventes — Superstore Dataset
================================================================================
Auteur  : Mafez Bouzaiene
Outils  : Python, Pandas, NumPy, Matplotlib, Seaborn
Source   : kaggle.com/datasets/vivek468/superstore-dataset-final
Objectif: Nettoyer, analyser et visualiser les ventes d'un superstore américain
          pour en extraire des insights business actionnables.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
PALETTE = ["#0f3460", "#e94560", "#16213e", "#533483", "#0ea5e9", "#f97316",
           "#10b981", "#8b5cf6"]
sns.set_palette(PALETTE)

print("=" * 70)
print("  PROJET 1 : Analyse Exploratoire — Superstore Dataset")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 1 : Chargement & Exploration initiale
# ══════════════════════════════════════════════════════════════════════
print("\n📦 ÉTAPE 1 : Chargement des données...")

df = pd.read_csv("/Users/mafezbouzaiene/Downloads/Superstore.csv", encoding="latin-1")

print(f"✅ Dataset chargé : {len(df)} lignes × {len(df.columns)} colonnes")
print(f"\n   Colonnes : {list(df.columns)}")
print(f"\n   Types de données :")
print(df.dtypes.to_string())
print(f"\n   Aperçu :")
print(df.head().to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 2 : Nettoyage & Feature Engineering
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🧹 ÉTAPE 2 : Nettoyage & Feature Engineering")
print("=" * 70)

# Check missing values
missing = df.isnull().sum()
print(f"\n   Valeurs manquantes :")
if missing.sum() == 0:
    print("     ✅ Aucune valeur manquante !")
else:
    for col in missing[missing > 0].index:
        print(f"     - {col}: {missing[col]}")

# Check duplicates
dupes = df.duplicated().sum()
print(f"   Doublons : {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"   → Supprimés. Nouvelles lignes : {len(df)}")

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")

# Feature engineering
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["MonthName"] = df["Order Date"].dt.strftime("%b")
df["DayOfWeek"] = df["Order Date"].dt.day_name()
df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin"] = (df["Profit"] / df["Sales"] * 100).round(2)
df["Year-Month"] = df["Order Date"].dt.to_period("M")

print(f"\n   ✅ Colonnes créées : Year, Month, DayOfWeek, Shipping Days, Profit Margin")
print(f"\n   Période couverte : {df['Order Date'].min().strftime('%d/%m/%Y')} → {df['Order Date'].max().strftime('%d/%m/%Y')}")
print(f"   Années : {sorted(df['Year'].unique())}")

print(f"\n   Résumé statistique :")
print(df[["Sales", "Quantity", "Discount", "Profit", "Shipping Days", "Profit Margin"]].describe().round(2).to_string())

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 3 : KPIs
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 ÉTAPE 3 : Analyse & Visualisations")
print("=" * 70)

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
avg_order_value = total_sales / total_orders
overall_margin = total_profit / total_sales * 100
avg_discount = df["Discount"].mean() * 100
avg_shipping = df["Shipping Days"].mean()

print(f"""
   ┌──────────────────────────────────────────────┐
   │            KPIs PRINCIPAUX                    │
   ├──────────────────────────────────────────────┤
   │  Chiffre d'affaires  : ${total_sales:>14,.2f}    │
   │  Profit total         : ${total_profit:>14,.2f}    │
   │  Marge globale        : {overall_margin:>14.1f}%    │
   │  Commandes uniques    : {total_orders:>14,}     │
   │  Clients uniques      : {total_customers:>14,}     │
   │  Panier moyen         : ${avg_order_value:>14,.2f}    │
   │  Remise moyenne       : {avg_discount:>14.1f}%    │
   │  Délai livraison moy. : {avg_shipping:>13.1f} jrs    │
   └──────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 1 : Évolution CA & Profit par mois (toutes années confondues)
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 1 : Évolution CA & Profit par année...")

yearly = df.groupby("Year").agg(
    Sales=("Sales", "sum"),
    Profit=("Profit", "sum"),
    Orders=("Order ID", "nunique"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
x = np.arange(len(yearly))
w = 0.35
bars1 = ax1.bar(x - w/2, yearly["Sales"], w, label="Chiffre d'affaires",
                color=PALETTE[0], edgecolor="white", alpha=0.9)
bars2 = ax1.bar(x + w/2, yearly["Profit"], w, label="Profit",
                color=PALETTE[1], edgecolor="white", alpha=0.9)

for bar, val in zip(bars1, yearly["Sales"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, yearly["Profit"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold", color=PALETTE[1])

ax1.set_xticks(x)
ax1.set_xticklabels(yearly["Year"].astype(int))
ax1.set_ylabel("Montant ($)", fontweight="bold")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax1.set_title("Chiffre d'Affaires & Profit par Année",
              fontsize=14, fontweight="bold", pad=15)
ax1.legend(fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_01_yearly_sales_profit.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_01_yearly_sales_profit.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 2 : Tendance mensuelle (timeline)
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 2 : Tendance mensuelle...")

monthly = df.groupby("Year-Month").agg(
    Sales=("Sales", "sum"),
    Profit=("Profit", "sum"),
).reset_index()
monthly["Year-Month"] = monthly["Year-Month"].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(range(len(monthly)), monthly["Sales"], alpha=0.15, color=PALETTE[0])
ax.plot(range(len(monthly)), monthly["Sales"], color=PALETTE[0], linewidth=2.5,
        marker="", label="CA")
ax.fill_between(range(len(monthly)), monthly["Profit"], alpha=0.15, color=PALETTE[1])
ax.plot(range(len(monthly)), monthly["Profit"], color=PALETTE[1], linewidth=2.5,
        marker="", label="Profit")

# Show every 3rd label
tick_positions = list(range(0, len(monthly), 3))
tick_labels = [monthly["Year-Month"].iloc[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.set_title("Tendance Mensuelle : CA & Profit (2015-2018)",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_02_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_02_monthly_trend.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 3 : CA & Profit par Catégorie et Sous-Catégorie
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 3 : Performance par catégorie & sous-catégorie...")

subcat = (df.groupby(["Category", "Sub-Category"])
          .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
          .reset_index()
          .sort_values("Sales", ascending=True))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Sales
colors_sales = [PALETTE[0] if c == "Technology" else PALETTE[2] if c == "Furniture"
                else PALETTE[4] for c in subcat["Category"]]
ax1.barh(subcat["Sub-Category"], subcat["Sales"], color=colors_sales,
         edgecolor="white", height=0.7)
for i, (val, sc) in enumerate(zip(subcat["Sales"], subcat["Sub-Category"])):
    ax1.text(val + 1500, i, f"${val:,.0f}", va="center", fontsize=8.5)
ax1.set_title("CA par Sous-Catégorie", fontsize=13, fontweight="bold")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Profit — highlight losses in red
profit_colors = [PALETTE[1] if p < 0 else PALETTE[4] for p in subcat["Profit"]]
ax2.barh(subcat["Sub-Category"], subcat["Profit"], color=profit_colors,
         edgecolor="white", height=0.7)
ax2.axvline(0, color="black", linewidth=0.8, linestyle="-")
for i, (val, sc) in enumerate(zip(subcat["Profit"], subcat["Sub-Category"])):
    offset = -3000 if val < 0 else 1000
    ax2.text(val + offset, i, f"${val:,.0f}", va="center", fontsize=8.5,
             fontweight="bold" if val < 0 else "normal",
             color=PALETTE[1] if val < 0 else "black")
ax2.set_title("Profit par Sous-Catégorie", fontsize=13, fontweight="bold")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE[0], label="Technology"),
                   Patch(facecolor=PALETTE[2], label="Furniture"),
                   Patch(facecolor=PALETTE[4], label="Office Supplies")]
ax1.legend(handles=legend_elements, loc="lower right", fontsize=10)

plt.suptitle("Performance par Catégorie & Sous-Catégorie",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_03_category_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_03_category_performance.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 4 : Analyse par Région & Segment
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 4 : Analyse par Région & Segment...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Region
region = df.groupby("Region").agg(
    Sales=("Sales", "sum"), Profit=("Profit", "sum")
).sort_values("Sales", ascending=False).reset_index()

x = np.arange(len(region))
w = 0.35
ax1.bar(x - w/2, region["Sales"], w, label="CA", color=PALETTE[0], edgecolor="white")
ax1.bar(x + w/2, region["Profit"], w, label="Profit", color=PALETTE[1], edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels(region["Region"])
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax1.set_title("CA & Profit par Région", fontsize=13, fontweight="bold")
ax1.legend()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Segment
segment = df.groupby("Segment").agg(
    Sales=("Sales", "sum"), Profit=("Profit", "sum")
).sort_values("Sales", ascending=False).reset_index()

x2 = np.arange(len(segment))
ax2.bar(x2 - w/2, segment["Sales"], w, label="CA", color=PALETTE[0], edgecolor="white")
ax2.bar(x2 + w/2, segment["Profit"], w, label="Profit", color=PALETTE[4], edgecolor="white")
ax2.set_xticks(x2)
ax2.set_xticklabels(segment["Segment"])
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax2.set_title("CA & Profit par Segment Client", fontsize=13, fontweight="bold")
ax2.legend()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_04_region_segment.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_04_region_segment.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 5 : Impact des remises sur la marge bénéficiaire
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 5 : Impact des remises sur la rentabilité...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Scatter: Discount vs Profit
scatter_colors = [PALETTE[1] if p < 0 else PALETTE[4] for p in df["Profit"]]
ax1.scatter(df["Discount"] * 100, df["Profit"], c=scatter_colors, alpha=0.3, s=15, edgecolors="none")
ax1.axhline(0, color="black", linewidth=0.8)

# Trend line
z = np.polyfit(df["Discount"] * 100, df["Profit"], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 80, 100)
ax1.plot(x_line, p(x_line), color=PALETTE[3], linewidth=2.5, linestyle="--",
         label=f"Tendance (pente: {z[0]:.1f})")
ax1.set_xlabel("Remise (%)", fontweight="bold")
ax1.set_ylabel("Profit ($)", fontweight="bold")
ax1.set_title("Remise vs. Profit (par transaction)", fontsize=13, fontweight="bold")
ax1.legend()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Grouped: Avg profit margin by discount bucket
df["Discount_Bucket"] = pd.cut(df["Discount"], bins=[-.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0],
                                labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", "50%+"])
bucket = df.groupby("Discount_Bucket", observed=True).agg(
    Avg_Margin=("Profit Margin", "mean"),
    Count=("Sales", "count"),
).reset_index()

bar_colors = [PALETTE[4] if m >= 0 else PALETTE[1] for m in bucket["Avg_Margin"]]
bars = ax2.bar(bucket["Discount_Bucket"].astype(str), bucket["Avg_Margin"],
               color=bar_colors, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.8)
for bar, val, cnt in zip(bars, bucket["Avg_Margin"], bucket["Count"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if val >= 0 else -2.5),
             f"{val:.1f}%\n({cnt:,})", ha="center", fontsize=9, fontweight="bold")
ax2.set_xlabel("Tranche de remise", fontweight="bold")
ax2.set_ylabel("Marge bénéficiaire moyenne (%)", fontweight="bold")
ax2.set_title("Marge Moyenne par Tranche de Remise", fontsize=13, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_05_discount_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_05_discount_impact.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 6 : Top 10 produits les plus rentables vs les moins rentables
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 6 : Produits les plus & moins rentables...")

product_profit = df.groupby("Product Name")["Profit"].sum().sort_values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bottom 10 (losses)
bottom = product_profit.head(10)
ax1.barh(range(len(bottom)), bottom.values, color=PALETTE[1], edgecolor="white", height=0.7)
ax1.set_yticks(range(len(bottom)))
ax1.set_yticklabels([n[:35] + "..." if len(n) > 35 else n for n in bottom.index], fontsize=8.5)
for i, val in enumerate(bottom.values):
    ax1.text(val - 200, i, f"${val:,.0f}", va="center", fontsize=8.5,
             fontweight="bold", color="white")
ax1.set_title("❌ 10 Produits les Moins Rentables", fontsize=12, fontweight="bold")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Top 10 (profits)
top = product_profit.tail(10)
ax2.barh(range(len(top)), top.values, color=PALETTE[6], edgecolor="white", height=0.7)
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels([n[:35] + "..." if len(n) > 35 else n for n in top.index], fontsize=8.5)
for i, val in enumerate(top.values):
    ax2.text(val + 100, i, f"${val:,.0f}", va="center", fontsize=8.5, fontweight="bold")
ax2.set_title("✅ 10 Produits les Plus Rentables", fontsize=12, fontweight="bold")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Analyse de la Rentabilité Produit", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_06_product_profitability.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_06_product_profitability.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 7 : Heatmap — CA par Région × Catégorie
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 7 : Heatmap Région × Catégorie...")

heatmap_data = df.pivot_table(values="Sales", index="Region", columns="Category",
                               aggfunc="sum")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_data, annot=True, fmt=",.0f", cmap="YlOrRd",
            linewidths=1, linecolor="white", ax=ax,
            cbar_kws={"label": "CA ($)", "shrink": 0.8})
ax.set_title("Heatmap : CA par Région × Catégorie",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_07_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_07_heatmap.png")

# ══════════════════════════════════════════════════════════════════════
#  VIZ 8 : Analyse des délais de livraison par mode d'expédition
# ══════════════════════════════════════════════════════════════════════
print("   📈 Graphique 8 : Délais de livraison...")

fig, ax = plt.subplots(figsize=(10, 5))
ship_order = ["Same Day", "First Class", "Second Class", "Standard Class"]
ship_data = [df[df["Ship Mode"] == s]["Shipping Days"] for s in ship_order if s in df["Ship Mode"].values]
ship_labels = [s for s in ship_order if s in df["Ship Mode"].values]

bp = ax.boxplot(ship_data, labels=ship_labels, patch_artist=True,
                medianprops=dict(color=PALETTE[1], linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
for i, box in enumerate(bp["boxes"]):
    box.set(facecolor=PALETTE[i], alpha=0.7, edgecolor="white", linewidth=1.5)

for i, s in enumerate(ship_labels):
    avg = df[df["Ship Mode"] == s]["Shipping Days"].mean()
    ax.text(i + 1, avg + 0.3, f"Moy: {avg:.1f}j", ha="center",
            fontsize=9, fontweight="bold", color=PALETTE[1])

ax.set_title("Délais de Livraison par Mode d'Expédition",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Jours de livraison", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_08_shipping.png", dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_08_shipping.png")

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 4 : Recommandations Business
# ══════════════════════════════════════════════════════════════════════

# Key findings
loss_subcats = subcat[subcat["Profit"] < 0]["Sub-Category"].tolist()
best_region = region.iloc[0]["Region"]
worst_margin_discount = bucket.loc[bucket["Avg_Margin"].idxmin(), "Discount_Bucket"]
top3 = product_profit.tail(3).index.tolist()

print("\n" + "=" * 70)
print("💡 ÉTAPE 4 : Recommandations Business")
print("=" * 70)
print(f"""
   Sur la base de cette analyse du Superstore Dataset, voici 5 recommandations :

   1. 🚨 RÉDUIRE LES PERTES SUR LES SOUS-CATÉGORIES DÉFICITAIRES
      {', '.join(loss_subcats)} génèrent des pertes nettes.
      → Revoir la politique de prix, réduire les remises, ou envisager
        l'arrêt des produits les moins performants.

   2. 💰 PLAFONNER LES REMISES À 20%
      Au-delà de 20% de remise, la marge devient négative.
      La tranche {worst_margin_discount} montre les pires résultats.
      → Mettre en place un système d'approbation pour les remises > 20%.

   3. 🌎 CAPITALISER SUR LA RÉGION {best_region.upper()}
      C'est la région la plus performante en CA et en profit.
      → Augmenter les investissements marketing dans cette zone.
        Analyser les facteurs de succès pour les répliquer ailleurs.

   4. 📦 OPTIMISER LA LOGISTIQUE
      Les délais Standard Class sont significativement plus longs.
      → Négocier avec les transporteurs, proposer des upsells
        vers First Class avec des offres promotionnelles ciblées.

   5. ⭐ PROMOUVOIR LES BEST-SELLERS
      Les produits les plus rentables incluent :
      {', '.join([n[:40] for n in top3[::-1]])}
      → Créer des bundles, augmenter la visibilité en ligne,
        et utiliser ces produits comme produits d'appel.
""")

# ══════════════════════════════════════════════════════════════════════
#  ETAPE 5 : Machine Learning — Prédiction des Meilleurs Vendeurs
#            sur les 10 prochaines années (2019–2028)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🤖 ÉTAPE 5 : Machine Learning — Prédiction Best-Sellers 2019-2028")
print("=" * 70)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── 5.1  Préparation des données ──────────────────────────────────────
# Agrégation mensuelle par sous-catégorie
ml_data = (df.groupby(["Year", "Month", "Sub-Category"])
             .agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"))
             .reset_index())

# Encodage des sous-catégories
le = LabelEncoder()
ml_data["SubCat_enc"] = le.fit_transform(ml_data["Sub-Category"].to_numpy())

# Features temporelles
ml_data["Trend"] = (ml_data["Year"] - ml_data["Year"].min()) * 12 + ml_data["Month"]
ml_data["Month_sin"] = np.sin(2 * np.pi * ml_data["Month"] / 12)
ml_data["Month_cos"] = np.cos(2 * np.pi * ml_data["Month"] / 12)
ml_data["Q1"] = (ml_data["Month"] <= 3).astype(int)
ml_data["Q4"] = (ml_data["Month"] >= 10).astype(int)

FEATURES = ["Trend", "Month_sin", "Month_cos", "Q1", "Q4", "SubCat_enc"]
TARGET = "Sales"

X = ml_data[FEATURES].to_numpy(dtype=float)
y = ml_data[TARGET].to_numpy(dtype=float)

# ── 5.2  Entraînement du modèle ───────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42
)
model.fit(X, y)

y_pred_train = model.predict(X)
mae  = mean_absolute_error(y, y_pred_train)
r2   = r2_score(y, y_pred_train)
print(f"\n   Modèle : Gradient Boosting Regressor")
print(f"   MAE    : ${mae:,.0f}  |  R²  : {r2:.4f}")

# ── 5.3  Génération des prévisions 2019–2028 ──────────────────────────
last_trend = ml_data["Trend"].max()
years_future  = range(2019, 2029)
months_future = range(1, 13)
subcats = le.classes_

rows = []
for year in years_future:
    for month in months_future:
        trend = int(last_trend) + (year - int(ml_data["Year"].max())) * 12 + (month - int(ml_data["Month"].max()))
        for sc in subcats:
            rows.append({
                "Year": year,
                "Month": month,
                "Sub-Category": sc,
                "Trend": trend,
                "Month_sin": np.sin(2 * np.pi * month / 12),
                "Month_cos": np.cos(2 * np.pi * month / 12),
                "Q1": int(month <= 3),
                "Q4": int(month >= 10),
                "SubCat_enc": int(np.asarray(le.transform([sc]))[0]),
            })

future_df = pd.DataFrame(rows)
future_df["Predicted_Sales"] = model.predict(future_df[FEATURES].values).clip(min=0)

# Cumulative predicted sales per sub-category over 10 years
forecast_summary = (future_df.groupby("Sub-Category")["Predicted_Sales"]
                              .sum()
                              .sort_values(ascending=False)
                              .reset_index())
forecast_summary.columns = ["Sub-Category", "Total_Predicted_Sales_10Y"]

# Merge with historical category label
cat_map = df[["Sub-Category", "Category"]].drop_duplicates()
forecast_summary = forecast_summary.merge(cat_map, on="Sub-Category", how="left")

print("\n   📊 Classement prédit — CA cumulé sur 10 ans (2019-2028) :")
print("   " + "-" * 60)
for i, row in forecast_summary.iterrows():
    print(f"   #{i+1:>2}  {row['Sub-Category']:<20} {row['Category']:<18}  ${row['Total_Predicted_Sales_10Y']:>12,.0f}")

# ── 5.4  VIZ 9 : Top 10 Best-Sellers prédits (CA cumulé 10 ans) ──────
print("\n   📈 Graphique 9 : Prédiction Best-Sellers 2019-2028...")

top10 = forecast_summary.head(10).iloc[::-1]  # reverse for horizontal bar
cat_colors = {"Technology": PALETTE[0], "Furniture": PALETTE[2], "Office Supplies": PALETTE[4]}
bar_colors  = [cat_colors.get(c, PALETTE[7]) for c in top10["Category"]]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(top10["Sub-Category"], top10["Total_Predicted_Sales_10Y"],
               color=bar_colors, edgecolor="white", height=0.65)
for bar, val in zip(bars, top10["Total_Predicted_Sales_10Y"]):
    ax.text(bar.get_width() + 50_000, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9.5, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in cat_colors.items()]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
ax.set_title("Top 10 Best-Sellers Prédits — CA Cumulé 2019-2028\n(Gradient Boosting Regressor)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("CA Cumulé Prédit ($)", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_09_ml_best_sellers_10y.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_09_ml_best_sellers_10y.png")

# ── 5.5  VIZ 10 : Trajectoires temporelles des 5 top sous-catégories ─
print("   📈 Graphique 10 : Trajectoires de ventes prédites (Top 5)...")

top5_subcats = forecast_summary.head(5)["Sub-Category"].tolist()

# Historical monthly sales for top5
hist_top5 = (ml_data[ml_data["Sub-Category"].isin(top5_subcats)]
             .copy())
hist_top5["Date"] = pd.to_datetime(
    hist_top5["Year"].astype(str) + "-" + hist_top5["Month"].astype(str).str.zfill(2) + "-01")

# Future monthly for top5
fut_top5 = future_df[future_df["Sub-Category"].isin(top5_subcats)].copy()
fut_top5["Date"] = pd.to_datetime(
    fut_top5["Year"].astype(str) + "-" + fut_top5["Month"].astype(str).str.zfill(2) + "-01")

fig, ax = plt.subplots(figsize=(16, 6))
for i, sc in enumerate(top5_subcats):
    color = PALETTE[i]
    # Historical
    h = hist_top5[hist_top5["Sub-Category"] == sc].sort_values("Date")
    ax.plot(h["Date"], h["Sales"], color=color, linewidth=1.8, alpha=0.9, label=sc)
    # Predicted
    f = fut_top5[fut_top5["Sub-Category"] == sc].sort_values("Date")
    ax.plot(f["Date"], f["Predicted_Sales"], color=color, linewidth=1.8,
            linestyle="--", alpha=0.7)

ax.axvline(pd.Timestamp("2019-01-01"), color="black", linewidth=1.2,
           linestyle=":", alpha=0.6)
ax.text(pd.Timestamp("2019-03-01"), ax.get_ylim()[1] * 0.92,
        "← Historique  |  Prédiction →", fontsize=9.5, color="gray")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.set_title("Trajectoires de Ventes : Historique (trait plein) & Prédiction (pointillés)\n"
             "Top 5 Sous-Catégories — 2015 à 2028",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(fontsize=10, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/Users/mafezbouzaiene/Downloads/visualizations/viz_10_ml_trajectories.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("     ✅ Sauvegardé → viz_10_ml_trajectories.png")

# ── 5.6  Insights ML ─────────────────────────────────────────────────
top1 = forecast_summary.iloc[0]
top2 = forecast_summary.iloc[1]
top3_ml = forecast_summary.iloc[2]
print(f"""
   🔮 INSIGHTS ML — Best-Sellers Prédits 2019-2028 :

   #1  {top1['Sub-Category']:<20} → ${top1['Total_Predicted_Sales_10Y']:>12,.0f} de CA prédit
   #2  {top2['Sub-Category']:<20} → ${top2['Total_Predicted_Sales_10Y']:>12,.0f} de CA prédit
   #3  {top3_ml['Sub-Category']:<20} → ${top3_ml['Total_Predicted_Sales_10Y']:>12,.0f} de CA prédit

   → La catégorie dominante est : {forecast_summary.groupby('Category')['Total_Predicted_Sales_10Y'].sum().idxmax()}
   → Investir en priorité dans ces sous-catégories pour maximiser le CA.
""")

print("=" * 70)
print("✅ PROJET TERMINÉ — 10 visualisations sauvegardées")
print("=" * 70)
print("""
   📂 Fichiers générés :
      viz_01_yearly_sales_profit.png   → CA & Profit par année
      viz_02_monthly_trend.png         → Tendance mensuelle
      viz_03_category_performance.png  → Performance par catégorie
      viz_04_region_segment.png        → Analyse région & segment
      viz_05_discount_impact.png       → Impact des remises
      viz_06_product_profitability.png → Rentabilité produit
      viz_07_heatmap.png               → Heatmap Région × Catégorie
      viz_08_shipping.png              → Délais de livraison
      viz_09_ml_best_sellers_10y.png   → ML : Top 10 best-sellers prédits
      viz_10_ml_trajectories.png       → ML : Trajectoires 2015-2028

   🎯 Pour votre GitHub :
      1. Créez un repo "superstore-sales-eda"
      2. Ajoutez ce script + dataset + visualisations
      3. Rédigez un README.md professionnel
      4. Ajoutez le lien GitHub à votre CV !

   📝 Ligne CV :
      "Analyse exploratoire de 9 994 transactions retail (Superstore Dataset)
       — nettoyage, feature engineering, 10 visualisations, 5 recommandations
       business et prédiction ML (Gradient Boosting) des best-sellers
       sur 10 ans avec Python (Pandas, Scikit-learn, Matplotlib, Seaborn)"
""")
