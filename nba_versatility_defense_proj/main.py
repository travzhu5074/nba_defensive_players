import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_excel("df_021226.xlsx")
features = [
    "OBPM",
    "DBPM",
    "USG%",
    "STL%",
    "BLK%",
    "TRB%",
    "AST%"
]

# Ensure numeric values
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing rows
df_cluster = df[features].dropna().copy()

# Standardize features

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_cluster)

# Optional: attach standardized columns
scaled_columns = [col + "_z" for col in features]
df_cluster[scaled_columns] = scaled_values

# Elbow method to determine optimal K, commented out after analysis suggested K=5 was a good choice
"""
wcss = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10) #random state is the seed, 
    kmeans.fit(scaled_values)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")
plt.show()
"""

# K value selection based on elbow plot
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=1, n_init=15)
clusters = kmeans.fit_predict(scaled_values)

df_cluster["Cluster"] = clusters
df.loc[df_cluster.index, "Cluster"] = clusters

# Find best cluster for defensive specialists

cluster_summary = df.groupby("Cluster")[["OBPM", "DBPM"]].mean()

cluster_summary["def_score"] = (
    -cluster_summary["OBPM"] + cluster_summary["DBPM"]
)

def_cluster = cluster_summary["def_score"].idxmax()

print("\nCluster Summary (OBPM, DBPM):")
print(cluster_summary)

print("\nDefensive Specialist Cluster:", def_cluster)

# List of defensive specialists in the identified cluster

def_players = df[df["Cluster"] == def_cluster]

print("\nDefensive Specialists:")
print(
    def_players.sort_values("DBPM", ascending=False)[
        ["Player", "OBPM", "DBPM", "USG%", "STL%", "BLK%", "TRB%", "AST%"]
    ]
)

# ------------------------------------------------------------
# Create Additional Cluster Metrics for Interpretation
# ------------------------------------------------------------

# Expand summary beyond OBPM/DBPM
full_cluster_summary = df.groupby("Cluster")[features].mean()

full_cluster_summary["def_score"] = (
    -full_cluster_summary["OBPM"] + full_cluster_summary["DBPM"]
)

full_cluster_summary["net_bpm"] = (
    full_cluster_summary["OBPM"] + full_cluster_summary["DBPM"]
)

print("\nFull Cluster Summary:")
print(full_cluster_summary)

# Cluster roles based on statistics

def_cluster = full_cluster_summary["def_score"].idxmax()
off_cluster = full_cluster_summary["net_bpm"].idxmax()
rim_cluster = full_cluster_summary["TRB%"].idxmax()

cluster_labels = {}

# Assign elite offense first
cluster_labels[off_cluster] = "Elite Offensive Engines"

# Assign defense (if not already assigned)
if def_cluster not in cluster_labels:
    cluster_labels[def_cluster] = "Defensive Specialists"

# Assign interior / rim protection (if not already assigned)
if rim_cluster not in cluster_labels:
    cluster_labels[rim_cluster] = "Rim Protectors and Interior Anchors"

# Assign remaining cluster as standard rotation
for c in full_cluster_summary.index:
    if c not in cluster_labels:
        cluster_labels[c] = "Standard Rotation Players"

# Map labels back
df["Cluster_Label"] = df["Cluster"].map(cluster_labels)

print("\nCluster Label Mapping:")
print(cluster_labels)

print("\nCluster Counts:")
print(df["Cluster_Label"].value_counts())

# Visualize clusters in OBPM vs DBPM space

import seaborn as sns
plt.figure(figsize=(14, 9))

palette = sns.color_palette("viridis", n_colors=optimal_k)

for c in range(optimal_k):
    subset = df[df["Cluster"] == c]

    plt.scatter(
        subset["OBPM"],
        subset["DBPM"],
        s=110,
        alpha=0.85,
        color=palette[c],
        edgecolor="black",
        linewidth=0.6,
        label=cluster_labels[c],
        zorder=2
    )

# Plot centroids (convert back from standardized space)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

for c in range(optimal_k):
    centroid = centroids_original[c]

    plt.scatter(
        centroid[features.index("OBPM")],
        centroid[features.index("DBPM")],
        s=350,
        color=palette[c],
        edgecolor="black",
        linewidth=2,
        marker="X",
        zorder=4
    )

plt.axvline(0, color='black', linestyle='--', linewidth=1.2)
plt.axhline(0, color='black', linestyle='--', linewidth=1.2)

plt.xlabel("OBPM (Offensive Box Plus/Minus)", fontsize=14)
plt.ylabel("DBPM (Defensive Box Plus/Minus)", fontsize=14)
plt.title("NBA Player Clusters by Role Archetype", fontsize=18)

plt.legend(title="Cluster Role")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

base_path = "/Users/Travis/Desktop/"

for cluster_id, label in cluster_labels.items():

    cluster_df = df[df["Cluster"] == cluster_id].copy()

    # Clean file name (remove spaces and slashes)
    safe_label = label.replace(" ", "_").replace("/", "_")

    file_path = f"{base_path}{safe_label}.xlsx"

    cluster_df.sort_values("DBPM", ascending=False).to_excel(
        file_path,
        index=False
    )

    print(f"Exported {label} to: {file_path}")

#ANOVA
# ============================================================
# ASSIGNMENT 3
# ANOVA + Post-Hoc (Tukey HSD)
# ============================================================

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ------------------------------------------------------------
# PART 1: ANOVA
# ------------------------------------------------------------

# Step 1: Ensure VORP is numeric and drop missing
df["VORP"] = pd.to_numeric(df["VORP"], errors="coerce")

anova_df = df[["Cluster_Label", "VORP"]].dropna().copy()

print("\nSample Size by Cluster:")
print(anova_df["Cluster_Label"].value_counts())

# Summary statistics table
summary_table = (
    anova_df
    .groupby("Cluster_Label")["VORP"]
    .agg(["count", "mean", "std"])
)

print("\nSummary Statistics (VORP by Cluster):")
print(summary_table)

# Step 3: One-Way ANOVA
model = ols("VORP ~ C(Cluster_Label)", data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA Table:")
print(anova_table)

# ------------------------------------------------------------
# PART 2: Tukey HSD Post-Hoc Test
# ------------------------------------------------------------

tukey = pairwise_tukeyhsd(
    endog=anova_df["VORP"],
    groups=anova_df["Cluster_Label"],
    alpha=0.05
)

print("\nFull Tukey HSD Results:")
print(tukey)

# Convert Tukey results to DataFrame for filtering
# Convert Tukey results to DataFrame (preserve full precision)
tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

# Convert numeric columns properly
numeric_cols = ["meandiff", "p-adj", "lower", "upper"]
for col in numeric_cols:
    tukey_df[col] = pd.to_numeric(tukey_df[col])

# Increase display precision
pd.set_option("display.float_format", lambda x: f"{x:.10f}")

# Filter Defensive Specialists comparisons
def_tukey = tukey_df[
    (tukey_df["group1"] == "Defensive Specialists") |
    (tukey_df["group2"] == "Defensive Specialists")
]

print("\nTukey Comparisons Involving Defensive Specialists (Full Precision):")
print(def_tukey[["group1", "group2", "meandiff", "p-adj", "reject"]])

# ============================================================
# ASSIGNMENT 4
# EDA: Boxplots of VORP by Archetype
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure VORP numeric and remove missing
boxplot_df = df[["Cluster_Label", "VORP"]].dropna().copy()

# Set figure size
plt.figure(figsize=(14, 8))

# Create color palette (different color per archetype)
palette = sns.color_palette("Set2", n_colors=boxplot_df["Cluster_Label"].nunique())

# Create boxplot
sns.boxplot(
    x="Cluster_Label",
    y="VORP",
    data=boxplot_df,
    palette=palette,
    showfliers=True  # ensures outliers are marked
)

# Formatting
plt.xlabel("Archetype", fontsize=14)
plt.ylabel("Value Over Replacement Player (VORP)", fontsize=14)
plt.title("Distribution of VORP across Player Archetypes", fontsize=18)

plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
sns.despine()
plt.show()
