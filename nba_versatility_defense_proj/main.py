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

# Elbow method to determine optimal K
"""
wcss = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
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
optimal_k = 5

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

# Visualize clusters in OBPM vs DBPM space

import seaborn as sns
plt.figure(figsize=(14, 9))

palette = sns.color_palette("viridis", n_colors=optimal_k)

for c in range(optimal_k):
    subset = df[df["Cluster"] == c]

    if c == def_cluster:
        plt.scatter(
            subset["OBPM"],
            subset["DBPM"],
            s=120,
            alpha=0.95,
            color=palette[c],
            edgecolor="black",
            linewidth=1,
            label="Defensive Specialists",
            zorder=3
        )
    else:
        plt.scatter(
            subset["OBPM"],
            subset["DBPM"],
            s=90,
            alpha=0.45, 
            color=palette[c],
            edgecolor="none",
            label=f"Cluster {c}",
            zorder=1
        )

for c in range(optimal_k):
    centroid = kmeans.cluster_centers_[c]
    color = palette[c]

    plt.scatter(
        centroid[features.index("OBPM")],
        centroid[features.index("DBPM")],
        s=350,
        color=color,
        edgecolor="black",
        linewidth=2,
        marker="X",
        zorder=4
    )

plt.axvline(0, color='black', linestyle='--', linewidth=1.2)
plt.axhline(0, color='black', linestyle='--', linewidth=1.2)

plt.xlabel("OBPM (Offensive Box Plus/Minus)", fontsize=14)
plt.ylabel("DBPM (Defensive Box Plus/Minus)", fontsize=14)
plt.title("NBA Player Clusters with Defensive Specialists Highlighted", fontsize=18)

plt.legend(title="Cluster")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


output_file = "/Users/Travis/Desktop/defensive_specialists.xlsx"
def_players.sort_values("DBPM", ascending=False).to_excel(output_file, index=False)
print(f"\nDefensive specialists exported to: {output_file}")