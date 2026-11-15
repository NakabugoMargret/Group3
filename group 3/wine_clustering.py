"""
Wine Dataset Clustering: K-means and Hierarchical Clustering

This program analyses the UCI Wine dataset without using the class labels
during preprocessing, feature transformation, model fitting, or model tuning.

The class labels are kept separately and are used only for optional external
validation after clustering has been completed.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway, kruskal

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
# A compact PDF report is created from the key charts and findings.
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. General settings and dataset loading

BASE_FOLDER = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE_FOLDER / "wine.data"
OUTPUT_FOLDER = BASE_FOLDER / "wine_results"
OUTPUT_FOLDER.mkdir(exist_ok=True)

RANDOM_STATE = 42

column_names = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "The file 'wine.data' was not found. Keep wine.data in the same "
        "folder as this Python file or notebook."
    )

wine_data = pd.read_csv(DATA_FILE, header=None, names=column_names)

# The first column contains class labels. It is separated before modelling.
true_labels = wine_data["class"].copy()
features = wine_data.drop(columns=["class"]).copy()

print("DATASET OVERVIEW")
print("Number of rows:", features.shape[0])
print("Number of chemical attributes:", features.shape[1])
print("\nFirst 10 rows of the chemical attributes:")
print(features.head(10).to_string(index=False))

print("\nData types:")
print(features.dtypes)

print("\nMissing values in each feature:")
print(features.isnull().sum())


# 2. Part A: Missing values and standardisation

# Median imputation is included so the program can also handle missing
# values if they are present in a modified copy of the dataset.
imputer = SimpleImputer(strategy="median")
features_imputed = pd.DataFrame(
    imputer.fit_transform(features),
    columns=features.columns,
    index=features.index,
)

scaler = StandardScaler()
scaled_features = pd.DataFrame(
    scaler.fit_transform(features_imputed),
    columns=features.columns,
    index=features.index,
)

scaled_file = OUTPUT_FOLDER / "wine_scaled_features.csv"
scaled_features.to_csv(scaled_file, index=False)

print("\nMissing values after imputation:", int(features_imputed.isnull().sum().sum()))
print("Scaled dataset saved to:", scaled_file)


# 3. Part B: First exploratory data analysis

selected_features = [
    "alcohol",
    "malic_acid",
    "ash",
    "flavanoids",
    "color_intensity",
    "proline",
]

print("\nDESCRIPTIVE STATISTICS")
print(features_imputed[selected_features].describe().round(3))

# Histograms for six selected chemical attributes.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feature_name in zip(axes.flat, selected_features):
    sns.histplot(
        features_imputed[feature_name],
        kde=True,
        bins=18,
        color="steelblue",
        ax=ax,
    )
    ax.set_title(feature_name.replace("_", " ").title())
    ax.set_xlabel(feature_name.replace("_", " ").title())
    ax.set_ylabel("Number of wine samples")

fig.suptitle("Distributions of Selected Wine Attributes", fontsize=15)
fig.tight_layout()
histogram_file = OUTPUT_FOLDER / "feature_histograms.png"
fig.savefig(histogram_file, dpi=300, bbox_inches="tight")
plt.show()

# PCA is fitted to standardised chemical attributes, not to the class labels.
pca_all = PCA()
pca_all.fit(scaled_features)

explained_variance = pca_all.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

fig, ax = plt.subplots(figsize=(10, 6))
component_numbers = np.arange(1, len(explained_variance) + 1)
ax.bar(component_numbers, explained_variance, alpha=0.75, label="Individual variance")
ax.plot(
    component_numbers,
    cumulative_variance,
    marker="o",
    color="darkred",
    label="Cumulative variance",
)
ax.axhline(0.95, linestyle="--", color="grey", label="95% variance")
ax.set_xlabel("Principal component")
ax.set_ylabel("Proportion of variance explained")
ax.set_title("PCA Scree Plot")
ax.set_xticks(component_numbers)
ax.legend()
fig.tight_layout()
scree_file = OUTPUT_FOLDER / "pca_scree_plot.png"
fig.savefig(scree_file, dpi=300, bbox_inches="tight")
plt.show()

number_of_components = int(np.argmax(cumulative_variance >= 0.95) + 1)
print("\nNumber of PCA components needed to retain at least 95% variance:",
      number_of_components)
print("Variance retained:", round(cumulative_variance[number_of_components - 1], 4))


# 4. Part C: Feature engineering and dimensionality reduction

# PCA keeps most of the information while reducing the number of dimensions.
# The 95% threshold is selected before clustering and does not use class labels.
pca_reduced = PCA(n_components=number_of_components, random_state=RANDOM_STATE)
pca_features = pca_reduced.fit_transform(scaled_features)

pca_columns = [f"PC{i + 1}" for i in range(number_of_components)]
pca_feature_data = pd.DataFrame(
    pca_features,
    columns=pca_columns,
    index=features.index,
)

print("\nOriginal number of dimensions:", scaled_features.shape[1])
print("Number of dimensions after PCA:", pca_feature_data.shape[1])

# 5. Part D: K-means clustering and internal validation

k_values = range(2, 9)
kmeans_results = []
kmeans_models = {}

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    cluster_assignments = model.fit_predict(pca_feature_data)
    kmeans_models[k] = model

    kmeans_results.append({
        "k": k,
        "silhouette": silhouette_score(pca_feature_data, cluster_assignments),
        "davies_bouldin": davies_bouldin_score(pca_feature_data, cluster_assignments),
        "calinski_harabasz": calinski_harabasz_score(
            pca_feature_data, cluster_assignments
        ),
        "inertia": model.inertia_,
    })

kmeans_results = pd.DataFrame(kmeans_results)
print("\nK-MEANS VALIDATION RESULTS")
print(kmeans_results.round(4).to_string(index=False))

# The highest silhouette score is used as the main selection rule.
# Davies-Bouldin and Calinski-Harabasz are considered as supporting measures.
best_k = int(
    kmeans_results.loc[kmeans_results["silhouette"].idxmax(), "k"]
)
best_kmeans = kmeans_models[best_k]
kmeans_clusters = best_kmeans.labels_

print("\nSelected number of clusters:", best_k)
print("Selection reason: it has the highest silhouette score among k = 2 to 8.")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
metric_columns = [
    ("silhouette", "Silhouette Score"),
    ("davies_bouldin", "Davies-Bouldin Index"),
    ("calinski_harabasz", "Calinski-Harabasz Index"),
]
for ax, (column, title) in zip(axes, metric_columns):
    ax.plot(kmeans_results["k"], kmeans_results[column], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_xticks(list(k_values))
    ax.grid(alpha=0.25)
fig.suptitle("K-means Cluster Validation", fontsize=15)
fig.tight_layout()
kmeans_metrics_file = OUTPUT_FOLDER / "kmeans_validation_metrics.png"
fig.savefig(kmeans_metrics_file, dpi=300, bbox_inches="tight")
plt.show()

# 6. Hierarchical clustering: Ward and Average linkage

distance_matrix = pdist(pca_feature_data, metric="euclidean")

ward_linkage = linkage(pca_feature_data, method="ward", metric="euclidean")
average_linkage = linkage(pca_feature_data, method="average", metric="euclidean")

ward_cophenetic, _ = cophenet(ward_linkage, distance_matrix)
average_cophenetic, _ = cophenet(average_linkage, distance_matrix)

print("\nHIERARCHICAL CLUSTERING")
print("Ward cophenetic correlation:", round(ward_cophenetic, 4))
print("Average cophenetic correlation:", round(average_cophenetic, 4))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
dendrogram(
    ward_linkage,
    ax=axes[0],
    no_labels=True,
    color_threshold=None,
)
axes[0].set_title("Hierarchical Dendrogram - Ward Linkage")
axes[0].set_xlabel("Wine samples")
axes[0].set_ylabel("Linkage distance")

dendrogram(
    average_linkage,
    ax=axes[1],
    no_labels=True,
    color_threshold=None,
)
axes[1].set_title("Hierarchical Dendrogram - Average Linkage")
axes[1].set_xlabel("Wine samples")
axes[1].set_ylabel("Linkage distance")

fig.tight_layout()
dendrogram_file = OUTPUT_FOLDER / "hierarchical_dendrograms.png"
fig.savefig(dendrogram_file, dpi=300, bbox_inches="tight")
plt.show()

# Use the K-means-selected cluster count for a fair descriptive comparison.
ward_clusters = fcluster(ward_linkage, t=best_k, criterion="maxclust") - 1
average_clusters = fcluster(average_linkage, t=best_k, criterion="maxclust") - 1

hierarchical_results = []
for method_name, assignments in [
    ("Ward", ward_clusters),
    ("Average", average_clusters),
]:
    hierarchical_results.append({
        "method": method_name,
        "clusters": len(np.unique(assignments)),
        "silhouette": silhouette_score(pca_feature_data, assignments),
        "davies_bouldin": davies_bouldin_score(pca_feature_data, assignments),
        "calinski_harabasz": calinski_harabasz_score(
            pca_feature_data, assignments
        ),
    })

hierarchical_results = pd.DataFrame(hierarchical_results)
print("\nHIERARCHICAL CLUSTER VALIDATION")
print(hierarchical_results.round(4).to_string(index=False))


# 7. Optional external validation using the true class labels

# The labels are used only here, after all model choices have been made.
external_validation = pd.DataFrame([
    {
        "method": "K-means",
        "adjusted_rand_index": adjusted_rand_score(true_labels, kmeans_clusters),
        "normalized_mutual_information": normalized_mutual_info_score(
            true_labels, kmeans_clusters
        ),
    },
    {
        "method": "Ward",
        "adjusted_rand_index": adjusted_rand_score(true_labels, ward_clusters),
        "normalized_mutual_information": normalized_mutual_info_score(
            true_labels, ward_clusters
        ),
    },
    {
        "method": "Average",
        "adjusted_rand_index": adjusted_rand_score(true_labels, average_clusters),
        "normalized_mutual_information": normalized_mutual_info_score(
            true_labels, average_clusters
        ),
    },
])

print("\nOPTIONAL EXTERNAL VALIDATION")
print(external_validation.round(4).to_string(index=False))


# 8. Part E: Cluster profiles and centroids in original units


# K-means was trained in PCA space. Its cluster centres are transformed
# back to the standardised feature space, then returned to original units.
centres_in_scaled_units = pca_reduced.inverse_transform(best_kmeans.cluster_centers_)
centres_in_original_units = scaler.inverse_transform(centres_in_scaled_units)

centroid_table = pd.DataFrame(
    centres_in_original_units,
    columns=features.columns,
)
centroid_table.index = [f"Cluster {i}" for i in range(best_k)]
centroid_table.index.name = "Cluster"

print("\nK-MEANS CLUSTER CENTROIDS IN ORIGINAL UNITS")
print(centroid_table.round(3).to_string())

# Cluster profiles use the original measured values.
clustered_features = features_imputed.copy()
clustered_features["cluster"] = kmeans_clusters

cluster_profiles = clustered_features.groupby("cluster")[features.columns].mean()
cluster_sizes = clustered_features["cluster"].value_counts().sort_index()
cluster_profiles.insert(0, "sample_count", cluster_sizes)

print("\nK-MEANS CLUSTER PROFILES")
print(cluster_profiles.round(3).to_string())

# PCA scatter plot using the first two principal components.
pca_for_plot = PCA(n_components=2, random_state=RANDOM_STATE)
pca_plot_values = pca_for_plot.fit_transform(scaled_features)

pca_plot_data = pd.DataFrame({
    "PC1": pca_plot_values[:, 0],
    "PC2": pca_plot_values[:, 1],
    "cluster": kmeans_clusters.astype(str),
})

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=pca_plot_data,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set2",
    s=75,
    ax=ax,
)
ax.set_title("Wine Samples Grouped by K-means Clusters")
ax.set_xlabel(f"PC1 ({pca_for_plot.explained_variance_ratio_[0] * 100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca_for_plot.explained_variance_ratio_[1] * 100:.1f}% variance)")
ax.legend(title="K-means cluster")
fig.tight_layout()
pca_cluster_file = OUTPUT_FOLDER / "pca_clusters.png"
fig.savefig(pca_cluster_file, dpi=300, bbox_inches="tight")
plt.show()


# 9. Statistical inference: alcohol and flavanoids


def calculate_eta_squared(groups, all_values):
    """Calculate one-way ANOVA eta-squared effect size."""
    overall_mean = np.mean(all_values)
    between_group_sum = sum(
        len(group) * (np.mean(group) - overall_mean) ** 2
        for group in groups
    )
    total_sum = np.sum((all_values - overall_mean) ** 2)
    return between_group_sum / total_sum if total_sum > 0 else np.nan


def calculate_epsilon_squared(kruskal_statistic, sample_size, group_count):
    """Calculate an effect size for the Kruskal-Wallis test."""
    if sample_size <= group_count:
        return np.nan
    value = (kruskal_statistic - group_count + 1) / (sample_size - group_count)
    return max(0.0, value)


statistical_results = []

for feature_name in ["alcohol", "flavanoids"]:
    groups = [
        clustered_features.loc[
            clustered_features["cluster"] == cluster_number, feature_name
        ].to_numpy()
        for cluster_number in range(best_k)
    ]

    anova_statistic, anova_p_value = f_oneway(*groups)
    kruskal_statistic, kruskal_p_value = kruskal(*groups)

    all_values = clustered_features[feature_name].to_numpy()
    eta_squared = calculate_eta_squared(groups, all_values)
    epsilon_squared = calculate_epsilon_squared(
        kruskal_statistic,
        len(all_values),
        len(groups),
    )

    statistical_results.append({
        "feature": feature_name,
        "anova_f": anova_statistic,
        "anova_p_value": anova_p_value,
        "eta_squared": eta_squared,
        "kruskal_h": kruskal_statistic,
        "kruskal_p_value": kruskal_p_value,
        "epsilon_squared": epsilon_squared,
    })

statistical_results = pd.DataFrame(statistical_results)

print("\nSTATISTICAL TESTS BETWEEN K-MEANS CLUSTERS")
print(statistical_results.to_string(
    index=False,
    formatters={
        "anova_f": "{:.3f}".format,
        "anova_p_value": "{:.3e}".format,
        "eta_squared": "{:.3f}".format,
        "kruskal_h": "{:.3f}".format,
        "kruskal_p_value": "{:.3e}".format,
        "epsilon_squared": "{:.3f}".format,
    },
))


# Prepare data for the heatmap (Z-scores / standardized values) using a dedicated variable name
heatmap_selected_features = ["alcohol", "malic_acid", "flavanoids", "color_intensity", "proline"]

heatmap_scaled_features = (features[heatmap_selected_features] - features[heatmap_selected_features].mean()) / features[heatmap_selected_features].std()
heatmap_scaled_features["kmeans_cluster"] = kmeans_clusters

cluster_profiles_scaled = heatmap_scaled_features.groupby("kmeans_cluster").mean()

# Plotting the Heatmap with explicit figure handles and show()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    cluster_profiles_scaled, 
    annot=True, 
    fmt=".2f", 
    cmap="vlag", 
    center=0, 
    linewidths=0.5,
    cbar_kws={'label': 'Standard deviations from dataset mean'},
    ax=ax
)
ax.set_title("K-means Cluster Profiles")
ax.set_xlabel("Chemical attribute")
ax.set_ylabel("K-means cluster")
fig.tight_layout()

heatmap_file = OUTPUT_FOLDER / "cluster_profiles_heatmap.png"
fig.savefig(heatmap_file, dpi=300, bbox_inches="tight")
plt.show()


# Plotting Effect Sizes Bar Chart with explicit figure handles and show()
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(statistical_results["feature"]))
width = 0.35

ax.bar(x - width/2, statistical_results["eta_squared"], width, label="ANOVA eta-squared")
ax.bar(x + width/2, statistical_results["epsilon_squared"], width, label="Kruskal epsilon-squared")

ax.set_xlabel("Feature")
ax.set_ylabel("Effect size")
ax.set_title("Effect sizes for differences between clusters")
ax.set_xticks(x)
ax.set_xticklabels(statistical_results["feature"].str.capitalize())
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig.tight_layout()

bar_chart_file = OUTPUT_FOLDER / "effect_sizes_bar_chart.png"
fig.savefig(bar_chart_file, dpi=300, bbox_inches="tight")
plt.show()


print("\nINTERPRETATION GUIDE")
print("A p-value below 0.05 suggests that at least one cluster differs.")
print("Eta-squared and epsilon-squared describe the size of the difference.")
print("A small p-value does not by itself mean the difference is large.")

# 10. Part F: Save results and create a simple analysis report


# Save the main result tables.
kmeans_results.to_csv(OUTPUT_FOLDER / "kmeans_validation_results.csv", index=False)
hierarchical_results.to_csv(
    OUTPUT_FOLDER / "hierarchical_validation_results.csv", index=False
)
external_validation.to_csv(
    OUTPUT_FOLDER / "external_validation_results.csv", index=False
)
centroid_table.to_csv(OUTPUT_FOLDER / "cluster_centroids_original_units.csv")
cluster_profiles.to_csv(OUTPUT_FOLDER / "cluster_profiles.csv")
statistical_results.to_csv(
    OUTPUT_FOLDER / "statistical_test_results.csv", index=False
)

# Save sample-level cluster assignments separately.
sample_assignments = pd.DataFrame({
    "sample_number": np.arange(1, len(features) + 1),
    "kmeans_cluster": kmeans_clusters,
    "ward_cluster": ward_clusters,
    "average_cluster": average_clusters,
})
sample_assignments.to_csv(
    OUTPUT_FOLDER / "sample_cluster_assignments.csv", index=False
)

# A compact PDF report is created from the key charts and findings.

pdf_file = OUTPUT_FOLDER / "wine_clustering_report.pdf"

with PdfPages(pdf_file) as pdf:
    # Summary page
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, "Wine Dataset Clustering Report", fontsize=20, weight="bold")
    summary_lines = [
        f"Dataset: UCI Wine dataset ({len(features)} samples, {features.shape[1]} attributes)",
        "Class labels were excluded from model fitting and cluster selection.",
        f"PCA retained {number_of_components} components and "
        f"{cumulative_variance[number_of_components - 1] * 100:.2f}% of variance.",
        f"Selected K-means cluster count: {best_k}",
        "",
        "K-means validation:",
        kmeans_results.round(4).to_string(index=False),
        "",
        "Hierarchical cophenetic correlations:",
        f"Ward: {ward_cophenetic:.4f}",
        f"Average: {average_cophenetic:.4f}",
        "",
        "External validation (labels used only after clustering):",
        external_validation.round(4).to_string(index=False),
        "",
        "Statistical tests:",
        statistical_results.to_string(
            index=False,
            formatters={
                "anova_p_value": "{:.3e}".format,
                "kruskal_p_value": "{:.3e}".format,
            },
        ),
        "",
        "Interpretation:",
        "Silhouette is better when higher. Davies-Bouldin is better when lower.",
        "Calinski-Harabasz is better when higher. Cophenetic correlation measures",
        "how faithfully a dendrogram represents the pairwise distances.",
        "ANOVA and Kruskal-Wallis test whether feature values differ across clusters.",
    ]
    fig.text(
        0.08, 0.89, "\n".join(summary_lines),
        fontsize=8.5, va="top", family="monospace"
    )
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    for image_file in [
        histogram_file,
        scree_file,
        kmeans_metrics_file,
        dendrogram_file,
        pca_cluster_file,
        heatmap_file,
        bar_chart_file,
    ]:
        image = plt.imread(image_file)
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        ax.imshow(image)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("\nANALYSIS COMPLETE")
print("All output files have been saved in:", OUTPUT_FOLDER)
print("PDF report:", pdf_file)
print("Scaled dataset:", scaled_file)