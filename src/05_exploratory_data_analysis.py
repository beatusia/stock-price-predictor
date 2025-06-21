# src/05_exploratiry_data_analysis.py
# Description: This script performs exploratory data analysis (EDA) on the stock price dataset.

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration


# Load all stock data and merge them into a single DataFrame
folder_path = "/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/merged"
df_APPL = pd.read_csv(f"{folder_path}/AAPL.csv")
df_APPL.head()
len(df_APPL.columns)

# Preliminary checks
print(df_APPL["Date"].min(), df_APPL["Date"].max())
print(df_APPL.groupby("TICKER").agg({"Date": ["min", "max"]}))
print(df_APPL.isna().sum().sort_values(ascending=False).head(20))
df_APPL.isna().sum()

# Remove psar
df_APPL = df_APPL.drop(columns=["psar"], errors="ignore")
len(df_APPL.columns)

# Summary of nulls
feature_cols = df_APPL.columns.tolist()
feature_nulls = df_APPL[feature_cols].isnull().sum().sort_values(ascending=False)
print("Missing data per feature:\n", feature_nulls[feature_nulls > 0])

# Correlation heatmap
# Select only numeric columns
numeric_cols = df_APPL.select_dtypes(include="number").columns

# Compute correlation matrix
corr = df_APPL[numeric_cols].corr()

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# Check for highly correlated features
# Only numeric columns
numeric_df = df_APPL.select_dtypes(include="number")

# Compute correlation matrix
corr_matrix = numeric_df.corr().abs()

# Keep upper triangle of the matrix (exclude self-correlations)
upper = corr_matrix.where(
    pd.DataFrame(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
        columns=corr_matrix.columns,
        index=corr_matrix.index,
    )
)

# Threshold for high correlation
correlation_threshold = 0.95

# Find pairs with correlation above the threshold
high_corr_pairs = [
    (col1, col2, corr_val)
    for col1 in upper.columns
    for col2 in upper.index
    if pd.notnull(upper.loc[col2, col1])
    and upper.loc[col2, col1] >= correlation_threshold
    for corr_val in [upper.loc[col2, col1]]
]

# Display
print(f"Highly correlated pairs (|r| ≥ {correlation_threshold}):")
for col1, col2, val in sorted(high_corr_pairs, key=lambda x: -x[2]):
    print(f"{col1} ↔ {col2}: r = {val:.3f}")


# PCA to reduce dimensionality
# Select numeric columns (excluding 'Target')
X = df_APPL.select_dtypes(include="number").drop(columns=["Target"], errors="ignore")
X = X.dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_var = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(explained_var >= 0.90) + 1
print(f"✅ Number of PCs explaining ≥90% variance: {n_components_90}")


# Step 1: Extract numeric features (excluding target)
X = df_APPL.select_dtypes(include="number").drop(columns=["Target"], errors="ignore")
feature_names = X.columns.tolist()

# Step 2: Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Step 3: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 4: PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 5: Get PCA loadings (feature contributions to each PC)
loadings = pd.DataFrame(
    pca.components_,  # Shape: [n_components, n_features]
    columns=feature_names,
    index=[f"PC{i + 1}" for i in range(len(pca.components_))],
)

# View Top Features Driving Each PC
top_features_per_pc = {}

for pc in loadings.index:
    top_features = loadings.loc[pc].abs().sort_values(ascending=False).head(5)
    top_features_per_pc[pc] = top_features.index.tolist()

print("🔍 Top 5 contributing features per PC:")
for pc, features in top_features_per_pc.items():
    print(f"{pc}: {features}")

# Visualize Loadings for First Few PCs
# Show feature loadings for PC1, PC2, PC3, PC4, and PC5
for i in range(7):  # First 3 PCs
    pc = f"PC{i + 1}"
    loadings_pc = loadings.loc[pc].sort_values(key=np.abs, ascending=False)[:10]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=loadings_pc.values, y=loadings_pc.index)
    plt.title(f"Top 10 Feature Loadings for {pc}")
    plt.xlabel("Loading Coefficient")
    plt.show()

# Union of top 5 features from PCs explaining 90% of variance
explained_var = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(explained_var >= 0.90) + 1

important_features = set()
for i in range(n_components_90):
    pc = f"PC{i + 1}"
    important_features.update(
        loadings.loc[pc].abs().sort_values(ascending=False).head(5).index
    )

print(f"🎯 Selected {len(important_features)} important features:")
print(sorted(important_features))

# 6. Get the PCA components (loadings)
loadings = pd.DataFrame(
    pca.components_[:n_components_90],
    columns=df_numeric.columns,
    index=[f"PC{i + 1}" for i in range(n_components_90)],
)

# Exploratory Data Analysis (EDA) using PCA to identify important features
# 1. Load your data (replace with your actual DataFrame)
df = X  # or use df_APPL if already loaded

# 2. Drop non-numeric columns and rows with NaN
df_numeric = df.select_dtypes(include=[np.number]).dropna()

# 3. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# 4. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 5. Determine number of components to explain 90% variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"✅ {n_components_90} components explain ≥90% of the variance.")

# 6. Get the PCA components (loadings)
loadings = pd.DataFrame(
    pca.components_[:n_components_90],
    columns=df_numeric.columns,
    index=[f"PC{i + 1}" for i in range(n_components_90)],
)

# 7. For each component, get top contributing features
top_features = {}
for pc in loadings.index:
    top = loadings.loc[pc].abs().sort_values(ascending=False)
    top_features[pc] = top.head(5).index.tolist()  # change 5 if you want more/less

# 8. Print or combine all top features
print("\n📌 Top contributing features to PCs explaining 90% variance:")
for pc, features in top_features.items():
    print(f"{pc}: {features}")

# 9. Unique set of important features (non-redundant)
important_features = sorted(set(f for lst in top_features.values() for f in lst))
print("\n✅ Recommended features based on PCA:", important_features)

len(important_features)

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), pca.explained_variance_ratio_[:5], marker="o", label="Individual")
plt.plot(
    range(1, 6),
    np.cumsum(pca.explained_variance_ratio_[:5]),
    marker="s",
    linestyle="--",
    label="Cumulative",
)
plt.title("Explained Variance of First 5 Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.show()

# Visualize PCA results using scatter plots for the first 5 principal components
# Create a DataFrame for first 5 PCs
pca_df = pd.DataFrame(X_pca[:, :5], columns=[f"PC{i + 1}" for i in range(5)])

# PC1 vs PC2
sns.scatterplot(x="PC1", y="PC2", data=pca_df)
plt.title("PC1 vs PC2")
plt.grid(True)
plt.show()

# PC2 vs PC3
sns.scatterplot(x="PC2", y="PC3", data=pca_df)
plt.title("PC2 vs PC3")
plt.grid(True)
plt.show()

# PC3 vs PC4
sns.scatterplot(x="PC3", y="PC4", data=pca_df)
plt.title("PC3 vs PC4")
plt.grid(True)
plt.show()

# UMAP for dimensionality reduction and visualization
# Range of clusters to try
cluster_range = range(2, 11)

inertia = []
silhouette = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(embedding, labels))

# Plot Elbow Method (Inertia)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.grid(True)

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette, "ro-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores For Various k")
plt.grid(True)

plt.tight_layout()
plt.show()

# After visually inspecting these plots,
# set your chosen k, for example:
optimal_k = 7

# Run KMeans with optimal clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df.loc[X.index, "feature_cluster"] = kmeans.fit_predict(embedding)

# Plot final clustering
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="UMAP1", y="UMAP2", hue="feature_cluster", data=df, palette="viridis", alpha=0.7
)
plt.title(f"UMAP Clustering of Stock Features (k={optimal_k})")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="Feature Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()

# Add UMAP embeddings as new columns
X.loc[X.index, "UMAP1"] = embedding[:, 0]
X.loc[X.index, "UMAP2"] = embedding[:, 1]
X.head()

# Add feature cluster labels to the original DataFrame
X.loc[X.index, "feature_cluster"] = kmeans.labels_
X["feature_cluster"] = X["feature_cluster"].astype("category")
X.head()


# Feature Agglomeration to reduce features
# agglo = FeatureAgglomeration(n_clusters=20)  # cluster into 20 groups
# X_reduced = agglo.fit_transform(X.select_dtypes(include='number'))
# print("Reduced features shape:", X_reduced.shape)

# Determine ideal number of clusters for Feature Agglomeration
X = X.select_dtypes(include="number")
print("X shape:", X.shape)  # (samples, features)
print(
    "Labels length:", len(labels)
)  # should be equal to number of features (X.shape[1])


def avg_within_cluster_corr(X, labels):
    """
    Calculate average pairwise absolute correlation within each feature cluster.
    """
    corr = X.corr().abs()
    avg_corrs = []
    unique_labels = np.unique(labels)

    for cluster in unique_labels:
        mask = labels == cluster
        features_in_cluster = X.columns[mask]

        if len(features_in_cluster) <= 1:
            # For clusters with 1 or 0 features, treat correlation as 1 (perfect)
            avg_corrs.append(1.0)
            continue

        sub_corr = corr.loc[features_in_cluster, features_in_cluster]
        # Extract upper triangle without diagonal values
        vals = sub_corr.where(
            np.triu(np.ones(sub_corr.shape), k=1).astype(bool)
        ).stack()
        avg_corrs.append(vals.mean())

    return np.mean(avg_corrs)


cluster_range = range(5, 31, 5)  # Try 5, 10, 15, 20, 25, 30 clusters
results = []

for n_clusters in cluster_range:
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X.T)  # Fit on features x samples (transpose)
    labels = agglo.labels_

    # Safety check
    assert len(labels) == X.shape[1], (
        "Mismatch between labels length and number of features!"
    )

    avg_corr = avg_within_cluster_corr(X, labels)
    print(
        f"n_clusters={n_clusters} -> Average within-cluster correlation: {avg_corr:.4f}"
    )
    results.append({"n_clusters": n_clusters, "avg_within_cluster_corr": avg_corr})

results_df = pd.DataFrame(results)

# Plotting results
plt.figure(figsize=(8, 5))
plt.plot(results_df["n_clusters"], results_df["avg_within_cluster_corr"], marker="o")
plt.xlabel("Number of Feature Clusters")
plt.ylabel("Average Within-Cluster Feature Correlation")
plt.title("FeatureAgglomeration Cluster Quality")
plt.grid(True)
plt.show()
