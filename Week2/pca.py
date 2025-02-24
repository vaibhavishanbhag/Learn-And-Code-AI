# Continuation after preprocessing.py

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Convert to DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_df["PC1"], X_pca_df["PC2"], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Titanic Dataset (2D Projection)")
plt.colorbar(label="Survived")
plt.show()

# Check variance explained by components
explained_variance = pca.explained_variance_ratio_
print(f"Variance Explained by PC1: {explained_variance[0]*100:.2f}%")
print(f"Variance Explained by PC2: {explained_variance[1]*100:.2f}%")
print(f"Total Variance Explained: {np.sum(explained_variance)*100:.2f}%")


# Apply PCA without specifying components to check cumulative variance
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Cumulative explained variance
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot explained variance vs. number of components
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o", linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance vs. Number of Components")
plt.axhline(y=0.90, color="r", linestyle="--", label="90% Variance")
plt.legend()
plt.grid()
plt.show()

# Find the number of components that explain at least 90% variance
optimal_components = np.argmax(explained_variance >= 0.90) + 1
print(f"Optimal number of components to retain 90% variance: {optimal_components}")


# Apply PCA with optimal number of components (6)
pca_optimal = PCA(n_components=6)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Check how much variance is retained
explained_variance_optimal = np.sum(pca_optimal.explained_variance_ratio_) * 100
print(f"Total Variance Retained with 6 Components: {explained_variance_optimal:.2f}%")
