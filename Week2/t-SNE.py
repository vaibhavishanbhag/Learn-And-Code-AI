#Continuation after pca.py 

from sklearn.manifold import TSNE

# Apply t-SNE on PCA-reduced data (6 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_pca_optimal)

# Convert to DataFrame
X_tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

# Plot t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne_df["TSNE1"], X_tsne_df["TSNE2"], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE: Titanic Dataset Clustering")
plt.colorbar(label="Survived")
plt.show()
