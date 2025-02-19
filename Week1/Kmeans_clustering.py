from sklearn.cluster import KMeans
import numpy as np

# Sample data: Customers' spending behavior
X = np.array([[500, 50], [520, 55], [700, 80], [710, 85], [300, 20]])

# Train K-Means model with 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Predict cluster for a new customer
print("Cluster:", kmeans.predict([[600, 60]]))
print("Cluster:", kmeans.predict([[650, 70]]))