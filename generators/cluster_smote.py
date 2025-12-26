import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
class ClusterSMOTE:
    def __init__(self, n_clusters=5, k_neighbors=5, random_state=42):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    def generate(self, X, y):
        minority_class = np.min(np.unique(y))
        X_min = X[y == minority_class]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(X_min)
        synthetic_samples = []
        for cluster_id in np.unique(clusters):
            cluster_points = X_min[clusters == cluster_id]
            if len(cluster_points) <= 1:
                continue
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(cluster_points)))
            nn.fit(cluster_points)
            for point in cluster_points:
                neighbor = cluster_points[
                    nn.kneighbors([point], return_distance=False)[0][1]
                ]
                gap = np.random.rand()
                synthetic = point + gap * (neighbor - point)
                synthetic_samples.append(synthetic)
        X_syn = np.vstack(synthetic_samples)
        y_syn = np.full(len(X_syn), minority_class)
        return np.vstack([X, X_syn]), np.hstack([y, y_syn])
