from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from collection import DataCollectionLayer
from preprocess import DataPreprocessingLayer
import pandas as pd
import numpy as np


class ClusteringModel:
    def __init__(self, algorithm="kmeans", n_clusters=5, random_state=42, **kwargs):
        """
        Initialize the clustering model.

        Args:
            algorithm (str): The clustering algorithm to use ('kmeans', 'dbscan', 'agglomerative').
            n_clusters (int): Number of clusters (used for KMeans and Agglomerative).
            random_state (int): Random state for reproducibility (used for KMeans).
            **kwargs: Additional keyword arguments for specific clustering algorithms.
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs

        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state
            )
        elif self.algorithm == "dbscan":
            self.model = DBSCAN(**self.kwargs)
        elif self.algorithm == "agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage="ward"
            )
        else:
            raise ValueError(
                "Unsupported algorithm. Choose 'kmeans', 'dbscan', or 'agglomerative'."
            )

    def fit(self, data, sample_fraction=1.0):
        """
        Fit the model on the data.

        Args:
            data (DataFrame): Input data containing 'lat' and 'lon' columns.
            sample_fraction (float): Fraction of the data to sample for DBSCAN. Default is 1.0 (no sampling).

        Returns:
            DataFrame: Input data with an additional 'cluster' column.
        """
        if "lat" not in data.columns or "lon" not in data.columns:
            raise ValueError("Data must contain 'lat' and 'lon' columns.")

        # Reduce dataset size for large-scale clustering
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=self.random_state)

        try:
            if self.algorithm in ["kmeans", "dbscan"]:
                data["cluster"] = self.model.fit_predict(data[["lat", "lon"]])
            elif self.algorithm == "agglomerative":
                data["cluster"] = self.model.fit_predict(data[["lat", "lon"]])
                self.cluster_data = data  # Save clustered data for centroid calculation
        except MemoryError as e:
            print(f"Memory error during clustering with {self.algorithm}: {e}")
            data["cluster"] = -1  # Assign -1 to all points if clustering fails
        return data

    def predict(self, new_data):
        """
        Predict the cluster for new data points.

        Args:
            new_data (DataFrame): Input data containing 'lat' and 'lon' columns.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if self.algorithm == "kmeans":
            return self.model.predict(new_data[["lat", "lon"]])
        elif self.algorithm == "agglomerative":
            return self._predict_agglomerative(new_data)
        elif self.algorithm == "dbscan":
            return self._predict_dbscan(new_data)
        else:
            raise NotImplementedError("Prediction is not supported for this algorithm.")

    def _predict_agglomerative(self, new_data):
        """
        Approximate prediction for Agglomerative Clustering by assigning to the nearest cluster.

        Args:
            new_data (DataFrame): Input data containing 'lat' and 'lon' columns.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        cluster_centers = self._calculate_cluster_centroids()
        distances = np.linalg.norm(
            new_data[["lat", "lon"]].values[:, np.newaxis] - cluster_centers, axis=2
        )
        return np.argmin(distances, axis=1)

    def _predict_dbscan(self, new_data):
        """
        Approximate prediction for DBSCAN by checking if the point satisfies density conditions.
        Optimized for large datasets using smaller chunks.
        """
        core_sample_indices = self.model.core_sample_indices_
        core_samples = self.model.components_

        if core_samples is None or len(core_samples) == 0:
            raise ValueError(
                "DBSCAN model has no core samples. Try adjusting eps or min_samples."
            )

        predictions = np.full(len(new_data), -1, dtype=int)  # Default to noise (-1)
        chunk_size = 10  # Smaller chunks for memory efficiency

        for start_idx in range(0, len(new_data), chunk_size):
            end_idx = min(start_idx + chunk_size, len(new_data))
            chunk = new_data.iloc[start_idx:end_idx]

            # Compute distances
            distances = np.linalg.norm(
                chunk[["lat", "lon"]].values[:, np.newaxis] - core_samples, axis=2
            )
            is_core_point = (distances < self.model.eps).any(axis=1)

            # Assign nearest core sample cluster label
            core_labels = self.model.labels_[core_sample_indices]
            nearest_core_index = np.argmin(distances, axis=1)
            predictions[start_idx:end_idx] = np.where(
                is_core_point, core_labels[nearest_core_index], -1
            )

        return predictions

    def _calculate_cluster_centroids(self):
        """
        Calculate cluster centroids for Agglomerative Clustering.

        Returns:
            np.ndarray: Array of cluster centroids.
        """
        cluster_centroids = []
        for cluster in np.unique(self.cluster_data["cluster"]):
            if cluster != -1:  # Exclude noise points
                cluster_points = self.cluster_data[
                    self.cluster_data["cluster"] == cluster
                ][["lat", "lon"]].values
                cluster_centroids.append(cluster_points.mean(axis=0))
        return np.array(cluster_centroids)


if __name__ == "__main__":
    # Load and preprocess data
    data_layer = DataCollectionLayer(
        "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    )
    raw_data = data_layer.load_data()
    preprocess_layer = DataPreprocessingLayer(raw_data)
    clean_data = preprocess_layer.preprocess_data()
    transformed_data = preprocess_layer.transform_data()

    # Split data into training and test sets
    train_data = transformed_data.sample(frac=0.8, random_state=42)  # 80% for training
    test_data = transformed_data.drop(train_data.index)  # 20% for testing

    if test_data.empty:
        raise ValueError("Test data is empty. Reduce the train_data sample fraction.")

    # KMeans Example
    try:
        kmeans_model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        kmeans_clustered_data = kmeans_model.fit(train_data)
        kmeans_predictions = kmeans_model.predict(test_data)
        print("KMeans Predictions:", kmeans_predictions[:5])
    except Exception as e:
        print(f"KMeans Error: {e}")

    # DBSCAN Example
    try:
        dbscan_model = ClusteringModel(algorithm="dbscan", eps=0.05, min_samples=5)
        dbscan_clustered_data = dbscan_model.fit(train_data, sample_fraction=0.001)
        test_data_dbscan = test_data.sample(
            frac=0.01, random_state=42
        )  # Reduce test size
        dbscan_predictions = dbscan_model.predict(test_data_dbscan)
        print("DBSCAN Predictions:", dbscan_predictions[:5])
    except Exception as e:
        print(f"DBSCAN Error: {e}")

    # Agglomerative Clustering Example
    try:
        agglomerative_model = ClusteringModel(algorithm="agglomerative", n_clusters=3)
        train_data_agglomerative = train_data.sample(
            n=10000, random_state=42
        )  # Reduce training size
        agglomerative_clustered_data = agglomerative_model.fit(train_data_agglomerative)
        test_data_agglomerative = test_data.sample(
            frac=0.01, random_state=42
        )  # Reduce test size
        agglomerative_predictions = agglomerative_model.predict(test_data_agglomerative)
        print("Agglomerative Predictions:", agglomerative_predictions[:5])
    except Exception as e:
        print(f"Agglomerative Error: {e}")
