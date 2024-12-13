from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import hdbscan
import pandas as pd


class ClusteringModel:
    def __init__(self, algorithm="kmeans", n_clusters=5, random_state=42, **kwargs):
        """
        Initialize the clustering model.

        Args:
            algorithm (str): The clustering algorithm to use ('kmeans', 'hdbscan', 'agglomerative').
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
        elif self.algorithm == "hdbscan":
            self.model = hdbscan.HDBSCAN(**self.kwargs)
        elif self.algorithm == "agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage="ward"
            )
        else:
            raise ValueError(
                "Unsupported algorithm. Choose 'kmeans', 'hdbscan', or 'agglomerative'."
            )

    def fit(self, data):
        """
        Fit the model on the data.

        Args:
            data (DataFrame): Input data with features for clustering.

        Returns:
            DataFrame: Input data with an additional 'cluster' column.
        """
        cluster_labels = self.model.fit_predict(data)
        data["cluster"] = cluster_labels
        return data


if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_file_path = "/home/alierdem/mcn_pjkt/data/preprocessed_data.csv"
    data = pd.read_csv(preprocessed_file_path)

    # Select relevant columns for clustering
    clustering_data = data[["lat", "lon", "speed"]]

    # Standardize the features
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    # # KMeans Example
    # print("Running KMeans...")
    # kmeans_model = ClusteringModel(algorithm="kmeans", n_clusters=3)
    # kmeans_clustered_data = kmeans_model.fit(pd.DataFrame(clustering_data_scaled))
    # kmeans_clustered_data_with_original = data.copy()
    # kmeans_clustered_data_with_original["cluster"] = kmeans_clustered_data["cluster"]
    #
    # kmeans_clustered_file_path = (
    #     "/home/alierdem/mcn_pjkt/data/kmeans_clustered_data.csv"
    # )
    # kmeans_clustered_data_with_original.to_csv(kmeans_clustered_file_path, index=False)
    # print(f"KMeans clustered data saved to {kmeans_clustered_file_path}")

    # HDBSCAN Example
    hdbscan_sample_fraction = 0.5  # Increased fraction
    print(f"Using {hdbscan_sample_fraction*100}% of the data for HDBSCAN.")

    # Sample data and retain indices
    hdbscan_sampled_indices = clustering_data.sample(
        frac=hdbscan_sample_fraction, random_state=42
    ).index
    hdbscan_sampled_data = clustering_data_scaled[hdbscan_sampled_indices]

    # Run HDBSCAN
    print("Running HDBSCAN...")
    hdbscan_model = ClusteringModel(
        algorithm="hdbscan", min_samples=10, min_cluster_size=5
    )
    hdbscan_clustered_data = hdbscan_model.fit(pd.DataFrame(hdbscan_sampled_data))

    # Merge cluster labels with original sampled data
    hdbscan_clustered_data_with_original = data.iloc[hdbscan_sampled_indices].copy()
    hdbscan_clustered_data_with_original["cluster"] = hdbscan_clustered_data[
        "cluster"
    ].values

    # Save the clustered data
    hdbscan_clustered_file_path = (
        "/home/alierdem/mcn_pjkt/data/hdbscan_clustered_data.csv"
    )
    hdbscan_clustered_data_with_original.to_csv(
        hdbscan_clustered_file_path, index=False
    )
    print(f"HDBSCAN clustered data saved to {hdbscan_clustered_file_path}")

    # # Downscale the data for Agglomerative Clustering
    # agglomerative_sample_fraction = 0.001  # Adjust as needed
    # print(
    #     f"Using {agglomerative_sample_fraction*100}% of the data for Agglomerative Clustering."
    # )
    # # Sample data and retain indices
    # agglomerative_sampled_indices = clustering_data.sample(
    #     frac=agglomerative_sample_fraction, random_state=42
    # ).index
    # agglomerative_sampled_data = clustering_data_scaled[agglomerative_sampled_indices]
    #
    # # Run Agglomerative Clustering
    # print("Running Agglomerative Clustering...")
    # agglomerative_model = ClusteringModel(algorithm="agglomerative", n_clusters=3)
    # agglomerative_clustered_data = agglomerative_model.fit(
    #     pd.DataFrame(agglomerative_sampled_data)
    # )
    #
    # # Merge cluster labels with original sampled data
    # agglomerative_clustered_data_with_original = data.iloc[
    #     agglomerative_sampled_indices
    # ].copy()
    # agglomerative_clustered_data_with_original["cluster"] = (
    #     agglomerative_clustered_data["cluster"].values
    # )
    #
    # # Save the clustered data
    # agglomerative_clustered_file_path = (
    #     "/home/alierdem/mcn_pjkt/data/agglomerative_clustered_data.csv"
    # )
    # agglomerative_clustered_data_with_original.to_csv(
    #     agglomerative_clustered_file_path, index=False
    # )
    # print(f"Agglomerative clustered data saved to {agglomerative_clustered_file_path}")
