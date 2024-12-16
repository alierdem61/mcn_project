import sys
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
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
        elif self.algorithm == "birch":
            self.model = Birch(n_clusters=self.n_clusters, **self.kwargs)
        else:
            raise ValueError(
                "Unsupported algorithm. Choose 'kmeans', 'hdbscan', 'birch' or 'agglomerative'."
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


def run_clustering(algorithm, data, clustering_data):
    """
    Run the specified clustering algorithm.

    Args:
        algorithm (str): The clustering algorithm to run.
        data (DataFrame): Original data.
        clustering_data (DataFrame): Data for clustering.
    """
    if algorithm == "kmeans":
        print("Running KMeans...")
        model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        clustered_data = model.fit(pd.DataFrame(clustering_data))
        save_clustered_data(clustered_data, data, algorithm)

    elif algorithm == "hdbscan":
        print("Running HDBSCAN...")
        sampled_indices = clustering_data.sample(frac=0.2, random_state=42).index
        sampled_data = clustering_data.loc[sampled_indices]
        model = ClusteringModel(
            algorithm="hdbscan",
            min_samples=15,
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_epsilon=0.2,
        )
        clustered_data = model.fit(pd.DataFrame(sampled_data))
        save_clustered_data(clustered_data, data.iloc[sampled_indices], algorithm)

    elif algorithm == "birch":
        print("Running Birch...")
        model = ClusteringModel(
            algorithm="birch", n_clusters=None, threshold=0.45, branching_factor=50
        )
        clustered_data = model.fit(pd.DataFrame(clustering_data))
        save_clustered_data(clustered_data, data, algorithm)

    elif algorithm == "agglomerative":
        print("Running Agglomerative Clustering...")
        sampled_indices = clustering_data.sample(frac=0.001, random_state=42).index
        sampled_data = clustering_data.loc[sampled_indices]
        model = ClusteringModel(algorithm="agglomerative", n_clusters=3)
        clustered_data = model.fit(pd.DataFrame(sampled_data))
        save_clustered_data(clustered_data, data.iloc[sampled_indices], algorithm)

    else:
        print(f"Unsupported algorithm: {algorithm}")


def save_clustered_data(clustered_data, original_data, algorithm):
    """
    Save clustered data to a CSV file.

    Args:
        clustered_data (DataFrame): Clustered data with cluster labels.
        original_data (DataFrame): Original data corresponding to the clustered data.
        algorithm (str): Name of the clustering algorithm.
    """
    clustered_data_with_original = original_data.copy()
    clustered_data_with_original["cluster"] = clustered_data["cluster"]
    output_file = f"/home/alierdem/mcn_pjkt/data/{algorithm}_clustered_data.csv"
    clustered_data_with_original.to_csv(output_file, index=False)
    print(f"{algorithm.capitalize()} clustered data saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python cluster.py <algorithm(s)>")
        print("Available algorithms: kmeans, hdbscan, birch, agglomerative, all")
        sys.exit(1)

    algorithms_to_run = sys.argv[1:]
    if "all" in algorithms_to_run:
        algorithms_to_run = ["kmeans", "hdbscan", "birch", "agglomerative"]

    # Load preprocessed data
    preprocessed_file_path = "/home/alierdem/mcn_pjkt/data/preprocessed_data_china.csv"
    data = pd.read_csv(preprocessed_file_path)

    # Select relevant columns for clustering
    clustering_data = data[["lat", "lon", "speed"]]

    # Run clustering for specified algorithms
    for algorithm in algorithms_to_run:
        run_clustering(algorithm.lower(), data, clustering_data)
