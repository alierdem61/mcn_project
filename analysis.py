import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pickle


class PatternAnalysis:
    def __init__(self, clustered_data, normalization_params_path):
        """
        Initialize PatternAnalysis with clustered data.

        Args:
            clustered_data (DataFrame): Data with cluster assignments and relevant features.
            normalization_params_path (str): Path to the file containing normalization parameters.
        """
        self.data = clustered_data
        self.normalization_params_path = normalization_params_path
        self.scaler = None
        self._load_normalization_params()

    def _load_normalization_params(self):
        """
        Load normalization parameters from the specified .pkl file.
        """
        if not os.path.exists(self.normalization_params_path):
            raise FileNotFoundError(
                f"Normalization parameters file not found at {self.normalization_params_path}."
            )
        with open(self.normalization_params_path, "rb") as f:
            self.scaler = pickle.load(f)  # Single scaler for all three features
        print("Normalization parameters loaded.")

    def undo_normalization(self):
        """
        Undo normalization for latitude, longitude, and speed using a single scaler.
        """
        if self.scaler is None:
            raise ValueError("Normalization parameters are not loaded.")

        # Extract normalized features
        normalized_features = self.data[["lat", "lon", "speed"]].values

        # Apply inverse transform using the combined scaler
        original_features = self.scaler.inverse_transform(normalized_features)

        # Restore original values
        self.data[["lat", "lon", "speed"]] = original_features

        print("Normalization undone.")

    def analyze_clusters(self):
        """
        Perform analysis on the clustered data:
        - Calculate cluster centers.
        - Calculate cluster sizes.
        - Perform temporal analysis for peak hours.
        """
        # Cluster center analysis
        cluster_centers = self.data.groupby("cluster")[["lat", "lon"]].mean()
        print("Cluster Centers:\n", cluster_centers)

        # Cluster size and density
        cluster_sizes = self.data["cluster"].value_counts()
        print("\nCluster Sizes:\n", cluster_sizes)

        # Temporal analysis (assuming 'hour' column exists)
        if "timestamp" in self.data.columns:
            self.data["hour"] = pd.to_datetime(self.data["timestamp"]).dt.hour
            peak_hours = (
                self.data.groupby("cluster")["hour"]
                .value_counts()
                .unstack(fill_value=0)
            )
            print("\nPeak Hours for Each Cluster:\n", peak_hours)

    def visualize_clusters(self, algo):
        """
        Create and save a scatter plot of the clusters.

        Args:
            algo (str): The clustering algorithm used (for file naming).
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.data["lon"],
            self.data["lat"],
            c=self.data["cluster"],
            cmap="viridis",
            alpha=0.5,
        )
        plt.colorbar(label="Cluster")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("User Mobility Clusters")
        plt.savefig(f"{algo}_cluster_map.png")  # Save plot as image file
        print(f"Cluster map saved as {algo}_cluster_map.png")

    def plot_temporal_heatmap(self, algo):
        """
        Create and save a heatmap for peak hours by cluster.

        Args:
            algo (str): The clustering algorithm used (for file naming).
        """
        if "hour" not in self.data.columns:
            self.data["hour"] = pd.to_datetime(self.data["timestamp"]).dt.hour

        heatmap_data = (
            self.data.groupby(["cluster", "hour"]).size().unstack(fill_value=0)
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap_data, aspect="auto", cmap="coolwarm", origin="lower")
        plt.colorbar(label="Frequency")
        plt.xlabel("Hour of Day")
        plt.ylabel("Cluster")
        plt.title("Peak Hours Heatmap for Each Cluster")
        plt.savefig(f"{algo}_temporal_heatmap.png")  # Save plot as image file
        print(f"Temporal heatmap saved as {algo}_temporal_heatmap.png")


if __name__ == "__main__":
    # Perform analysis and visualization
    algorithm_used = "kmeans"  # Replace with the actual algorithm used

    # Path to the clustered data CSV file
    clustered_file_path = (
        f"/home/alierdem/mcn_pjkt/data/{algorithm_used}_clustered_data.csv"
    )
    normalization_params_path = "/home/alierdem/mcn_pjkt/data/normalization_data.pkl"

    if not os.path.exists(clustered_file_path):
        raise FileNotFoundError(
            f"Clustered data file not found at {clustered_file_path}. Please run clustering first."
        )

    # Load the clustered data
    clustered_data = pd.read_csv(clustered_file_path)

    # Initialize PatternAnalysis
    pattern_analysis = PatternAnalysis(clustered_data, normalization_params_path)

    # Undo normalization before analysis
    pattern_analysis.undo_normalization()

    # Perform analysis and visualization
    pattern_analysis.analyze_clusters()
    pattern_analysis.visualize_clusters(algorithm_used)
    pattern_analysis.plot_temporal_heatmap(algorithm_used)
