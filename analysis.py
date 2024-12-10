import matplotlib.pyplot as plt
from collection import DataCollectionLayer
from preprocess import DataPreprocessingLayer
from cluster import ClusteringModel


class PatternAnalysis:
    def __init__(self, clustered_data):
        self.data = clustered_data

    def analyze_clusters(self):
        # Cluster center analysis
        cluster_centers = self.data.groupby("cluster")[["lat", "lon"]].mean()
        print("Cluster Centers:\n", cluster_centers)

        # Cluster size and density
        cluster_sizes = self.data["cluster"].value_counts()
        print("\nCluster Sizes:\n", cluster_sizes)

        # Temporal analysis (assuming 'timestamp' column exists)
        self.data["hour"] = self.data["timestamp"].dt.hour
        peak_hours = (
            self.data.groupby("cluster")["hour"].value_counts().unstack(fill_value=0)
        )
        print("\nPeak Hours for Each Cluster:\n", peak_hours)

    def visualize_clusters(self):
        # Map clusters with color-coded markers
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
        plt.savefig("cluster_map.png")  # Save plot as image file
        print("Cluster map saved as cluster_map.png")

    def plot_temporal_heatmap(self):
        # Example temporal heatmap for peak hours by cluster
        heatmap_data = (
            self.data.groupby(["cluster", "hour"]).size().unstack(fill_value=0)
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap_data, aspect="auto", cmap="coolwarm")
        plt.colorbar(label="Frequency")
        plt.xlabel("Hour of Day")
        plt.ylabel("Cluster")
        plt.title("Peak Hours Heatmap for Each Cluster")
        plt.savefig("temporal_heatmap.png")  # Save plot as image file
        print("Temporal heatmap saved as temporal_heatmap.png")


if __name__ == "__main__":
    data_layer = DataCollectionLayer(
        "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    )
    raw_data = data_layer.load_data()
    preprocess_layer = DataPreprocessingLayer(raw_data)
    clean_data = preprocess_layer.preprocess_data()
    transformed_data = preprocess_layer.transform_data()

    clustering_model = ClusteringModel(n_clusters=3)
    clustered_data = clustering_model.fit(transformed_data)

    pattern_analysis = PatternAnalysis(clustered_data)
    pattern_analysis.analyze_clusters()
    pattern_analysis.visualize_clusters()
    pattern_analysis.plot_temporal_heatmap()
