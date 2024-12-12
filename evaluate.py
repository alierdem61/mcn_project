from sklearn.metrics import silhouette_score, davies_bouldin_score
from collection import DataCollectionLayer
from preprocess import DataPreprocessingLayer
from cluster import ClusteringModel
from analysis import PatternAnalysis


class EvaluationLayer:
    def __init__(self, clustered_data, clustering_model):
        self.data = clustered_data
        self.clustering_model = clustering_model

    def evaluate_clustering(self):
        # Extract cluster labels and feature data
        sample_data = None
        if self.clustering_model.algorithm == "kmeans":
            sample_data = self.data.sample(
                frac=0.0001, random_state=42
            )  # Adjust sample size as needed
        else:
            sample_data = self.data.sample(
                frac=1, random_state=42
            )  # Adjust sample size as needed
        labels = sample_data["cluster"]
        features = sample_data[["lat", "lon"]]

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(features, labels, n_jobs=-1)
        print("Silhouette Score:", silhouette_avg)

        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(features, labels)
        print("Davies-Bouldin Index:", db_index)

        # Calculate Within-Cluster Sum of Squares (Inertia)
        inertia = None
        if clustering_model.algorithm == "kmeans":
            inertia = self.clustering_model.model.inertia_
            print("Within-Cluster Sum of Squares (Inertia):", inertia)

        return {
            "silhouette_score": silhouette_avg,
            "davies_bouldin_index": db_index,
            "inertia": inertia,
        }

    def export_results(self, evaluation_results):
        # Save clustered data with cluster assignments to a CSV file
        # self.data.to_csv("clustered_data.csv", index=False)
        # print("Clustered data saved as clustered_data.csv")

        # Save evaluation metrics to a text file
        with open(
            f"{self.clustering_model.algorithm}_evaluation_metrics.txt", "w"
        ) as f:
            for metric, value in evaluation_results.items():
                f.write(f"{metric}: {value}\n")
        print("Evaluation metrics saved as evaluation_metrics.txt")


# Usage example
if __name__ == "__main__":
    data_layer = DataCollectionLayer(
        "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    )
    raw_data = data_layer.load_data()
    preprocess_layer = DataPreprocessingLayer(raw_data)
    clean_data = preprocess_layer.preprocess_data()
    transformed_data = preprocess_layer.transform_data()
    train_data = transformed_data.sample(frac=0.8, random_state=42)  # 80% for training
    test_data = transformed_data.drop(train_data.index)  # 20% for testing
    clustered_data = None
    clustering_model = None

    try:
        clustering_model = ClusteringModel(algorithm="kmeans", n_clusters=3)
        clustered_data = clustering_model.fit(train_data)
        # kmeans_predictions = clustering_model.predict(test_data)
        # print("KMeans Predictions:", kmeans_predictions[:5])
    except Exception as e:
        print(f"KMeans Error: {e}")

    # try:
    #     clustering_model = ClusteringModel(algorithm="agglomerative", n_clusters=3)
    #     train_data_agglomerative = train_data.sample(
    #         n=10000, random_state=42
    #     )  # Reduce training size
    #     clustered_data = clustering_model.fit(train_data_agglomerative)
    #     test_data_agglomerative = test_data.sample(
    #         frac=0.01, random_state=42
    #     )  # Reduce test size
    #     agglomerative_predictions = clustering_model.predict(test_data_agglomerative)
    #     print("Agglomerative Predictions:", agglomerative_predictions[:5])
    # except Exception as e:
    #     print(f"Agglomerative Error: {e}")

    # try:
    #     clustering_model = ClusteringModel(algorithm="dbscan", eps=0.05, min_samples=5)
    #     clustered_data = clustering_model.fit(train_data, sample_fraction=0.001)
    #     test_data_dbscan = test_data.sample(
    #         frac=0.01, random_state=42
    #     )  # Reduce test size
    #     dbscan_predictions = clustering_model.predict(test_data_dbscan)
    #     print("DBSCAN Predictions:", dbscan_predictions[:5])
    # except Exception as e:
    #     print(f"DBSCAN Error: {e}")

    pattern_analysis = PatternAnalysis(clustered_data)
    pattern_analysis.analyze_clusters()
    pattern_analysis.visualize_clusters(clustering_model.algorithm)
    pattern_analysis.plot_temporal_heatmap(clustering_model.algorithm)

    evaluation_layer = EvaluationLayer(clustered_data, clustering_model)
    evaluation_results = evaluation_layer.evaluate_clustering()
    evaluation_layer.export_results(evaluation_results)
    # Optional: evaluation_layer.save_predictions(predictions) if you have predictions to save
