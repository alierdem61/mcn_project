# TODO: this layer is taking way too long to run.
# just to test if it works or not, reduce the size of the eval
# sample. for real tests, eval with entire data and leave it running overnight.


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
        sample_data = self.data.sample(
            frac=0.1, random_state=42
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
        inertia = self.clustering_model.model.inertia_
        print("Within-Cluster Sum of Squares (Inertia):", inertia)

        return {
            "silhouette_score": silhouette_avg,
            "davies_bouldin_index": db_index,
            "inertia": inertia,
        }

    def export_results(self, evaluation_results):
        # Save clustered data with cluster assignments to a CSV file
        self.data.to_csv("clustered_data.csv", index=False)
        print("Clustered data saved as clustered_data.csv")

        # Save evaluation metrics to a text file
        with open("evaluation_metrics.txt", "w") as f:
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

    clustering_model = ClusteringModel(n_clusters=3)
    clustered_data = clustering_model.fit(transformed_data)

    pattern_analysis = PatternAnalysis(clustered_data)
    pattern_analysis.analyze_clusters()
    pattern_analysis.visualize_clusters()
    pattern_analysis.plot_temporal_heatmap()

    evaluation_layer = EvaluationLayer(clustered_data, clustering_model)
    evaluation_results = evaluation_layer.evaluate_clustering()
    evaluation_layer.export_results(evaluation_results)
    # Optional: evaluation_layer.save_predictions(predictions) if you have predictions to save
