import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class EvaluationLayer:
    def __init__(self, clustered_data, algorithm, sample_fraction=0.1, random_state=42):
        """
        Initialize the evaluation layer.

        Args:
            clustered_data (DataFrame): DataFrame containing features and cluster labels.
            algorithm (str): Name of the clustering algorithm used.
            sample_fraction (float): Fraction of the data to sample for evaluation.
            random_state (int): Random state for reproducibility.
        """
        self.data = clustered_data
        self.algorithm = algorithm
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def evaluate_clustering(self):
        """
        Evaluate clustering performance using Silhouette Score and Davies-Bouldin Index.

        Returns:
            dict: Evaluation metrics.
        """
        # Sample a subset of the data
        sample_data = self.data.sample(
            frac=self.sample_fraction, random_state=self.random_state
        )
        features = sample_data[["lat", "lon", "speed"]]
        labels = sample_data["cluster"]

        # Scale features for evaluation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Check for multiple clusters
        if len(set(labels)) > 1:
            # Calculate Silhouette Score
            silhouette_avg = silhouette_score(features_scaled, labels)
            print("Silhouette Score:", silhouette_avg)

            # Calculate Davies-Bouldin Index
            db_index = davies_bouldin_score(features_scaled, labels)
            print("Davies-Bouldin Index:", db_index)
        else:
            print("Cannot calculate metrics with a single cluster.")
            silhouette_avg = None
            db_index = None

        return {
            "silhouette_score": silhouette_avg,
            "davies_bouldin_index": db_index,
        }

    def export_results(self, evaluation_results):
        """
        Export evaluation results to a text file.

        Args:
            evaluation_results (dict): Metrics to save.
        """
        output_file = f"{self.algorithm}_evaluation_metrics.txt"
        with open(output_file, "w") as f:
            for metric, value in evaluation_results.items():
                f.write(f"{metric}: {value}\n")
        print(f"Evaluation metrics saved as {output_file}")


if __name__ == "__main__":
    # Specify the algorithm used for clustering
    algorithm_used = "hdbscan"  # Update this value as needed

    # Path to clustered data
    clustered_file_path = (
        f"/home/alierdem/mcn_pjkt/data/{algorithm_used}_clustered_data.csv"
    )

    # Load clustered data
    try:
        clustered_data = pd.read_csv(clustered_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Clustered data file not found at {clustered_file_path}. Ensure clustering is complete."
        )

    # Initialize EvaluationLayer with sampling
    evaluation_layer = EvaluationLayer(
        clustered_data, algorithm_used, sample_fraction=0.001
    )

    # Evaluate clustering
    evaluation_results = evaluation_layer.evaluate_clustering()

    # Export results
    evaluation_layer.export_results(evaluation_results)
