import sys
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score


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

        # Check for multiple clusters
        if len(set(labels)) > 1:
            # Calculate Silhouette Score
            silhouette_avg = silhouette_score(features, labels)
            print(f"{self.algorithm.capitalize()} - Silhouette Score:", silhouette_avg)

            # Calculate Davies-Bouldin Index
            db_index = davies_bouldin_score(features, labels)
            print(f"{self.algorithm.capitalize()} - Davies-Bouldin Index:", db_index)
        else:
            print(
                f"{self.algorithm.capitalize()} - Cannot calculate metrics with a single cluster."
            )
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
        print(
            f"{self.algorithm.capitalize()} evaluation metrics saved as {output_file}"
        )


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <algorithm(s)>")
        print("Available algorithms: kmeans, hdbscan, birch, agglomerative, all")
        sys.exit(1)

    algorithms_to_run = sys.argv[1:]
    available_algorithms = ["kmeans", "hdbscan", "birch", "agglomerative"]

    if "all" in algorithms_to_run:
        algorithms_to_run = available_algorithms

    for algorithm_used in algorithms_to_run:
        if algorithm_used not in available_algorithms:
            print(f"Unsupported algorithm: {algorithm_used}")
            continue

        # Set sample fractions based on the algorithm
        sample_fraction = 1.0
        if algorithm_used in ["kmeans", "birch"]:
            sample_fraction = 0.001
        elif algorithm_used == "hdbscan":
            sample_fraction = 0.01

        # Path to clustered data
        clustered_file_path = (
            f"/home/alierdem/mcn_pjkt/data/{algorithm_used}_clustered_data.csv"
        )

        # Load clustered data
        try:
            clustered_data = pd.read_csv(clustered_file_path)
        except FileNotFoundError:
            print(
                f"Clustered data file not found at {clustered_file_path}. Ensure clustering is complete."
            )
            continue

        # Initialize EvaluationLayer with sampling
        evaluation_layer = EvaluationLayer(
            clustered_data, algorithm_used, sample_fraction=sample_fraction
        )

        # Evaluate clustering
        evaluation_results = evaluation_layer.evaluate_clustering()

        # Export results
        evaluation_layer.export_results(evaluation_results)
