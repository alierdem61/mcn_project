import pandas as pd
import numpy as np


def process_cluster_sequences(file_path):
    """
    Process the clustered data to generate sorted cluster sequences for each user.

    Args:
        file_path (str): Path to the clustered data CSV file.

    Returns:
        dict: A dictionary with user IDs as keys (or a single sequence for datasets without user IDs).
    """
    # Load the clustered data
    data = pd.read_csv(file_path)

    # Ensure the timestamp column is in datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort data by timestamp
    data = data.sort_values(by="timestamp")

    # Check if user_id exists; group by user if present
    if "user_id" in data.columns:
        cluster_sequences = data.groupby("user_id")["cluster"].apply(list).to_dict()
    else:
        # If no user_id, treat the dataset as a single sequence
        cluster_sequences = {"all_users": data["cluster"].tolist()}

    return cluster_sequences


def split_sequence(sequence, test_size=0.2):
    """
    Split a single sequence into training and testing parts.

    Args:
        sequence (list): The full sequence of clusters.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: Training and testing subsequences.
    """
    split_index = int(len(sequence) * (1 - test_size))
    train_sequence = sequence[:split_index]
    test_sequence = sequence[split_index:]
    return train_sequence, test_sequence


def build_transition_matrix(cluster_sequences, n_clusters):
    """
    Build a transition probability matrix from cluster sequences.

    Args:
        cluster_sequences (dict): Dictionary of cluster sequences.
        n_clusters (int): Number of unique clusters.

    Returns:
        np.ndarray: Transition probability matrix.
    """
    # Initialize transition matrix
    transition_matrix = np.zeros((n_clusters, n_clusters))

    # Count transitions
    for sequence in cluster_sequences.values():
        for i in range(len(sequence) - 1):
            from_cluster = sequence[i]
            to_cluster = sequence[i + 1]
            transition_matrix[from_cluster, to_cluster] += 1

    # Normalize rows to convert counts to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_matrix,
        row_sums,
        out=np.zeros_like(
            transition_matrix
        ),  # Set default values for rows with zero sum
        where=row_sums != 0,
    )

    # Replace any NaN with uniform probabilities (for clusters with no outgoing transitions)
    nan_rows = np.isnan(transition_matrix).any(axis=1)
    transition_matrix[nan_rows] = 1 / n_clusters  # Uniform probability for all clusters

    return transition_matrix


def predict_next_cluster(current_cluster, transition_matrix):
    """
    Predict the next cluster based on the current cluster using the transition matrix.

    Args:
        current_cluster (int): The current cluster.
        transition_matrix (np.ndarray): The transition probability matrix.

    Returns:
        int: The predicted next cluster.
    """
    # Get probabilities for the current cluster
    probabilities = transition_matrix[current_cluster]

    # Predict the next cluster (choose the one with the highest probability)
    next_cluster = np.argmax(probabilities)
    return next_cluster


def evaluate_predictions(test_sequence, transition_matrix):
    """
    Evaluate prediction accuracy using the transition matrix.

    Args:
        test_sequence (list): Test sequence of clusters.
        transition_matrix (np.ndarray): The transition probability matrix.

    Returns:
        float: Prediction accuracy as a percentage.
    """
    total_predictions = 0
    correct_predictions = 0

    for i in range(len(test_sequence) - 1):
        current_cluster = test_sequence[i]
        actual_next_cluster = test_sequence[i + 1]
        predicted_next_cluster = predict_next_cluster(
            current_cluster, transition_matrix
        )

        total_predictions += 1
        if predicted_next_cluster == actual_next_cluster:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def save_predictions_to_file(algorithm_name, accuracy, file_path="predictions.txt"):
    """
    Save prediction results to a file.

    Args:
        algorithm_name (str): Name of the clustering algorithm used.
        accuracy (float): Prediction accuracy as a percentage.
        file_path (str): File path for saving predictions.
    """
    output_file = f"{algorithm_name}_{file_path}"
    with open(output_file, "w") as f:
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Prediction Accuracy: {accuracy:.2f}%\n")
    print(f"Prediction results saved to {output_file}")


if __name__ == "__main__":
    algorithm_name = "agglomerative"

    # Example usage
    file_path = f"/home/alierdem/mcn_pjkt/data/{algorithm_name}_clustered_data.csv"  # Replace with the actual file path
    cluster_sequences = process_cluster_sequences(file_path)

    # Output example sequences
    for user_id, sequence in cluster_sequences.items():
        print(
            f"User ID: {user_id}, Sequence: {sequence[:10]}"
        )  # First 10 clusters for each user

    # Handle single sequence case
    all_users_sequence = cluster_sequences["all_users"]
    train_sequence, test_sequence = split_sequence(all_users_sequence, test_size=0.2)
    print(
        f"\nTraining Sequence Length: {len(train_sequence)}, Testing Sequence Length: {len(test_sequence)}"
    )

    # Get the number of unique clusters
    n_clusters = len(set(all_users_sequence))

    # Build transition matrix using training data
    train_sequences = {"all_users": train_sequence}
    transition_matrix = build_transition_matrix(train_sequences, n_clusters)
    print("\nTransition Probability Matrix:")
    print(transition_matrix)

    # Example prediction
    current_cluster = 1  # Replace with an actual cluster from your data
    predicted_cluster = predict_next_cluster(current_cluster, transition_matrix)
    print(
        f"\nPredicted next cluster for current cluster {current_cluster}: {predicted_cluster}"
    )

    # Evaluate predictions on test data
    accuracy = evaluate_predictions(test_sequence, transition_matrix)
    print(f"\nPrediction Accuracy: {accuracy:.2f}%")

    # Save predictions to a file
    save_predictions_to_file(algorithm_name, accuracy)
