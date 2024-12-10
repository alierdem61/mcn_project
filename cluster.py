from sklearn.cluster import KMeans
from collection import DataCollectionLayer
from preprocess import DataPreprocessingLayer


class ClusteringModel:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, data):
        # Fit the model on the coordinates data (lat, lon)
        self.model.fit(data[["lat", "lon"]])

        # Store the cluster assignments
        data["cluster"] = self.model.labels_

        return data

    def predict(self, new_data):
        # Predict the cluster for new data points
        return self.model.predict(new_data[["lat", "lon"]])


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
    print(clustered_data[clustered_data["cluster"] == 2].head())
