import numpy as np
from collection import DataCollectionLayer


class DataPreprocessingLayer:

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def preprocess_data(self):
        data = self.raw_data[(self.raw_data["lat"] != 0) & (self.raw_data["lon"] != 0)]
        return data

    def calculate_speed_vectorized(self):
        # Shift lat, lon, and timestamp columns to calculate deltas between consecutive points
        data = self.raw_data
        lat1 = np.radians(data["lat"].values[:-1])
        lon1 = np.radians(data["lon"].values[:-1])
        lat2 = np.radians(data["lat"].values[1:])
        lon2 = np.radians(data["lon"].values[1:])

        # Haversine formula vectorized
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        distances = R * c  # Distance between consecutive points in km

        # Time delta in hours
        time_deltas = (
            data["timestamp"].values[1:] - data["timestamp"].values[:-1]
        ) / np.timedelta64(1, "h")

        # Calculate speed (km/h) and handle division by zero
        speeds = np.divide(
            distances, time_deltas, out=np.zeros_like(distances), where=time_deltas != 0
        )

        # Insert 0 speed for the first point (no previous point to calculate from)
        speeds = np.insert(speeds, 0, 0)

        # Add speeds to the data
        data["speed"] = speeds
        return data

    def transform_data(self):
        return self.calculate_speed_vectorized()


if __name__ == "__main__":
    data_directory = "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    collection_layer = DataCollectionLayer(data_directory)
    raw_data = collection_layer.load_data()
    preprocess_layer = DataPreprocessingLayer(raw_data)

    cleaned_data = preprocess_layer.preprocess_data()
    transformed_data = preprocess_layer.transform_data()
    print(transformed_data.head())
