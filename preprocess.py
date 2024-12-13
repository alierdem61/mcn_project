import os
import pandas as pd
import numpy as np
from collection import DataCollectionLayer


class DataPreprocessingLayer:

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def preprocess_data(self):
        """
        Removes invalid GPS points (latitude and longitude equal to 0).
        """
        data = self.raw_data[(self.raw_data["lat"] != 0) & (self.raw_data["lon"] != 0)]
        return data

    def calculate_speed_vectorized(self, data):
        """
        Calculates speed between consecutive points using the Haversine formula.
        Adds a 'speed' column to the data.
        """
        # Shift lat, lon, and timestamp columns to calculate deltas
        lat1 = np.radians(data["lat"].values[:-1])
        lon1 = np.radians(data["lon"].values[:-1])
        lat2 = np.radians(data["lat"].values[1:])
        lon2 = np.radians(data["lon"].values[1:])

        # Haversine formula for distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        distances = R * c  # Distance in kilometers

        # Time delta in hours
        time_deltas = (
            data["timestamp"].values[1:] - data["timestamp"].values[:-1]
        ) / np.timedelta64(1, "h")

        # Calculate speed (km/h), handling division by zero
        speeds = np.divide(
            distances, time_deltas, out=np.zeros_like(distances), where=time_deltas != 0
        )

        # Insert 0 speed for the first point (no previous point to calculate from)
        speeds = np.insert(speeds, 0, 0)

        # Add speeds to the data
        data["speed"] = speeds
        return data

    def transform_data(self):
        """
        Combines preprocessing and speed calculation.
        """
        cleaned_data = self.preprocess_data()
        transformed_data = self.calculate_speed_vectorized(cleaned_data)
        return transformed_data

    @staticmethod
    def save_preprocessed_data(data, file_path):
        """
        Saves the preprocessed data to a file in CSV format.
        """
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")

    @staticmethod
    def load_preprocessed_data(file_path):
        """
        Loads preprocessed data from a file if it exists.
        """
        if os.path.exists(file_path):
            print(f"Loading preprocessed data from {file_path}")
            return pd.read_csv(file_path, parse_dates=["timestamp"])
        return None


if __name__ == "__main__":
    data_directory = "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    preprocessed_file_path = "/home/alierdem/mcn_pjkt/data/preprocessed_data.csv"

    # Attempt to load preprocessed data
    preprocessed_data = DataPreprocessingLayer.load_preprocessed_data(
        preprocessed_file_path
    )

    if preprocessed_data is None:
        # Preprocessing needed
        collection_layer = DataCollectionLayer(data_directory)
        raw_data = collection_layer.load_data()

        preprocess_layer = DataPreprocessingLayer(raw_data)
        transformed_data = preprocess_layer.transform_data()

        # Save the preprocessed data for future use
        DataPreprocessingLayer.save_preprocessed_data(
            transformed_data, preprocessed_file_path
        )
        preprocessed_data = transformed_data

    # Print the head of the preprocessed data
    print(preprocessed_data.head())
