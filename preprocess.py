import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collection import DataCollectionLayer
import pickle  # For saving normalization data


class DataPreprocessingLayer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.scalers = {}  # Store scalers for normalization

    def preprocess_data(self):
        """
        Cleans the raw data by removing invalid points and duplicates,
        and filters points within the geographical boundaries of China.
        Returns:
            Cleaned DataFrame.
        """
        # Remove invalid latitude and longitude values (lat=0, lon=0)
        data = self.raw_data[(self.raw_data["lat"] != 0) & (self.raw_data["lon"] != 0)]

        # Filter points within China's geographical boundaries
        data = data[
            (data["lat"] >= 20.14)
            & (data["lat"] <= 53.33)
            & (data["lon"] >= 73.5)
            & (data["lon"] <= 134.77)
        ]

        # Remove duplicate rows
        data = data.drop_duplicates()

        return data

    def calculate_speed_vectorized(self, data):
        """
        Calculates speed between consecutive points using the Haversine formula.
        Adds a 'speed' column to the data for each user_id.
        """

        def compute_speed(group):
            """
            Compute speed for a single user group using the Haversine formula.
            """
            # Drop user_id column temporarily
            group_data = group.drop(columns=["user_id"])

            # Shift lat, lon, and timestamp columns to calculate deltas
            lat1 = np.radians(group_data["lat"].values[:-1])
            lon1 = np.radians(group_data["lon"].values[:-1])
            lat2 = np.radians(group_data["lat"].values[1:])
            lon2 = np.radians(group_data["lon"].values[1:])

            # Haversine formula for distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            R = 6371  # Radius of Earth in kilometers
            distances = R * c  # Distance in kilometers

            # Time delta in hours
            time_deltas = (
                group_data["timestamp"].values[1:] - group_data["timestamp"].values[:-1]
            ) / np.timedelta64(1, "h")

            # Calculate speed (km/h), handling division by zero
            speeds = np.divide(
                distances,
                time_deltas,
                out=np.zeros_like(distances),
                where=time_deltas != 0,
            )

            # Insert 0 speed for the first point (no previous point to calculate from)
            speeds = np.insert(speeds, 0, 0)

            # Add speeds back to the group
            group["speed"] = speeds
            return group

        # Group data by user_id and calculate speed independently for each group
        data = data.groupby("user_id", group_keys=False).apply(compute_speed)
        return data

    def add_temporal_features(self, data):
        """
        Adds temporal features: hour of the day and day of the week.
        Args:
            data: DataFrame with 'timestamp' column.
        Returns:
            DataFrame with new columns 'hour' and 'day_of_week'.
        """
        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.day_name()
        return data

    def normalize_data(self, data):
        """
        Normalizes latitude, longitude, and speed for clustering.
        Args:
            data: DataFrame with 'lat', 'lon', and 'speed' columns.
        Returns:
            Normalized DataFrame.
        """
        # Initialize a single scaler for latitude, longitude, and speed
        self.scalers["features"] = StandardScaler()

        # Scale lat, lon, and speed together
        data[["lat", "lon", "speed"]] = self.scalers["features"].fit_transform(
            data[["lat", "lon", "speed"]]
        )

        return data

    def save_normalization_data(self, file_path):
        """
        Saves normalization scalers to a file for future use.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.scalers["features"], f)  # Save combined scaler
        print(f"Normalization data saved to {file_path}")

    def transform_data(self):
        """
        Combines all preprocessing steps to clean, transform, and normalize the data.
        Returns:
            Fully preprocessed DataFrame.
        """
        cleaned_data = self.preprocess_data()
        data_with_speed = self.calculate_speed_vectorized(cleaned_data)
        data_with_temporal = self.add_temporal_features(data_with_speed)
        normalized_data = self.normalize_data(data_with_temporal)
        return normalized_data

    @staticmethod
    def save_preprocessed_data(data, file_path):
        """
        Saves the preprocessed data to a file in CSV format.
        """
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")


if __name__ == "__main__":
    data_directory = "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    preprocessed_file_path = "/home/alierdem/mcn_pjkt/data/preprocessed_data_china.csv"
    normalization_file_path = "/home/alierdem/mcn_pjkt/data/normalization_data.pkl"

    # Preprocessing needed
    collection_layer = DataCollectionLayer(data_directory)
    raw_data = collection_layer.load_data()

    preprocess_layer = DataPreprocessingLayer(raw_data)
    transformed_data = preprocess_layer.transform_data()

    # Save the preprocessed data for future use
    DataPreprocessingLayer.save_preprocessed_data(
        transformed_data, preprocessed_file_path
    )
    preprocess_layer.save_normalization_data(normalization_file_path)

    # Print the head of the preprocessed data
    print(transformed_data.head())
