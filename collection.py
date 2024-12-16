import pandas as pd
import os


class DataCollectionLayer:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_data(self):
        """
        Load GPS trajectory data from the dataset directory.
        Includes user ID as a new column based on the folder structure.

        Returns:
            DataFrame: Combined data from all trajectories with user ID.
        """
        trajectories = []
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".plt"):
                    user_id = self.get_user_id(root)
                    file_path = os.path.join(root, file)
                    trajectory = self.parse_file(file_path, user_id)
                    if trajectory is not None:
                        trajectories.append(trajectory)

        return pd.concat(trajectories, ignore_index=True)

    def get_user_id(self, path):
        """
        Extract the user ID from the directory structure.

        Args:
            path (str): The path of the current file.

        Returns:
            str: The user ID derived from the directory name.
        """
        return os.path.basename(os.path.dirname(path))

    def parse_file(self, file_path, user_id):
        """
        Parse a single .plt file and add user ID.

        Args:
            file_path (str): Path to the .plt file.
            user_id (str): The user ID extracted from the directory.

        Returns:
            DataFrame: Parsed data with latitude, longitude, altitude, timestamp, and user ID.
        """
        try:
            data = pd.read_csv(
                file_path,
                skiprows=6,
                names=["lat", "lon", "zero", "alt", "days", "date", "time"],
            )
            # Combine date and time into a single timestamp column
            data["timestamp"] = pd.to_datetime(
                data["date"] + " " + data["time"], format="%Y-%m-%d %H:%M:%S"
            )
            # Add user ID as a new column
            data["user_id"] = user_id
            # Keep relevant columns
            data = data[["lat", "lon", "alt", "timestamp", "user_id"]]
            return data
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Define the path to the dataset directory
    data_directory = "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    # Initialize the DataCollectionLayer
    data_layer = DataCollectionLayer(data_directory)
    # Load all trajectory data with user IDs
    raw_data = data_layer.load_data()
    # Display the first few rows of the collected data
    print(raw_data.head())
