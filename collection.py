import pandas as pd
import os


class DataCollectionLayer:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_data(self):
        trajectories = []
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".plt"):
                    file_path = os.path.join(root, file)
                    trajectory = self.parse_file(file_path)
                    if trajectory is not None:
                        trajectories.append(trajectory)

        return pd.concat(trajectories, ignore_index=True)

    def parse_file(self, file_path):
        try:
            data = pd.read_csv(
                file_path,
                skiprows=6,
                names=["lat", "lon", "zero", "alt", "days", "date", "time"],
            )
            data["timestamp"] = pd.to_datetime(
                data["date"] + " " + data["time"], format="%Y-%m-%d %H:%M:%S"
            )
            data = data[["lat", "lon", "alt", "timestamp"]]
            return data
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
            return None


if __name__ == "__main__":
    data_directory = "/home/alierdem/mcn_pjkt/data/Geolife Trajectories 1.3/Data"
    data_layer = DataCollectionLayer(data_directory)
    raw_data = data_layer.load_data()
    # cleaned_data = data_layer.preprocess_data(raw_data)
    # transformed_data = data_layer.transform_data(cleaned_data)
    print(raw_data.head())
