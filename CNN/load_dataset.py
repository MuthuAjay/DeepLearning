import logging
import requests
import zipfile
from pathlib import Path
import argparse

class CustomData:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    @staticmethod
    def remove_directory(path):
        path = Path(path)
        try:
            path.rmdir()
            logging.info(f"Directory '{path}' removed successfully.")
        except OSError as e:
            logging.debug(f"Error removing directory '{path}': {e}")

    def download_data(self, directory, download_url):
        image_path = self.data_path / directory

        if image_path.is_dir():
            if not CustomData.local_drive(image_path):
                CustomData.remove_directory(image_path)
                self.download_data(directory, download_url)
            logging.info(f"{image_path} directory exists.")
        else:
            logging.info(f"Did not find {image_path} directory, creating one...")
            image_path.mkdir(parents=True, exist_ok=True)

            # Download pizza, steak, sushi data
            zip_file_path = self.data_path / "pizza_steak_sushi.zip"
            with open(zip_file_path, "wb") as f:
                request = requests.get(download_url, verify=False)
                if request.status_code == 200:
                    logging.info("Downloading pizza, steak, sushi data...")
                    f.write(request.content)
                else:
                    logging.error(f"Failed to download data. Status code: {request.status_code}")
                    return

            # Unzip pizza, steak, sushi data
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                logging.info("Unzipping pizza, steak, sushi data...")
                zip_ref.extractall(image_path)

            logging.info("Data download and extraction completed.")

    @staticmethod
    def local_drive(path):
        expected_folders = ['train', 'test']
        if isinstance(path, str):
            path = Path(path)

        if not all((path / folder).exists() for folder in expected_folders):
            return False

        for folder in expected_folders:
            sub_folders_path = path / folder
            sub_folders = [sub_folder for sub_folder in sub_folders_path.iterdir() if sub_folder.is_dir()]

            if not all(sub_folder.is_dir() for sub_folder in sub_folders):
                return False

        return True

    def __repr__(self):
        return f"CustomData(data_path={self.data_path})"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser(prog="Load Custom Dataset")
    arg_parser.add_argument('--path', required=False, default="pizza_steak_sushi")
    arg_parser.add_argument('--url',
                            required=False,
                            default="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data"
                                    "/pizza_steak_sushi.zip")
    args = arg_parser.parse_args()

    custom_dataset = CustomData(data_path=r"C:\Users\CD138JR\PycharmProjects\DeepLearning\CNN\data")
    custom_dataset.download_data(directory=args.path, download_url=args.url)
    print(custom_dataset)
