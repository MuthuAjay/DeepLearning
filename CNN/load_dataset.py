import logging
import requests
import zipfile
from pathlib import Path

default_url = ''


class CustomData:
    def __init__(self, data_path: str):
        """
        Initialize the CustomData object.

        Args:
            data_path (str): Path to the root directory where the dataset will be stored.
        """
        self.data_path = Path(data_path)
        self.image_path = None

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")

    @staticmethod
    def remove_directory(path: Path):
        """
        Remove a directory if it exists.

        Args:
            path (Path): Path to the directory to be removed.
        """
        try:
            path.rmdir()
            logging.info(f"Directory '{path}' removed successfully.")
        except OSError as e:
            logging.debug(f"Error removing directory '{path}': {e}")

    def download_data(self, directory: str, download_url: str):
        """
        Download and extract the pizza_steak_sushi dataset.

        Args:
            directory (str): Name of the subdirectory to create for the dataset.
            download_url (str): URL to download the dataset zip file.
        """
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
    def local_drive(path: Path) -> bool:
        """
        Check if the dataset is present on the local drive.

        Args:
            path (Path): Path to the root directory of the dataset.

        Returns:
            bool: True if the dataset is present, False otherwise.
        """
        expected_folders = ['train', 'test']
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


def main(medium: str,
         url: str = default_url,
         dir_name: str | Path = "pizza_steak_sushi") -> Path:
    """
    Main function for handling dataset download and local checks.

    Args:
        url (str): URL to download the dataset zip file.
        medium (str): Download medium (either 'download' or 'local').
        dir_name (str | Path, optional): Name of the subdirectory to create for the dataset.
            Defaults to "pizza_steak_sushi".

    Returns:
        Path: Path to the dataset directory.

    Raises:
        FileNotFoundError: If the dataset is not present locally.
    """
    logging.basicConfig(level=logging.INFO)

    custom_dataset = CustomData(data_path=r"C:\Users\CD138JR\PycharmProjects\DeepLearning\CNN\data")

    if medium == 'download':
        custom_dataset.download_data(directory=dir_name, download_url=url)
        return custom_dataset.image_path
    else:
        path = custom_dataset.data_path / dir_name
        present = custom_dataset.local_drive(path=path)

        if present:
            logging.info("Dataset is present locally.")
            return path
        else:
            logging.warning("Dataset is not present locally.")
            raise FileNotFoundError

