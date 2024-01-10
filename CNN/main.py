import torch
from torch import nn
import logging
import subprocess
from utils import TransformData
from load_dataset import main
from pathlib import Path


class Main(TransformData):
    def __init__(self, req: int = None, method=None):
        super().__init__()
        self.image_path: str | Path = ''
        self.req = req
        self.method = method

    def prepare_dataset(self):
        """
        Prepare the dataset based on user input (download or use local).

        Returns:
            None
        """
        logging.info("Give URL or local path of the dataset")
        method = int(input("Press 1 to download, press 2 to use a local dataset: "))

        if method == 1:
            self.method = "download"
            url = str(input())
            logging.info("Creating the dataset")
            self.image_path = main(url=url,
                                   method='download')

        else:
            self.method = "local"
            directory = str(input())
            self.image_path = main(dir_name=directory,
                                   method='download')
            logging.info("Checking the dataset")

    def process(self):
        if self.req == 1:
            self.prepare_dataset()
            self.load_data(image_path= self.image_path)
        else:
            logging.info("Predicting the image")
            logging.info("Choosing the best Model")


if __name__ == "__main__":
    logging.info("Train Your Own Model or Predict the image")
    logging.info("Press 1 to train your Own Model, press 2 to predict the Image ")
    request = int(input())
    main = Main(req=request)
    main.process()
