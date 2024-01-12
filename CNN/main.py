import torch
from torch import nn
import logging
from utils import TransformData
from load_dataset import main
from train import Train
from model import TinyVGG, TinyVGGDynamic
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


class Main(TransformData, Train):
    def __init__(self, req: int = None, method=None):
        super(Train, self).__init__()
        super(TransformData, self).__init__()
        self.results: Dict = {}
        self.model = None
        self.BATCH_SIZE = None
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
        method = input("Press 1 to download, press 2 to use a local dataset: ")

        try:
            method = int(method)
            if method == 1:
                self.method = "download"
                url = input("Enter the dataset URL: ")
                logging.info("Creating the dataset")
                self.image_path = main(url=url, medium='download')

            elif method == 2:
                self.method = "local"
                directory = input("Enter the local directory path: ")
                self.image_path = main(dir_name=directory, medium='download')
                logging.info("Checking the dataset")

            else:
                raise ValueError("Invalid input. Please enter 1 or 2.")

        except ValueError:
            logging.error("Invalid input. Please enter a numeric value (1 or 2).")

    @staticmethod
    def initialize_model(
            input_shape: int,
            output_shape: int,
            hidden_units: int = 10,
            num_of_conv_blocks: int = 2) -> nn.Module:
        """
        Initialize the neural network model.

        Args:
            input_shape (int): Number of input channels.
            output_shape (int): Number of output classes.
            hidden_units (int): Number of hidden units in the model.
            num_of_conv_blocks (int): Number of convolutional blocks.

        Returns:
            nn.Module: Initialized neural network model.
        """
        print(input_shape, output_shape, hidden_units)
        if num_of_conv_blocks == 2:
            return TinyVGG(input_shape=input_shape,
                           hidden_units=hidden_units,
                           output_classes=output_shape)
        else:
            return TinyVGGDynamic(input_shape=input_shape,
                                  output_classes=output_shape,
                                  hidden_unit=hidden_units,
                                  num_conv_blocks=num_of_conv_blocks)

    def process(self, input_shape: Optional[int] = 3,
                hidden_units: Optional[int] = 10,
                epochs: int = 5,
                num_of_layers: int = 2
                ):
        """
        Process the chosen action (train or predict) based on user input.

        Args:
            input_shape (Optional[int]): Number of input channels (default is 3 for RGB).
            hidden_units (Optional[int]): Number of hidden units in the model (default is 10).
            epochs (int): Number of training epochs (default is 5).
            num_of_layers (int): Number of convolutional blocks in the model (default is 2).

        Returns:
            None
        """
        if self.req == 1:
            self.BATCH_SIZE = 32
            self.prepare_dataset()
            train_data, test_data = self.load_data(image_path=self.image_path)
            train_data_loader, test_data_loader = self.create_dataloaders(train_data=train_data,
                                                                          test_data=test_data,
                                                                          batch_size=self.BATCH_SIZE)

            # Initialize the model based on user-defined parameters
            self.model = self.initialize_model(input_shape=input_shape,
                                               hidden_units=hidden_units,
                                               output_shape=len(train_data.classes),
                                               num_of_conv_blocks=num_of_layers
                                               )
            # Train the model and store the results
            self.results = self.train(model=self.model,
                                      train_dataloader=train_data_loader,
                                      test_dataloader=test_data_loader,
                                      epochs=epochs,
                                      loss_fn=nn.CrossEntropyLoss(),
                                      optimizer=torch.optim.Adam(params=self.model.parameters(),
                                                                 lr=0.001))

            print(pd.DataFrame(self.results))
        else:
            logging.info("Predicting the image")
            logging.info("Choosing the best Model")


if __name__ == "__main__":
    logging.info("Train Your Own Model or Predict the image")
    logging.info("Press 1 to train your Own Model, press 2 to predict the Image ")
    request = input()

    try:
        request = int(request)
        main_instance = Main(req=request)
        main_instance.process(input_shape=3,
                              hidden_units=20,
                              num_of_layers=4,
                              epochs=10)

    except ValueError:
        logging.error("Invalid input. Please enter a numeric value (1 or 2).")