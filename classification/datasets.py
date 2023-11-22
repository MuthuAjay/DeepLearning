from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
import torch
from utils import *


class Datasets:

    def __init__(self, test_size: float, random_state: int = 42) -> None:
        self.TESTSIZE = test_size
        self.RANDOMSTATE = random_state
        pass

    def create_train_test_split(self, X: torch.Tensor,
                                y: torch.Tensor) -> tuple:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.TESTSIZE,
                                                            random_state=self.RANDOMSTATE)

        return X_train, X_test, y_train, y_test

    def circle_dataset(self, n_samples: int = 1000,
                       noise: float = 0.02,
                       random_state: int = 42) -> tuple:
        X, y = make_circles(n_samples=n_samples,
                            noise=noise,
                            random_state=random_state)
        X, y = torch.from_numpy(X).type(torch.float).to(Utils.set_device()), torch.from_numpy(y).type(torch.float).to(
            Utils.set_device())
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)

        return X_train, X_test, y_train, y_test

    def blob_dataset(self,
                     num_features: int,
                     num_classes: int,
                     cluster_std: int = 1,
                     n_samples: int = 1000,
                     random_state: int = 42) -> torch.tensor:
        X, y = make_blobs(n_samples=n_samples,
                          n_features=num_features,
                          cluster_std=cluster_std,
                          centers=num_classes,
                          random_state=random_state
                          )

        X, y = torch.from_numpy(X).type(torch.float).to(Utils.set_device()), torch.from_numpy(y).type(
            torch.LongTensor).to(Utils.set_device())
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)

        return X_train, X_test, y_train, y_test
