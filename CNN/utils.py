import os
from pathlib import Path
from typing import Optional, Tuple
import torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TransformData:
    def __init__(self):
        self.test_dir = None
        self.train_dir = None


    @staticmethod
    def data_transform(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Create data transformation pipelines for training and testing.

        Args:
            augment (bool): Whether to apply data augmentation for training.

        Returns:
            Tuple[transforms.Compose, transforms.Compose]: Train and test transformation pipelines.
        """
        train_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31) if augment else transforms.RandomResizedCrop(64),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

        return train_transform, test_transform

    def load_data(self,
                  image_path: str | Path,
                  transform: bool = True,
                  augment: bool = True) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
        """
        Load training and testing data.

        Args:
            image_path (Path): Path to the root directory containing 'train' and 'test' subdirectories.
            transform (bool): Whether to apply data transformation.
            augment (bool): Whether to apply data augmentation for training.

        Returns:
            Tuple[datasets.ImageFolder, datasets.ImageFolder]: Train and test datasets.
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        self.train_dir = image_path / "train"
        self.test_dir = image_path / "test"

        if not (self.train_dir.exists() and self.test_dir.exists()):
            raise FileNotFoundError("Train or test directory not found.")

        train_transform, test_transform = self.data_transform(augment=augment)

        train_data = datasets.ImageFolder(root=self.train_dir,
                                          transform=train_transform,
                                          target_transform=None)

        test_data = datasets.ImageFolder(root=self.test_dir,
                                         transform=test_transform,
                                         target_transform=None)

        return train_data, test_data

    @staticmethod
    def create_dataloaders(train_data: torchvision.datasets.ImageFolder,
                           test_data: torchvision.datasets.ImageFolder,
                           batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = os.cpu_count()) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and testing data loaders.

        Args:
            train_data (datasets.ImageFolder): Training dataset.
            test_data (datasets.ImageFolder): Testing dataset.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loaders.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and testing data loaders.
        """
        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        return train_dataloader, test_dataloader
