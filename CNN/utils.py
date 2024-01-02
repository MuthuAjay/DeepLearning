import os

from pathlib import Path
from typing import Optional, Tuple

import torchvision.datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TransformData:

    def __init__(self):
        self.image_path = None
        self.train_dir = None
        self.test_dir = None

    @staticmethod
    def data_transform(augment: bool) -> Optional[Tuple]:
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.TrivialAugmentWide(num_magnitude_bins=31),
                transforms.ToTensor()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor()
            ])
        test_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

        return train_transform, test_transform

    def load_data(self,
                  image_path: str|Path,
                  transform: bool = True,
                  augment: bool = True):

        self.train_dir = image_path / "train"
        self.test_dir = image_path / "test"
        train_transform, test_transform = TransformData.data_transform(augment=augment)
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
                           BATCH_SIZE: Optional[int] = 32,
                           NUM_WORKERS: Optional[int] = os.cpu_count()
                           ):

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS)

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)

        return train_dataloader, test_dataloader
