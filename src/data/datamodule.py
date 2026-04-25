import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.config import Config
from src.data.KaggleDataset import KaggleDataset
from src.data.KaggleTestDataset import KaggleTestDataset

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class DataModule:
    def __init__(self, config: Config):
        full_train = KaggleDataset(
            csv_file=os.path.join(config.data_dir, "train.csv"),
            data_dir=config.data_dir,
            transform=TRANSFORM
        )

        val_size = int(len(full_train) * config.val_split)
        train_size = len(full_train) - val_size

        train_dataset, val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.num_classes = len(full_train.labels)

        test = KaggleTestDataset(
            csv_file=os.path.join(config.data_dir, "test.csv"),
            data_dir=config.data_dir,
            transform=TRANSFORM
        )

        self.test_loader = DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False
        )

