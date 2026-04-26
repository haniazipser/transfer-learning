from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ..config import Config
from .KaggleDataset import KaggleDataset
from .KaggleTestDataset import KaggleTestDataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class DataModule:
    def __init__(self, config: Config):
        data_dir = Path(config.data_dir)
        train_csv = data_dir / "train.csv"
        test_csv = data_dir / "test.csv"

        full_train = KaggleDataset(
            csv_file=str(train_csv),
            data_dir=str(data_dir),
            transform=TRANSFORM,
        )

        val_size = int(len(full_train) * config.val_split)
        train_size = len(full_train) - val_size

        train_dataset, val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

        self.num_classes = len(full_train.labels)

        test = KaggleTestDataset(
            csv_file=str(test_csv),
            data_dir=str(data_dir),
            transform=TRANSFORM,
        )

        self.test_loader = DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False,
        )

