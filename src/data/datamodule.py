from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ..config import Config
from .KaggleDataset import KaggleDataset
from .KaggleTestDataset import KaggleTestDataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])


class DataModule:
    def __init__(self, config: Config):
        data_dir = Path(config.data_dir)
        train_csv = data_dir / "train.csv"
        test_csv = data_dir / "test.csv"

        full_train_aug = KaggleDataset(
            csv_file=str(train_csv),
            data_dir=str(data_dir),
            transform=TRAIN_TRANSFORM,
        )

        full_train_eval = KaggleDataset(
            csv_file=str(train_csv),
            data_dir=str(data_dir),
            transform=EVAL_TRANSFORM,
        )

        val_size = int(len(full_train_aug) * config.val_split)
        train_size = len(full_train_aug) - val_size

        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(full_train_aug), generator=generator).tolist()

        train_dataset = torch.utils.data.Subset(full_train_aug, indices[:train_size])
        val_dataset = torch.utils.data.Subset(full_train_eval, indices[train_size:])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.num_classes = len(full_train_aug.labels)
        self.train_labels = full_train_aug.encoded_labels()

        test = KaggleTestDataset(
            csv_file=str(test_csv),
            data_dir=str(data_dir),
            transform=EVAL_TRANSFORM,
        )

        self.test_loader = DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False,
        )
