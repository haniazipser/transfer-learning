import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class KaggleDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

        # mapowanie labeli → int
        self.labels = sorted(self.data["label"].unique())
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

    def encoded_labels(self):
        return self.data["label"].map(self.label2idx).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.data_dir, row["image"])
        label = self.label2idx[row["label"]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label