from dataclasses import dataclass
import os
import torch


@dataclass
class Config:
    data_dir: str = os.path.join("src", "data", "corn")
    val_split: float = 0.2
    batch_size: int = 32
    num_epochs: int = 15
    lr: float = 1e-3
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"