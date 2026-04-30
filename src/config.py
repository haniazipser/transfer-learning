from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = REPO_ROOT / "src" / "data" / "corn"
    data_dir: str = str(DEFAULT_DATA_DIR)
    val_split: float = 0.2
    batch_size: int = 64
    num_epochs: int = 40
    head_lr: float = 1e-4
    backbone_lr: float = 1e-6
    l2: float = 1e-3
    seed: int = 42
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
