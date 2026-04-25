import torch

from src.config import Config
from src.data.datamodule import DataModule
from src.models.resnet import ResNet50
from src.training.trainer import Trainer

config = Config()

BACKBONE   = ResNet50        # ResNet50 | EfficientNetB0 | ConvNeXtTiny
N_UNFREEZE = 0               # 0=head only, 1,2,3=last blocs, -1=entire model

# ── composition root ──────────────────────────────────────────────────────────
data     = DataModule(config)

print(f"Dataset size: {len(data.train_loader.dataset)}")
print(f"Num classes: {data.num_classes}")
print(f"Train batches: {len(data.train_loader)}")
print(f"Val batches: {len(data.test_loader)}")

backbone = BACKBONE(num_classes=data.num_classes)
backbone.unfreeze_last_n_blocks(N_UNFREEZE)

print(f"Backbone: {BACKBONE.__name__}")
print(f"Unfreeze level: {N_UNFREEZE}")


print("torch:", torch.__version__)
print("cuda built:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

trainer  = Trainer(
    backbone          = backbone,
    data              = data,
    config            = config,
)

history = trainer.fit()