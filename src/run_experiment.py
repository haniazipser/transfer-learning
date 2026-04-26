import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.data.datamodule import DataModule
from src.models.resnet import ResNet50
from src.training.trainer import Trainer

# ── config ────────────────────────────────────────────────────────────────────
config = Config()

BACKBONE = ResNet50
UNFREEZE_LEVELS = [0, 1, 2, 3, -1]

# ── data ──────────────────────────────────────────────────────────────────────
data = DataModule(config)

print(f"Dataset size: {len(data.train_loader.dataset)}")
print(f"Num classes: {data.num_classes}")
print(f"Train batches: {len(data.train_loader)}")
print(f"Val batches: {len(data.val_loader)}")

# ── experiments loop ──────────────────────────────────────────────────────────
all_results = {}

for n in UNFREEZE_LEVELS:
    print("\n" + "="*50)
    print(f"Backbone: {BACKBONE.__name__}")
    print(f"Unfreeze level: {n}")

    backbone = BACKBONE(num_classes=data.num_classes)
    backbone.unfreeze_last_n_blocks(n)

    trainer = Trainer(
        backbone=backbone,
        data=data,
        config=config,
    )

    history = trainer.fit()
    all_results[n] = history

# ── summary ───────────────────────────────────────────────────────────────────
print("\nFINAL RESULTS:")
for k, v in all_results.items():
    best_acc = max(v["val_acc"])
    print(f"Unfreeze {k}: best val_acc = {best_acc:.4f}")

    results_path = REPO_ROOT / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Saved results to", results_path)