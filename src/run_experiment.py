import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.data.datamodule import DataModule
from src.models.convnext import ConvNeXtTiny
from src.models.efficientnet import EfficientNetB0
from src.models.resnet import ResNet50
from src.training.trainer import Trainer
from multiprocessing import freeze_support

def main():
    # ── config ────────────────────────────────────────────────────────────────────
    config = Config()

    BACKBONES = [ConvNeXtTiny, ResNet50, EfficientNetB0]
    UNFREEZE_LEVELS = [0, 1, 2, 3, -1]

    # ── data ──────────────────────────────────────────────────────────────────────
    data = DataModule(config)

    print(f"Dataset size: {len(data.train_loader.dataset)}")
    print(f"Num classes: {data.num_classes}")
    print(f"Train batches: {len(data.train_loader)}")
    print(f"Val batches: {len(data.val_loader)}")

    # ── experiments loop ──────────────────────────────────────────────────────────
    all_results = {}

    for backbone_cls in BACKBONES:
        all_results[backbone_cls.__name__] = {}

        for n in UNFREEZE_LEVELS:
            print("\n" + "="*50)
            print(f"Backbone: {backbone_cls.__name__}")
            print(f"Unfreeze level: {n}")

            backbone = backbone_cls(num_classes=data.num_classes)
            backbone.unfreeze_last_n_blocks(n)

            trainer = Trainer(
                backbone=backbone,
                data=data,
                config=config,
            )

            history = trainer.fit()
            all_results[backbone_cls.__name__][n] = history

    # ── summary ───────────────────────────────────────────────────────────────────
    print("\nFINAL RESULTS:")
    for backbone_name, runs in all_results.items():
        print(f"\n{backbone_name}:")
        for level, history in runs.items():
            best_acc = max(history["val_acc"])
            print(f"  Unfreeze {level}: best val_acc = {best_acc:.4f}")

    results_path = REPO_ROOT / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Saved results to", results_path)




if __name__ == "__main__":
    freeze_support()
    main()