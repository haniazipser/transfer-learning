import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiprocessing import freeze_support
from src.config import Config
from src.data.datamodule import DataModule
from src.models.convnext import ConvNeXtTiny
from src.models.efficientnet import EfficientNetB0
from src.models.resnet import ResNet50
from src.training.trainer import Trainer

RESULTS_DIR = REPO_ROOT / "results"


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_results(results: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def is_done(results: dict, backbone_name: str, level: int) -> bool:
    return (
        backbone_name in results
        and str(level) in results[backbone_name]
        and results[backbone_name][str(level)].get("status") == "done"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID istniejącego runu do wznowienia (np. 2025-04-30_14-22-01)",
    )

    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Krótki opis runu (np. 'baseline' lub 'augmentation-v2')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.run_id:
        run_id = args.run_id
        results_path = RESULTS_DIR / f"{run_id}.json"
        if not results_path.exists():
            print(f"[ERROR] File not found: {results_path}")
            sys.exit(1)
        all_results = load_results(results_path)
        print(f"[RESUME] Resuming run: {run_id}")
    else:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = RESULTS_DIR / f"{run_id}.json"
        all_results = {
            "_meta": {
                "run_id": run_id,
                "note": args.note,
                "started_at": datetime.now().isoformat(),
            }
        }
        print(f"[NEW] New run: {run_id}")

    config = Config()
    BACKBONES = [ConvNeXtTiny, ResNet50]
    UNFREEZE_LEVELS = [2, 3, -1]

    data = DataModule(config)
    print(f"Dataset size:  {len(data.train_loader.dataset)}")
    print(f"Num classes:   {data.num_classes}")
    print(f"Train batches: {len(data.train_loader)}")
    print(f"Val batches:   {len(data.val_loader)}")

    for backbone_cls in BACKBONES:
        backbone_name = backbone_cls.__name__
        all_results.setdefault(backbone_name, {})

        for n in UNFREEZE_LEVELS:
            key = str(n)

            if args.run_id and is_done(all_results, backbone_name, n):
                print(f"[SKIP] {backbone_name} / unfreeze={n} — already done")
                continue

            print("\n" + "=" * 50)
            print(f"Backbone:       {backbone_name}")
            print(f"Unfreeze level: {n}")

            all_results[backbone_name][key] = {
                "status": "running",
                "run_id": run_id,
                "started_at": datetime.now().isoformat(),
            }
            save_results(all_results, results_path)

            try:
                backbone = backbone_cls(num_classes=data.num_classes)
                backbone.unfreeze_last_n_blocks(n)
                trainer = Trainer(backbone=backbone, data=data, config=config, unfreeze=n)
                history = trainer.fit()

                history_records = [
                    {
                        "epoch": i + 1,
                        "train_loss": history["train_loss"][i],
                        "val_loss": history["val_loss"][i],
                        "val_acc": history["val_acc"][i],
                    }
                    for i in range(len(history["val_acc"]))
                ]

                all_results[backbone_name][key] = {
                    "status": "done",
                    "run_id": run_id,
                    "started_at": all_results[backbone_name][key]["started_at"],
                    "finished_at": datetime.now().isoformat(),
                    "epochs_trained": len(history_records),
                    "early_stopped": len(history_records) < config.num_epochs,
                    "history": history_records,
                    "best_val_acc": max(history["val_acc"]),
                }

            except Exception as e:
                all_results[backbone_name][key] = {
                    "status": "failed",
                    "run_id": run_id,
                    "error": str(e),
                }
                print(f"[ERROR] {backbone_name} / unfreeze={n}: {e}")

            save_results(all_results, results_path)

    print("\nFINAL RESULTS:")
    for backbone_name, runs in all_results.items():
        if backbone_name == "_meta":
            continue
        print(f"\n{backbone_name}:")
        for level, entry in runs.items():
            status = entry.get("status", "?")
            if status == "done":
                print(f"  Unfreeze {level}: best val_acc = {entry['best_val_acc']:.4f}")
            else:
                print(f"  Unfreeze {level}: {status}")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    freeze_support()
    main()