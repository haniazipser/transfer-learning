import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

UNFREEZE_LEVELS = [0, 1, 2, 3, -1]


def load_results(run_id: str) -> dict:
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_backbone(backbone_name: str, runs: dict, run_id: str, note: str | None):
    levels = [str(l) for l in UNFREEZE_LEVELS if str(l) in runs and runs[str(l)]["status"] == "done"]

    if not levels:
        print(f"[SKIP] {backbone_name} — no done experiments")
        return

    n = len(levels)
    fig_loss, axes_loss = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    fig_acc,  axes_acc  = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)

    if n == 1:
        axes_loss = [axes_loss]
        axes_acc  = [axes_acc]

    title_suffix = f" - {note}" if note else ""
    fig_loss.suptitle(f"{backbone_name} - Loss{title_suffix}", fontsize=13)
    fig_acc.suptitle( f"{backbone_name} - Accuracy{title_suffix}", fontsize=13)

    for ax_loss, ax_acc, level in zip(axes_loss, axes_acc, levels):
        entry   = runs[level]
        history = entry["history"]
        epochs      = [h["epoch"]      for h in history]
        train_loss  = [h["train_loss"] for h in history]
        val_loss    = [h["val_loss"]   for h in history]
        val_acc     = [h["val_acc"]    for h in history]

        # --- loss ---
        ax_loss.plot(epochs, train_loss, label="train loss")
        ax_loss.plot(epochs, val_loss,   label="val loss")
        ax_loss.set_title(f"unfreeze={level}")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.legend()
        if entry.get("early_stopped"):
            ax_loss.axvline(x=entry["epochs_trained"], color="red",
                            linestyle="--", alpha=0.5, label="early stop")
            ax_loss.legend()

        # --- accuracy ---
        ax_acc.plot(epochs, val_acc, label="val acc", color="green")
        ax_acc.set_title(f"unfreeze={level}  best={entry['best_val_acc']:.4f}")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.legend()
        if entry.get("early_stopped"):
            ax_acc.axvline(x=entry["epochs_trained"], color="red",
                           linestyle="--", alpha=0.5, label="early stop")
            ax_acc.legend()

    fig_loss.tight_layout()
    fig_acc.tight_layout()

    out_dir = RESULTS_DIR / "visualisations" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = out_dir / f"{backbone_name}_loss.png"
    acc_path  = out_dir / f"{backbone_name}_accuracy.png"
    fig_loss.savefig(loss_path, dpi=150)
    fig_acc.savefig(acc_path,   dpi=150)
    print(f"  Saved: {loss_path}")
    print(f"  Saved: {acc_path}")

    plt.close(fig_loss)
    plt.close(fig_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID to visualize")
    args = parser.parse_args()

    results = load_results(args.run_id)
    note    = results.get("_meta", {}).get("note")

    print(f"Run:  {args.run_id}")
    print(f"Note: {note or '—'}\n")

    for backbone_name, runs in results.items():
        if backbone_name == "_meta":
            continue
        print(f"Plotting {backbone_name}...")
        plot_backbone(backbone_name, runs, args.run_id, note)


if __name__ == "__main__":
    main()