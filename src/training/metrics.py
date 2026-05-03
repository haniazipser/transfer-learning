from sklearn.metrics import classification_report
import wandb


class MetricsLogger:
    def __init__(self, class_names: list[str]):
        self.class_names = class_names

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        all_preds: list[int],
        all_labels: list[int],
        lr_head: float,
        lr_backbone: float,
    ):
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        per_class = {}
        for name in self.class_names:
            per_class[f"precision_{name}"] = report[name]["precision"]
            per_class[f"recall_{name}"]    = report[name]["recall"]
            per_class[f"f1_{name}"]        = report[name]["f1-score"]

        wandb.log({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "val_acc":      val_acc,
            "lr_head":      lr_head,
            "lr_backbone":  lr_backbone,
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=self.class_names,
            ),
            **per_class,
        })

    def finish(self):
        wandb.finish()