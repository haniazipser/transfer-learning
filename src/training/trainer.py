import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from .metrics import MetricsLogger
from ..config import Config
from ..data.datamodule import DataModule
from ..models.base_backbone import BaseBackbone


class Trainer:
    def __init__(
        self,
        backbone: BaseBackbone,
        data: DataModule,
        config: Config,
        unfreeze: int
    ):
        self.backbone   = backbone
        self.data       = data
        self.config     = config
        self.device     = torch.device(config.device)
        self.history    = {"train_loss": [], "val_loss": [], "val_acc": []}

        wandb.init(
            project="corn-classification",
            name=f"{type(backbone).__name__}_unfreeze{unfreeze}",
            config={
                "backbone": type(backbone).__name__,
                "backbone_lr": config.backbone_lr,
                "head_lr": config.head_lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "patience": config.patience,
                "l2": config.l2,
                "val_split": config.val_split,
            }
        )

        self.metrics = MetricsLogger(
            class_names=data.train_loader.dataset.dataset.labels
        )

    def fit(self):
        model     = self.backbone.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            self.backbone.parameter_groups(
                backbone_lr=self.config.backbone_lr,
                head_lr=self.config.head_lr,
            ),
            weight_decay=self.config.l2,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.eta_min
        )

        min_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            # --- train ---
            model.train()
            train_loss = 0.0
            for x, y in self.data.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(self.data.train_loader.dataset)

            # --- val ---
            model.eval()
            val_loss, correct = 0.0, 0

            all_preds, all_labels = [], []

            with torch.no_grad():
                for x, y in self.data.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out  = model(x)

                    val_loss += criterion(out, y).item() * x.size(0)
                    correct  += (out.argmax(1) == y).sum().item()

                    all_preds.extend(out.argmax(1).cpu().tolist())
                    all_labels.extend(y.cpu().tolist())

            val_loss /= len(self.data.val_loader.dataset)
            val_acc   = correct / len(self.data.val_loader.dataset)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:>2}/{self.config.num_epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            self.metrics.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                all_preds=all_preds,
                all_labels=all_labels,
                lr_head=optimizer.param_groups[0]["lr"],
                lr_backbone=optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            )

            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in val_loss for {self.config.patience} consecutive epochs.")
                break

            scheduler.step()

        self.metrics.finish()
        return self.history
