import torch
import torch.nn as nn
from torch.optim import Adam

from src.config import Config
from src.data.datamodule import DataModule
from src.models.base_backbone import BaseBackbone


class Trainer:
    def __init__(
        self,
        backbone: BaseBackbone,
        data: DataModule,
        config: Config,
    ):
        self.backbone   = backbone
        self.data       = data
        self.config     = config
        self.device     = torch.device(config.device)
        self.history    = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(self):
        print(f"using {self.device}")
        model     = self.backbone.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.backbone.trainable_params(), lr=self.config.lr)

        for epoch in range(1, self.config.num_epochs + 1):
            print("training")
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
                print("ok")
            train_loss /= len(self.data.train_loader.dataset)

            print("end of training")

            # --- val ---
            model.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for x, y in self.data.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out  = model(x)
                    val_loss += criterion(out, y).item() * x.size(0)
                    correct  += (out.argmax(1) == y).sum().item()
            val_loss /= len(self.data.test_loader.dataset)
            val_acc   = correct / len(self.data.test_loader.dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:>2}/{self.config.num_epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        return self.history