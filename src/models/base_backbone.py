from abc import ABC, abstractmethod
import torch.nn as nn


class BaseBackbone(ABC):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model: nn.Module = self._build()

    @abstractmethod
    def _build(self) -> nn.Module:
        ...

    @abstractmethod
    def unfreeze_last_n_blocks(self, n: int) -> None:
        ...

    @abstractmethod
    def backbone_parameters(self):
        ...

    @abstractmethod
    def head_parameters(self):
        ...

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        groups = []
        backbone_params = [p for p in self.backbone_parameters() if p.requires_grad]
        head_params = [p for p in self.head_parameters() if p.requires_grad]

        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})

        return groups
