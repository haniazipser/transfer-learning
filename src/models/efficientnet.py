from .base_backbone import BaseBackbone
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0(BaseBackbone):
    # We treat each feature stage, including the final 1x1 conv, as a block.
    BLOCKS = [1, 2, 3, 4, 5, 6, 7, 8]

    def _build(self):
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, self.num_classes)
        return m

    def unfreeze_last_n_blocks(self, n: int):
        if n == -1:
            self.unfreeze_all()
            return

        self.freeze_all()
        for block_idx in self.BLOCKS[-n:] if n > 0 else []:
            for p in self.model.features[block_idx].parameters():
                p.requires_grad = True
        # Head is always trainable.
        for p in self.model.classifier.parameters():
            p.requires_grad = True

    def backbone_parameters(self):
        yield from self.model.features.parameters()

    def head_parameters(self):
        yield from self.model.classifier.parameters()
