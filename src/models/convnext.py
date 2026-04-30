from .base_backbone import BaseBackbone
import torch.nn as nn
import torchvision.models as models


class ConvNeXtTiny(BaseBackbone):
    # ConvNeXt is organized into 4 main stages, separated by downsampling layers.
    BLOCKS = [1, 3, 5, 7]

    def _build(self):
        m = models.convnext_tiny( weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, self.num_classes)
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
