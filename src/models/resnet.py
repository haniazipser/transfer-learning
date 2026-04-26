from .base_backbone import BaseBackbone
import torch.nn as nn
import torchvision.models as models

# input
#  ↓
# conv1
#  ↓
# layer1  → low-level features (edges, textures)
# layer2  → shapes
# layer3  → parts
# layer4  → objects
#  ↓
# fc

class ResNet50(BaseBackbone):
    BLOCKS = ["layer1", "layer2", "layer3", "layer4"]

    def _build(self):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        #m.fc is a deafult head which we override
        # resnet50 had global avg pooling, so no need to add it here
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.fc.in_features, self.num_classes)
            )
        return m

    def unfreeze_last_n_blocks(self, n: int):
        self.freeze_all()
        for block in self.BLOCKS[-n:] if n > 0 else []:
            for p in getattr(self.model, block).parameters():
                p.requires_grad = True
        # head always trainable
        for p in self.model.fc.parameters():
            p.requires_grad = True
