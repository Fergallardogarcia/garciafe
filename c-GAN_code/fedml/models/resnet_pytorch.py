"""Implementation of a ResNet-18 neural network for 3."""

import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock

from .model import BaseModel

model_urls = "https://download.pytorch.org/models/resnet18-5c106cde.pth"

class Net(ResNet, BaseModel):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes, pretrained=False) -> None:
        super(Net, self).__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        self.num_classes = num_classes

        if pretrained:
            state_dict = model_zoo.load_url(model_urls)
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            self.load_state_dict(state_dict, strict=False)
