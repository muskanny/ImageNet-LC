"""
Torchvision-based classifier wrappers for ImageNet-LC inference.

Every CNN from Table 1 is exposed under a short name matching the paper
(``alexnet``, ``resnet50v2``, ``vgg19``, ``densenet201``, ``mobilenetv2``,
``inceptionv3``, ``efficientnetb1``, ``xception``, ``nasnetmobile``).

Note on specific architectures
------------------------------
- ``Xception``, ``NASNet-Mobile`` and ``ResNet50V2`` are not in torchvision; 
they come from ``timm`` in ``timm_models.py``.
"""

import torch
import torch.nn as nn
from torchvision import models as tvm
from torchvision import transforms as T

from .base import BaseClassifier, register_model


# Standard ImageNet preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _standard_transform(image_size=224):
    return T.Compose(
        [
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class _TorchvisionClassifier(BaseClassifier):
    """Shared implementation for every torchvision-based model."""

    image_size = 224
    weights_enum = None  # override per subclass
    builder = None  # override per subclass

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        weights = self.weights_enum.DEFAULT if self.weights_enum else None
        self.model = self.builder(weights=weights).to(device).eval()
        self._transform = _standard_transform(self.image_size)

    def preprocess(self, pil_image):
        return self._transform(pil_image.convert("RGB"))

    @torch.no_grad()
    def predict(self, batch):
        batch = batch.to(self.device)
        logits = self.model(batch)
        return logits.argmax(dim=1)


@register_model("alexnet")
class AlexNet(_TorchvisionClassifier):
    weights_enum = tvm.AlexNet_Weights
    builder = staticmethod(tvm.alexnet)


@register_model("vgg19")
class VGG19(_TorchvisionClassifier):
    weights_enum = tvm.VGG19_Weights
    builder = staticmethod(tvm.vgg19)


@register_model("densenet201")
class DenseNet201(_TorchvisionClassifier):
    weights_enum = tvm.DenseNet201_Weights
    builder = staticmethod(tvm.densenet201)


@register_model("mobilenetv2")
class MobileNetV2(_TorchvisionClassifier):
    weights_enum = tvm.MobileNet_V2_Weights
    builder = staticmethod(tvm.mobilenet_v2)


@register_model("inceptionv3")
class InceptionV3(_TorchvisionClassifier):
    image_size = 299  # inception's native size
    weights_enum = tvm.Inception_V3_Weights
    builder = staticmethod(tvm.inception_v3)

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        # Inception v3 has aux-logits in training mode; eval turns this off.
        self.model.aux_logits = False


@register_model("efficientnetb1")
class EfficientNetB1(_TorchvisionClassifier):
    image_size = 240  # b1's native size
    weights_enum = tvm.EfficientNet_B1_Weights
    builder = staticmethod(tvm.efficientnet_b1)
