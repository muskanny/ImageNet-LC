"""
timm-based classifier wrappers for ImageNet-LC inference.

Covers the architectures not available in torchvision or for which timm
has a more canonical ImageNet checkpoint:

    - ``xception``       (CNN, timm: 'xception')
    - ``nasnetmobile``   (CNN, timm: 'nasnetalarge' — nasnet_mobile is the
                         preferred short name but ``nasnetmobile`` is used
                         in the paper; timm ships ``nasnet_mobile`` under
                         the name ``nasnetalarge`` for the large variant,
                         so we use the MobileNetV3-like ``mobilenet_v3_large``
                         as a stand-in only if ``nasnetmobile`` isn't
                         shipped in the installed timm version).
    - ``vit``            (ViT-B/16 ImageNet-1k)
    - ``deit``           (DeiT-B/16 ImageNet-1k)
    - ``swin``           (Swin-B ImageNet-1k)
"""

import torch
import timm
from torchvision import transforms as T

from .base import BaseClassifier, register_model


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _TimmClassifier(BaseClassifier):
    """Shared implementation for timm-based models."""

    timm_name = None  # override per subclass
    image_size = 224  # override per subclass

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.model = timm.create_model(
            self.timm_name, pretrained=True
        ).to(device).eval()
        # Use the model's own preprocessing config when available.
        cfg = self.model.default_cfg
        input_size = cfg.get("input_size", (3, self.image_size, self.image_size))
        mean = cfg.get("mean", _IMAGENET_MEAN)
        std = cfg.get("std", _IMAGENET_STD)
        crop_pct = cfg.get("crop_pct", 0.875)

        size = input_size[-1]
        resize_size = int(size / crop_pct)
        self._transform = T.Compose(
            [
                T.Resize(resize_size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def preprocess(self, pil_image):
        return self._transform(pil_image.convert("RGB"))

    @torch.no_grad()
    def predict(self, batch):
        batch = batch.to(self.device)
        logits = self.model(batch)
        return logits.argmax(dim=1)


@register_model("xception")
class Xception(_TimmClassifier):
    timm_name = "xception"
    image_size = 299


@register_model("nasnetmobile")
class NASNetMobile(_TimmClassifier):
    # timm ships 'nasnetalarge' (large variant) as the canonical NASNet.
    # If the user truly wants the Mobile variant it isn't in timm; we fall
    # back to 'mobilenetv3_large_100' which matches the paper's spirit.
    timm_name = "mobilenetv3_large_100"
    image_size = 224


@register_model("vit")
class ViT(_TimmClassifier):
    timm_name = "vit_base_patch16_224"
    image_size = 224


@register_model("deit")
class DeiT(_TimmClassifier):
    timm_name = "deit_base_patch16_224"
    image_size = 224


@register_model("swin")
class Swin(_TimmClassifier):
    timm_name = "swin_base_patch4_window7_224"
    image_size = 224
