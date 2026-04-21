"""
Base classifier interface and registry.

Every concrete model class inherits from ``BaseClassifier`` and registers
itself in ``MODEL_REGISTRY`` via the ``@register_model`` decorator.
"""

from abc import ABC, abstractmethod


MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a classifier factory under a short name."""
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return _wrap


class BaseClassifier(ABC):
    """
    Minimal unified interface for an ImageNet-pretrained classifier.

    All architectures in Table 1 of the paper are wrapped behind this API:

        - ``preprocess(pil_image) -> torch.Tensor``: single-image preprocess.
        - ``predict(batch_tensor) -> torch.LongTensor``: top-1 predictions
          on a batch, as ImageNet class indices in ``[0, 1000)``.

    Parameters
    ----------
    device : str, optional
        Torch device string, e.g. ``'cpu'`` or ``'cuda'``. Defaults to CPU.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def preprocess(self, pil_image):
        """Return a ``(3, H, W)`` float tensor for a PIL.Image input."""

    @abstractmethod
    def predict(self, batch):
        """Return top-1 class indices for a ``(B, 3, H, W)`` batch."""


def list_models():
    """Return the list of registered model short names."""
    return sorted(MODEL_REGISTRY.keys())


def load_model(name, device="cpu"):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered: {list_models()}"
        )
    return MODEL_REGISTRY[name](device=device)
