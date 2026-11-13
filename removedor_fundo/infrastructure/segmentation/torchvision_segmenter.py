from typing import Any, Mapping, Optional

import cv2
import numpy as np

from removedor_fundo.domain.models import ImageData, MaskData
from removedor_fundo.infrastructure.segmentation.labels import classes_for_object


_MODEL = None


def load_torchvision_model():
    global _MODEL
    if _MODEL is None:
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        model.eval()
        _MODEL = model, weights.transforms()
    return _MODEL


class TorchVisionSegmenter:
    """Adaptador para o DeepLabV3 ja utilizado pelo projeto."""

    name = "ia"

    def generate_mask(
        self,
        image: ImageData,
        *,
        object_name: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> MaskData:
        import torch
        import torch.nn.functional as functional

        threshold = float((parameters or {}).get("limiar", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("O limiar da IA deve estar entre 0 e 1.")
        indices = classes_for_object(object_name)
        model, transform = load_torchvision_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

        with torch.no_grad():
            logits = model(transform(tensor).unsqueeze(0))["out"]
            logits = functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False
            )
            probabilities = logits.softmax(dim=1)[0]
            object_probability = probabilities[indices].sum(dim=0)
            mask = (object_probability > threshold).cpu().numpy()

        return mask.astype(np.uint8) * 255
