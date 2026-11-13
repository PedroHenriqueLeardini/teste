from typing import Any, Mapping, Optional

import cv2
import numpy as np

from removedor_fundo.domain.models import ImageData, MaskData


_COLOR_CONVERSIONS = {
    "rgb": cv2.COLOR_BGR2RGB,
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
}

_DEFAULT_PARAMETERS = {
    "rgb": {"inferior": (80, 30, 20), "superior": (255, 180, 160)},
    "hsv": {"inferior": (0, 30, 30), "superior": (25, 255, 255)},
    "lab": {"inferior": (0, 120, 120), "superior": (255, 180, 180)},
}


class ColorThresholdSegmenter:
    def __init__(self, method: str, parameters: Optional[Mapping[str, Any]] = None) -> None:
        if method not in _COLOR_CONVERSIONS:
            raise ValueError(f"Metodo de cor desconhecido: {method}")
        self.name = method
        self._parameters = dict(parameters or _DEFAULT_PARAMETERS[method])

    def generate_mask(
        self,
        image: ImageData,
        *,
        object_name: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> MaskData:
        selected = dict(self._parameters)
        if parameters:
            selected.update(parameters)
        if not isinstance(image, np.ndarray) or not image.size:
            raise ValueError("A imagem deve ser um array NumPy nao vazio.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("A imagem deve estar no formato BGR com tres canais.")
        converted = cv2.cvtColor(image, _COLOR_CONVERSIONS[self.name])
        return cv2.inRange(
            converted,
            np.asarray(selected["inferior"], dtype=np.uint8),
            np.asarray(selected["superior"], dtype=np.uint8),
        )
