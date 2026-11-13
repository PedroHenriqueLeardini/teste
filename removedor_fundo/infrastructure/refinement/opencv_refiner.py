from typing import Any, Mapping, Optional

import cv2
import numpy as np

from removedor_fundo.domain.models import MaskData


class OpenCVMorphologicalRefiner:
    def refine(
        self, mask: MaskData, parameters: Optional[Mapping[str, Any]] = None
    ) -> MaskData:
        normalized = _normalize_mask(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def opening(mask: MaskData, kernel_size: int = 3) -> MaskData:
        return _apply_morphology(mask, cv2.MORPH_OPEN, kernel_size)

    @staticmethod
    def closing(mask: MaskData, kernel_size: int = 3) -> MaskData:
        return _apply_morphology(mask, cv2.MORPH_CLOSE, kernel_size)

    @staticmethod
    def erosion(mask: MaskData, kernel_size: int = 3) -> MaskData:
        return _apply_morphology(mask, cv2.MORPH_ERODE, kernel_size)

    @staticmethod
    def dilation(mask: MaskData, kernel_size: int = 3) -> MaskData:
        return _apply_morphology(mask, cv2.MORPH_DILATE, kernel_size)

    @staticmethod
    def gradient(mask: MaskData, kernel_size: int = 3) -> MaskData:
        return _apply_morphology(mask, cv2.MORPH_GRADIENT, kernel_size)


def _normalize_mask(mask: MaskData) -> MaskData:
    if not isinstance(mask, np.ndarray) or not mask.size:
        raise ValueError("A mascara deve ser um array NumPy nao vazio.")
    if mask.ndim == 3:
        if mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError("A mascara deve ter um, tres ou quatro canais.")
    elif mask.ndim != 2:
        raise ValueError("A mascara deve ser bidimensional ou ter canais de cor.")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _apply_morphology(mask: MaskData, operation: int, kernel_size: int) -> MaskData:
    if not isinstance(mask, np.ndarray) or not mask.size:
        raise ValueError("A mascara deve ser um array NumPy nao vazio.")
    if not isinstance(kernel_size, int) or kernel_size < 1:
        raise ValueError("O tamanho do kernel deve ser um inteiro positivo.")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    if operation == cv2.MORPH_ERODE:
        return cv2.erode(mask, kernel, iterations=1)
    if operation == cv2.MORPH_DILATE:
        return cv2.dilate(mask, kernel, iterations=1)
    return cv2.morphologyEx(mask, operation, kernel)
