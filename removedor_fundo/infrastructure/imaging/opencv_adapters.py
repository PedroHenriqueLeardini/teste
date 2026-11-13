from typing import Any

import cv2
import numpy as np

from removedor_fundo.domain.models import ImageData, MaskData


class OpenCVMaskInspector:
    def object_ratio(self, mask: MaskData) -> float:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size:
            raise ValueError("A mascara deve ser um array 2D nao vazio.")
        return float(np.count_nonzero(mask)) / mask.size


class OpenCVImageComposer:
    def compose(
        self, image: ImageData, mask: MaskData, *, keep_foreground: bool = True
    ) -> ImageData:
        alpha = mask if keep_foreground else cv2.bitwise_not(mask)
        _validate_bgr_image(image)
        alpha = _normalize_mask(alpha)
        if alpha.shape != image.shape[:2]:
            raise ValueError("A mascara deve ter as mesmas dimensoes da imagem.")

        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = alpha
        return result


def load_image(path: Any) -> ImageData:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {path}")
    return image


def save_image(path: Any, image: ImageData) -> None:
    if not isinstance(image, np.ndarray) or not image.size:
        raise ValueError("A imagem a salvar deve ser um array NumPy nao vazio.")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Nao foi possivel salvar a imagem: {path}")


def _validate_bgr_image(image: ImageData) -> None:
    if not isinstance(image, np.ndarray) or not image.size:
        raise ValueError("A imagem deve ser um array NumPy nao vazio.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve estar no formato BGR com tres canais.")


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
