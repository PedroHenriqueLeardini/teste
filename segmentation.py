"""Fachada de compatibilidade para os adaptadores OpenCV da aplicacao."""

import cv2

from image_processing import converter_grayscale
from removedor_fundo.infrastructure.refinement.opencv_refiner import (
    OpenCVMorphologicalRefiner,
)
from removedor_fundo.infrastructure.segmentation.opencv_segmenters import (
    ColorThresholdSegmenter,
)


def criar_mascara_rgb(imagem_bgr, limite_inferior, limite_superior):
    return ColorThresholdSegmenter("rgb").generate_mask(
        imagem_bgr,
        parameters={"inferior": limite_inferior, "superior": limite_superior},
    )


def criar_mascara_hsv(imagem_bgr, limite_inferior, limite_superior):
    return ColorThresholdSegmenter("hsv").generate_mask(
        imagem_bgr,
        parameters={"inferior": limite_inferior, "superior": limite_superior},
    )


def criar_mascara_lab(imagem_bgr, limite_inferior, limite_superior):
    return ColorThresholdSegmenter("lab").generate_mask(
        imagem_bgr,
        parameters={"inferior": limite_inferior, "superior": limite_superior},
    )


def criar_mascara_grayscale(imagem_bgr):
    gray = converter_grayscale(imagem_bgr)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


def refinar_mascara(mascara):
    return OpenCVMorphologicalRefiner().refine(mascara)


def abertura(mascara, tamanho_kernel=3):
    return OpenCVMorphologicalRefiner.opening(mascara, tamanho_kernel)


def fechamento(mascara, tamanho_kernel=3):
    return OpenCVMorphologicalRefiner.closing(mascara, tamanho_kernel)


def erosao(mascara, tamanho_kernel=3):
    return OpenCVMorphologicalRefiner.erosion(mascara, tamanho_kernel)


def dilatacao(mascara, tamanho_kernel=3):
    return OpenCVMorphologicalRefiner.dilation(mascara, tamanho_kernel)


def gradiente(mascara):
    return OpenCVMorphologicalRefiner.gradient(mascara)
