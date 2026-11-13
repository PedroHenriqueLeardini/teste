"""Adaptadores de leitura, composicao e armazenamento de imagens."""

from .opencv_adapters import load_image, save_image

__all__ = ["load_image", "save_image"]
