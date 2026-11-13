from typing import Mapping, Optional

from ...domain.contracts import ImageComposer, MaskInspector, MaskRefiner, Segmenter
from ...domain.errors import InvalidSegmentationMask, UnknownSegmentationMethod, UnsafeSegmentation
from ...domain.models import ProcessingResult, RemovalRequest


class RemoveBackground:
    """Gera e refina uma mascara; nao carrega dataset nem calcula metricas."""

    def __init__(
        self,
        segmenters: Mapping[str, Segmenter],
        refiner: MaskRefiner,
        inspector: MaskInspector,
        composer: ImageComposer,
        *,
        minimum_area: float = 0.01,
        maximum_area: float = 0.95,
    ) -> None:
        if not 0.0 <= minimum_area <= maximum_area <= 1.0:
            raise ValueError("A faixa de area valida deve estar entre 0 e 1.")
        self._segmenters = dict(segmenters)
        self._refiner = refiner
        self._inspector = inspector
        self._composer = composer
        self._minimum_area = minimum_area
        self._maximum_area = maximum_area

    def execute(self, request: RemovalRequest) -> ProcessingResult:
        segmenter = self._segmenters.get(request.method)
        if segmenter is None:
            raise UnknownSegmentationMethod(f"Metodo de segmentacao desconhecido: {request.method}")

        generated = segmenter.generate_mask(
            request.image,
            object_name=request.object_name,
            parameters=request.parameters,
        )
        raw_mask = _copy_array(generated)
        self._validate_mask(raw_mask, request.image)
        raw_ratio = self._inspector.object_ratio(raw_mask)
        self._validate_ratio(raw_ratio)

        processed_mask = (
            self._refiner.refine(_copy_array(raw_mask))
            if request.refine
            else _copy_array(raw_mask)
        )
        self._validate_mask(processed_mask, request.image)
        processed_ratio = self._inspector.object_ratio(processed_mask)
        self._validate_ratio(processed_ratio)

        return ProcessingResult(
            raw_mask=raw_mask,
            processed_mask=_copy_array(processed_mask),
            transparent_image=self._composer.compose(
                request.image, processed_mask, keep_foreground=request.keep_foreground
            ),
            method=request.method,
            refined=request.refine,
        )

    def _validate_ratio(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise InvalidSegmentationMask("A proporcao da mascara deve estar entre 0 e 1.")
        if ratio < self._minimum_area:
            raise UnsafeSegmentation(
                f"A regiao selecionada e pequena demais ({ratio * 100:.2f}% da imagem)."
            )
        if ratio > self._maximum_area:
            raise UnsafeSegmentation(
                f"A regiao selecionada ocupa quase toda a imagem ({ratio * 100:.1f}%)."
            )

    def _validate_mask(self, mask: object, image: object) -> None:
        shape = getattr(mask, "shape", None)
        image_shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 2 or not shape[0] or not shape[1]:
            raise InvalidSegmentationMask("O segmentador deve retornar uma mascara 2D nao vazia.")
        if image_shape is None or tuple(shape) != tuple(image_shape[:2]):
            raise InvalidSegmentationMask(
                "A mascara gerada deve ter as mesmas dimensoes da imagem."
            )


def _copy_array(value: object) -> object:
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value
