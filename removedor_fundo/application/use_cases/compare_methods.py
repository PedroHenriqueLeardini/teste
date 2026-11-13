from typing import Dict, Mapping, Optional, Sequence

from ...domain.contracts import ImageComposer, MaskRefiner, Segmenter
from ...domain.errors import UnknownSegmentationMethod
from ...domain.models import ImageData, ProcessingResult


class CompareMethods:
    """Compara resultados visuais para uma imagem, sem calcular metricas."""

    def __init__(
        self,
        segmenters: Mapping[str, Segmenter],
        refiner: MaskRefiner,
        composer: ImageComposer,
    ) -> None:
        self._segmenters = dict(segmenters)
        self._refiner = refiner
        self._composer = composer

    def execute(
        self,
        image: ImageData,
        methods: Sequence[str],
        *,
        object_name: Optional[str] = None,
        refine: bool = True,
        keep_foreground: bool = True,
    ) -> Mapping[str, ProcessingResult]:
        results: Dict[str, ProcessingResult] = {}
        for method in methods:
            segmenter = self._segmenters.get(method)
            if segmenter is None:
                raise UnknownSegmentationMethod(
                    f"Metodo de segmentacao desconhecido: {method}"
                )
            raw_mask = _copy_array(
                segmenter.generate_mask(image, object_name=object_name)
            )
            processed_mask = (
                self._refiner.refine(_copy_array(raw_mask))
                if refine
                else _copy_array(raw_mask)
            )
            results[method] = ProcessingResult(
                raw_mask=raw_mask,
                processed_mask=_copy_array(processed_mask),
                transparent_image=self._composer.compose(
                    image, processed_mask, keep_foreground=keep_foreground
                ),
                method=method,
                refined=refine,
            )
        return results


def _copy_array(value: object) -> object:
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value
