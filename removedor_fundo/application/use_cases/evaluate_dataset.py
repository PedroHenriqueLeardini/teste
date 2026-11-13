from typing import Dict, Mapping, Optional, Sequence

from ...domain.contracts import MaskEvaluator, MaskRefiner, Segmenter
from ...domain.errors import UnknownSegmentationMethod
from ...domain.models import DatasetEvaluation, DatasetSample


class EvaluateDataset:
    """Avalia mascaras somente para amostras pareadas com trimaps do dataset."""

    def __init__(
        self,
        segmenters: Mapping[str, Segmenter],
        refiner: MaskRefiner,
        evaluator: MaskEvaluator,
    ) -> None:
        self._segmenters = dict(segmenters)
        self._refiner = refiner
        self._evaluator = evaluator

    def execute(
        self,
        samples: Sequence[DatasetSample],
        methods: Optional[Sequence[str]] = None,
        object_name: Optional[str] = "animal",
    ) -> Sequence[DatasetEvaluation]:
        selected_methods = list(methods) if methods is not None else list(self._segmenters)
        evaluations = []
        for sample in samples:
            for method in selected_methods:
                segmenter = self._segmenters.get(method)
                if segmenter is None:
                    raise UnknownSegmentationMethod(
                        f"Metodo de segmentacao desconhecido: {method}"
                    )
                raw_mask = segmenter.generate_mask(
                    sample.image, object_name=object_name
                )
                processed_mask = self._refiner.refine(_copy_array(raw_mask))
                evaluations.append(
                    DatasetEvaluation(
                        sample_id=sample.sample_id,
                        method=method,
                        raw_metrics=dict(self._evaluator.evaluate(raw_mask, sample.trimap)),
                        processed_metrics=dict(
                            self._evaluator.evaluate(processed_mask, sample.trimap)
                        ),
                        raw_mask=_copy_array(raw_mask),
                        processed_mask=_copy_array(processed_mask),
                    )
                )
        return evaluations


def _copy_array(value: object) -> object:
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value
