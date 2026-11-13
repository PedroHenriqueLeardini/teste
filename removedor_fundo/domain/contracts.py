from typing import Any, Mapping, Optional, Protocol, Sequence

from .models import DatasetEvaluation, DatasetSample, ImageData, MaskData, ProcessingResult


class Segmenter(Protocol):
    name: str

    def generate_mask(
        self,
        image: ImageData,
        *,
        object_name: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> MaskData: ...


class MaskRefiner(Protocol):
    def refine(self, mask: MaskData, parameters: Optional[Mapping[str, Any]] = None) -> MaskData: ...


class MaskInspector(Protocol):
    def object_ratio(self, mask: MaskData) -> float: ...


class ImageComposer(Protocol):
    def compose(self, image: ImageData, mask: MaskData, *, keep_foreground: bool = True) -> ImageData: ...


class MaskEvaluator(Protocol):
    def evaluate(self, mask: MaskData, trimap: MaskData) -> Mapping[str, float]: ...


class CommandInterpreter(Protocol):
    def interpret(self, text: str) -> Mapping[str, Any]: ...


class DatasetRepository(Protocol):
    def samples(self) -> Sequence[DatasetSample]: ...


class ArtifactRepository(Protocol):
    def save_processing_result(self, result: ProcessingResult, name: str) -> Mapping[str, str]: ...

    def save_dataset_evaluations(
        self, evaluations: Sequence[DatasetEvaluation], name: str
    ) -> str: ...

