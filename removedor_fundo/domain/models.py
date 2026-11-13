from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


ImageData = Any
MaskData = Any


@dataclass(frozen=True)
class RemovalRequest:
    image: ImageData
    method: str
    object_name: Optional[str] = None
    refine: bool = True
    keep_foreground: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResult:
    raw_mask: MaskData
    processed_mask: MaskData
    transparent_image: ImageData
    method: str
    refined: bool


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image: ImageData
    trimap: MaskData


@dataclass(frozen=True)
class DatasetEvaluation:
    sample_id: str
    method: str
    raw_metrics: Mapping[str, float]
    processed_metrics: Mapping[str, float]
    raw_mask: MaskData
    processed_mask: MaskData


@dataclass(frozen=True)
class UserCommand:
    action: str
    object_name: Optional[str]
    method: str
    refine: bool
