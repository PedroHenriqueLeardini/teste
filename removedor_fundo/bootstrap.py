"""Composition root: cria adaptadores concretos fora dos casos de uso."""

import os
from typing import Any, Callable, Mapping, Optional

from removedor_fundo.application.use_cases.compare_methods import CompareMethods
from removedor_fundo.application.use_cases.evaluate_dataset import EvaluateDataset
from removedor_fundo.application.use_cases.execute_command import ExecuteCommand
from removedor_fundo.application.use_cases.remove_background import RemoveBackground
from removedor_fundo.domain.models import RemovalRequest
from removedor_fundo.infrastructure.evaluation.trimap_evaluator import OxfordPetTrimapEvaluator
from removedor_fundo.infrastructure.imaging.opencv_adapters import (
    OpenCVImageComposer,
    OpenCVMaskInspector,
)
from removedor_fundo.infrastructure.refinement.opencv_refiner import OpenCVMorphologicalRefiner
from removedor_fundo.infrastructure.generative.rule_based_interpreter import (
    RuleBasedCommandInterpreter,
)
from removedor_fundo.infrastructure.segmentation.opencv_segmenters import ColorThresholdSegmenter
from removedor_fundo.infrastructure.segmentation.torchvision_segmenter import TorchVisionSegmenter


def build_segmenters():
    return {
        "rgb": ColorThresholdSegmenter("rgb"),
        "hsv": ColorThresholdSegmenter("hsv"),
        "lab": ColorThresholdSegmenter("lab"),
        "ia": TorchVisionSegmenter(),
    }


def build_remove_background() -> RemoveBackground:
    return RemoveBackground(
        segmenters=build_segmenters(),
        refiner=OpenCVMorphologicalRefiner(),
        inspector=OpenCVMaskInspector(),
        composer=OpenCVImageComposer(),
    )


def build_compare_methods() -> CompareMethods:
    return CompareMethods(
        segmenters=build_segmenters(),
        refiner=OpenCVMorphologicalRefiner(),
        composer=OpenCVImageComposer(),
    )


def build_evaluate_dataset() -> EvaluateDataset:
    return EvaluateDataset(
        segmenters=build_segmenters(),
        refiner=OpenCVMorphologicalRefiner(),
        evaluator=OxfordPetTrimapEvaluator(),
    )


def build_execute_command(
    use_llm: bool = True,
    handlers: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> ExecuteCommand:
    from dotenv import load_dotenv

    load_dotenv()
    if use_llm and os.environ.get("GROQ_API_KEY"):
        from removedor_fundo.infrastructure.generative.groq_interpreter import (
            GroqCommandInterpreter,
        )

        interpreter = GroqCommandInterpreter()
    else:
        interpreter = RuleBasedCommandInterpreter()

    def remove_handler(command, **context):
        return build_remove_background().execute(
            RemovalRequest(
                image=context["image"],
                method=command.method,
                object_name=command.object_name,
                refine=command.refine,
            )
        )

    def compare_handler(command, **context):
        return build_compare_methods().execute(
            context["image"],
            methods=("rgb", "hsv", "lab", "ia"),
            object_name=command.object_name,
            refine=command.refine,
        )

    return ExecuteCommand(
        interpreter,
        handlers
        or {
            "remover_fundo": remove_handler,
            "comparar_metodos": compare_handler,
        },
    )
