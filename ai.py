"""Fachada de compatibilidade para os adaptadores de IA separados."""

import logging
import os
from typing import Any, Mapping

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from removedor_fundo.application.use_cases.execute_command import ExecuteCommand
from removedor_fundo.domain.errors import InvalidCommand
from removedor_fundo.infrastructure.segmentation.labels import (
    CATEGORIAS_VOC,
    OBJETOS_PT,
    classes_for_object,
    normalize_text,
)


logger = logging.getLogger(__name__)
ACOES_VALIDAS = {"remover_fundo", "comparar_metodos"}
METODOS_VALIDOS = {"rgb", "hsv", "lab", "ia"}


def _normalizar_texto(texto: object) -> str:
    return normalize_text(texto)


def classes_do_objeto(objeto: str):
    return classes_for_object(objeto)


def carregar_modelo():
    from removedor_fundo.infrastructure.segmentation.torchvision_segmenter import (
        load_torchvision_model,
    )

    return load_torchvision_model()


def segmentar_ia(imagem_bgr, objeto=None, limiar=0.5):
    from removedor_fundo.infrastructure.segmentation.torchvision_segmenter import (
        TorchVisionSegmenter,
    )

    return TorchVisionSegmenter().generate_mask(
        imagem_bgr,
        object_name=objeto,
        parameters={"limiar": limiar},
    )


def interpretar_comando(texto: str, usar_llm: bool = True) -> Mapping[str, Any]:
    if usar_llm and os.environ.get("GROQ_API_KEY"):
        from removedor_fundo.infrastructure.generative.groq_interpreter import (
            GroqCommandInterpreter,
        )

        return GroqCommandInterpreter().interpret(texto)

    from removedor_fundo.infrastructure.generative.rule_based_interpreter import (
        RuleBasedCommandInterpreter,
    )

    return RuleBasedCommandInterpreter().interpret(texto)


def validar_acao(dados):
    try:
        command = ExecuteCommand.validate(dados)
        if command.action == "remover_fundo" and command.method == "ia":
            classes_for_object(command.object_name)
    except (InvalidCommand, ValueError) as error:
        return False, None, str(error)

    cleaned = {
        "acao": command.action,
        "objeto": command.object_name,
        "metodo": command.method,
        "refinar": command.refine,
    }
    return True, cleaned, ""


def explicar_resultado(acao, resultado, usar_llm=True):
    summary = {
        key: value
        for key, value in resultado.items()
        if key in {"metodo", "refinado"}
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    if usar_llm and os.environ.get("GROQ_API_KEY"):
        try:
            from removedor_fundo.infrastructure.generative.groq_result_explainer import (
                GroqResultExplainer,
            )

            return GroqResultExplainer().explain(acao, summary)
        except Exception as error:
            logger.warning("Explicacao Groq indisponivel; usando texto local: %s", error)
    return f"Ação '{acao}' concluída."
