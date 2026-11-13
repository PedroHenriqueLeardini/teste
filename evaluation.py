"""Fachada de compatibilidade para a avaliacao por trimaps Oxford Pet."""

from removedor_fundo.infrastructure.evaluation.trimap_evaluator import (
    OxfordPetTrimapEvaluator,
)


def avaliar_mascara(mascara, trimap):
    return OxfordPetTrimapEvaluator().evaluate(mascara, trimap)
