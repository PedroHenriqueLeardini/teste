from typing import Mapping

import numpy as np

from removedor_fundo.domain.models import MaskData


class OxfordPetTrimapEvaluator:
    """Calcula metricas contra trimaps; nao e usado pelo fluxo da interface."""

    def evaluate(self, mask: MaskData, trimap: MaskData) -> Mapping[str, float]:
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            raise ValueError("A mascara deve ser um array NumPy nao vazio.")
        if not isinstance(trimap, np.ndarray) or trimap.size == 0:
            raise ValueError("O trimap deve ser um array NumPy nao vazio.")
        if mask.ndim != 2 or trimap.ndim != 2:
            raise ValueError("Mascara e trimap devem ser bidimensionais.")
        if mask.shape != trimap.shape:
            raise ValueError("Mascara e trimap devem ter as mesmas dimensoes.")

        values = set(np.unique(trimap).tolist())
        if not values.issubset({1, 2, 3}):
            raise ValueError("O trimap deve conter somente os valores 1, 2 e 3.")

        prediction = mask > 0
        reference = trimap == 1
        uncertainty = trimap == 3
        valid_pixels = trimap != 3
        predicted_valid = prediction[valid_pixels]
        reference_valid = reference[valid_pixels]

        tp = int(np.sum(predicted_valid & reference_valid))
        fp = int(np.sum(predicted_valid & ~reference_valid))
        fn = int(np.sum(~predicted_valid & reference_valid))
        tn = int(np.sum(~predicted_valid & ~reference_valid))

        def divide(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        uncertain_count = int(np.sum(uncertainty))
        predicted_count = int(np.sum(prediction))
        predicted_uncertain = int(np.sum(prediction & uncertainty))
        return {
            "iou": divide(tp, tp + fp + fn),
            "precisao": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn),
            "f1_score": divide(2 * tp, 2 * tp + fp + fn),
            "acuracia": divide(tp + tn, tp + tn + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pixels_regiao_incerta": uncertain_count,
            "pixels_previstos_na_incerteza": predicted_uncertain,
            "proporcao_mascara_na_incerteza": divide(
                predicted_uncertain, predicted_count
            ),
            "proporcao_incerteza_como_animal": divide(
                predicted_uncertain, uncertain_count
            ),
            "proporcao_incerteza_como_fundo": divide(
                uncertain_count - predicted_uncertain, uncertain_count
            ),
        }
