import unittest

import numpy as np

import evaluation
import segmentation
from removedor_fundo.infrastructure.evaluation.trimap_evaluator import (
    OxfordPetTrimapEvaluator,
)
from removedor_fundo.infrastructure.refinement.opencv_refiner import (
    OpenCVMorphologicalRefiner,
)
from removedor_fundo.infrastructure.segmentation.opencv_segmenters import (
    ColorThresholdSegmenter,
)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_color_segmentation_facades_preserve_public_results(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[2:6, 2:6] = (10, 30, 80)
        bounds = ((0, 0, 0), (255, 255, 255))

        for method, facade in (
            ("rgb", segmentation.criar_mascara_rgb),
            ("hsv", segmentation.criar_mascara_hsv),
            ("lab", segmentation.criar_mascara_lab),
        ):
            with self.subTest(method=method):
                expected = ColorThresholdSegmenter(method).generate_mask(
                    image,
                    parameters={"inferior": bounds[0], "superior": bounds[1]},
                )
                actual = facade(image, *bounds)
                np.testing.assert_array_equal(actual, expected)

    def test_refinement_and_morphology_facades_delegate_to_adapters(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:7, 2:7] = 255
        mask[0, 0] = 255
        original = mask.copy()

        np.testing.assert_array_equal(
            segmentation.refinar_mascara(mask),
            OpenCVMorphologicalRefiner().refine(mask),
        )
        np.testing.assert_array_equal(
            segmentation.abertura(mask, 5),
            OpenCVMorphologicalRefiner.opening(mask, 5),
        )
        np.testing.assert_array_equal(mask, original)

    def test_evaluation_facade_matches_dataset_evaluator(self):
        mask = np.array([[255, 0], [255, 255]], dtype=np.uint8)
        trimap = np.array([[1, 2], [3, 1]], dtype=np.uint8)

        self.assertEqual(
            evaluation.avaliar_mascara(mask, trimap),
            OxfordPetTrimapEvaluator().evaluate(mask, trimap),
        )


if __name__ == "__main__":
    unittest.main()
