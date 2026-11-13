import unittest

import numpy as np

from removedor_fundo.application.use_cases.evaluate_dataset import EvaluateDataset
from removedor_fundo.application.use_cases.execute_command import ExecuteCommand
from removedor_fundo.application.use_cases.remove_background import RemoveBackground
from removedor_fundo.domain.errors import InvalidCommand, UnsafeSegmentation, UnknownSegmentationMethod
from removedor_fundo.domain.models import DatasetSample, RemovalRequest


class FakeSegmenter:
    name = "fake"

    def __init__(self, mask):
        self.mask = mask

    def generate_mask(self, image, *, object_name=None, parameters=None):
        return self.mask


class MutatingRefiner:
    def refine(self, mask, parameters=None):
        mask[0, 0] = 0
        return mask


class Inspector:
    def object_ratio(self, mask):
        return float(np.count_nonzero(mask)) / mask.size


class Composer:
    def compose(self, image, mask, *, keep_foreground=True):
        return image, mask.copy(), keep_foreground


class Evaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, mask, trimap):
        self.calls.append((mask.copy(), trimap))
        return {"iou": 0.75}


class Interpreter:
    def __init__(self, command):
        self.command = command

    def interpret(self, text):
        return self.command


class ApplicationUseCaseTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.mask = np.full((4, 4), 255, dtype=np.uint8)
        self.segmenter = FakeSegmenter(self.mask)
        self.use_case = RemoveBackground(
            {"fake": self.segmenter},
            MutatingRefiner(),
            Inspector(),
            Composer(),
            minimum_area=0.0,
            maximum_area=1.0,
        )

    def test_removal_preserves_raw_mask_and_returns_separate_processed_mask(self):
        result = self.use_case.execute(RemovalRequest(self.image, "fake", refine=True))

        self.assertTrue(np.all(result.raw_mask == 255))
        self.assertEqual(result.processed_mask[0, 0], 0)
        self.assertFalse(np.shares_memory(result.raw_mask, result.processed_mask))
        self.assertEqual(result.method, "fake")

    def test_removal_without_refinement_returns_copy_and_no_evaluation(self):
        result = self.use_case.execute(RemovalRequest(self.image, "fake", refine=False))

        np.testing.assert_array_equal(result.raw_mask, result.processed_mask)
        self.assertFalse(np.shares_memory(result.raw_mask, result.processed_mask))
        self.assertFalse(hasattr(self.use_case, "_evaluator"))

    def test_rejects_unknown_segmenter(self):
        with self.assertRaises(UnknownSegmentationMethod):
            self.use_case.execute(RemovalRequest(self.image, "missing"))

    def test_rejects_area_after_refinement(self):
        mask = np.full((4, 4), 255, dtype=np.uint8)
        mask[-1, -1] = 0
        segmenter = FakeSegmenter(mask)
        use_case = RemoveBackground(
            {"fake": segmenter}, MutatingRefiner(), Inspector(), Composer(),
            minimum_area=0.9, maximum_area=1.0,
        )
        with self.assertRaises(UnsafeSegmentation):
            use_case.execute(RemovalRequest(self.image, "fake"))

    def test_dataset_evaluation_is_a_separate_use_case(self):
        evaluator = Evaluator()
        dataset_use_case = EvaluateDataset(
            {"fake": self.segmenter}, MutatingRefiner(), evaluator
        )
        sample = DatasetSample("pet-01", self.image, np.ones((4, 4), dtype=np.uint8))

        result = dataset_use_case.execute([sample], methods=["fake"])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(evaluator.calls), 2)
        self.assertEqual(result[0].sample_id, "pet-01")
        self.assertEqual(result[0].raw_metrics["iou"], 0.75)

    def test_command_dispatch_requires_allowlisted_action_and_valid_parameters(self):
        invoked = []
        use_case = ExecuteCommand(
            Interpreter(
                {"acao": "remover_fundo", "objeto": "gato", "metodo": "ia", "refinar": True}
            ),
            {"remover_fundo": lambda **kwargs: invoked.append(kwargs["command"])},
        )

        use_case.execute("remova o fundo", image=self.image)

        self.assertEqual(invoked[0].object_name, "gato")

    def test_command_validation_rejects_unexpected_types_and_methods(self):
        with self.assertRaises(InvalidCommand):
            ExecuteCommand.validate(
                {"acao": "remover_fundo", "objeto": "gato", "metodo": "shell", "refinar": True}
            )
        with self.assertRaises(InvalidCommand):
            ExecuteCommand.validate(
                {"acao": "remover_fundo", "objeto": "gato", "metodo": "ia", "refinar": "yes"}
            )


if __name__ == "__main__":
    unittest.main()
