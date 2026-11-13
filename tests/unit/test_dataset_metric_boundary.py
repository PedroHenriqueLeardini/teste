import unittest

import numpy as np

import pipeline
import tools


class DatasetMetricBoundaryTests(unittest.TestCase):
    def test_comparison_without_trimap_does_not_calculate_evaluation_metrics(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        results = pipeline.executar_experimento(image, metodos=["rgb"])

        self.assertIn("RGB", results)
        self.assertIsNone(results["RGB"]["metricas"])
        self.assertIsNone(results["RGB"]["alta_freq"])

    def test_user_tool_rejects_trimap_evaluation(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        trimap = np.ones((16, 16), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "exclusivas dos experimentos"):
            tools.comparar_metodos(image, trimap=trimap)

    def test_user_command_flow_rejects_dataset_evaluation(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        trimap = np.ones((16, 16), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "somente no fluxo de experimentos"):
            pipeline.executar_pedido(
                image, "compare métodos RGB", trimap=trimap, usar_llm=False
            )


if __name__ == "__main__":
    unittest.main()
