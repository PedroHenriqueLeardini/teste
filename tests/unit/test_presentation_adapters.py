import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import pipeline
from Front.backend_bridge import backend_processar


class PresentationAdapterTests(unittest.TestCase):
    def _write_test_image(self, path):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = (30, 80, 150)
        self.assertTrue(cv2.imwrite(str(path), image))

    def test_tkinter_bridge_returns_preview_without_evaluation_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.png"
            self._write_test_image(path)
            progress = []

            result = backend_processar(
                str(path),
                "fundo",
                qualidade=10,
                sensibilidade=0.5,
                callback_progresso=lambda percent, message: progress.append(percent),
                objeto="animal",
                metodo="rgb",
            )

        self.assertEqual(result.imagem.mode, "RGBA")
        self.assertFalse(hasattr(result, "metricas"))
        self.assertEqual(progress[-1], 100)

    def test_local_command_flow_processes_an_image_without_trimaps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "input.png"
            self._write_test_image(image_path)
            image = cv2.imread(str(image_path))

            result = pipeline.executar_pedido(
                image,
                "remova o fundo usando rgb",
                pasta_saida=str(Path(temporary_directory) / "results"),
                usar_llm=False,
            )

        self.assertEqual(result["status"], "ok")
        self.assertNotIn("metricas", result)


if __name__ == "__main__":
    unittest.main()
