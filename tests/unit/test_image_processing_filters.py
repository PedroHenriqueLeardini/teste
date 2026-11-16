"""
Testes unitários mínimos para as novas funções de filtragem de frequência (LPF/HPF) em image_processing.py.
"""
import unittest
import numpy as np
from image_processing import aplicar_lpf, aplicar_hpf, visualizar_perfil_frequencia


def _create_test_image_bgr(shape=(64, 64), dtype=np.uint8):
    """Criar uma imagem BGR sintética para teste."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, size=shape + (3,), dtype=dtype)


class TestImageProcessingFilters(unittest.TestCase):
    def test_aplicar_lpf_imagem_bgr(self):
        bgr = _create_test_image_bgr()
        resultado = aplicar_lpf(bgr, frequencia_corte=0.2)
        self.assertEqual(resultado.shape, bgr.shape)
        self.assertEqual(resultado.dtype, bgr.dtype)

    def test_aplicar_hpf_imagem_bgr(self):
        bgr = _create_test_image_bgr()
        resultado = aplicar_hpf(bgr, frequencia_corte=0.2)
        self.assertEqual(resultado.shape, bgr.shape)
        self.assertEqual(resultado.dtype, bgr.dtype)

    def test_visualizar_perfil_frequencia(self):
        bgr = _create_test_image_bgr()
        freq, energia = visualizar_perfil_frequencia(bgr, mostrar=False)
        self.assertIsInstance(freq, np.ndarray)
        self.assertIsInstance(energia, np.ndarray)
        self.assertGreater(len(freq), 0)
        self.assertGreater(len(energia), 0)

    def test_frequencia_corte_invalida(self):
        bgr = _create_test_image_bgr()
        with self.assertRaises(ValueError):
            aplicar_lpf(bgr, frequencia_corte=0.0)
        with self.assertRaises(ValueError):
            aplicar_lpf(bgr, frequencia_corte=1.0)

    def test_aplicar_lpf_imagem_inválida(self):
        with self.assertRaises(ValueError):
            aplicar_lpf(None, 0.1)

if __name__ == "__main__":
    unittest.main()