"""Adaptador entre configuracoes/formatos do Tkinter e o caso de uso."""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
from PIL import Image

from removedor_fundo.application.use_cases.remove_background import RemoveBackground
from removedor_fundo.bootstrap import build_remove_background
from removedor_fundo.domain.models import RemovalRequest
from removedor_fundo.infrastructure.imaging.opencv_adapters import load_image


# ─── Contrato de dados ───
METODOS = {
    "IA (segmentação)": "ia",
    "HSV": "hsv",
    "LAB": "lab",
    "RGB": "rgb",
}

# Qualidade >= este valor liga o refinamento morfológico do back.
QUALIDADE_MINIMA_REFINO = 50


@dataclass
class ResultadoProcessamento:
    imagem: Image.Image
    caminho_saida: str
    tempo_segundos: float
    mensagem: str = ""


# ─── Cancelamento cooperativo ───
_cancel_event = threading.Event()


def cancelar_backend():
    """Pede para o processamento parar na próxima etapa."""
    _cancel_event.set()


def _checar_cancelamento():
    if _cancel_event.is_set():
        raise InterruptedError("Processamento cancelado")


# ─── Conversões ───
def _bgra_para_pil(bgra) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA), "RGBA")


def limiar_da_sensibilidade(sensibilidade: float) -> float:
    """Sensibilidade alta detecta mais => limiar de probabilidade menor."""
    return min(0.95, max(0.05, 1.0 - float(sensibilidade)))


# ─── Chamada principal (substitui o mock do front) ───
def backend_processar(
    caminho_entrada: str,
    modo: str,
    qualidade: int,
    sensibilidade: float,
    callback_progresso: Callable[[int, str], None],
    objeto: Optional[str] = None,
    metodo: str = "ia",
    use_case: Optional[RemoveBackground] = None,
) -> ResultadoProcessamento:
    """
    modo:   "fundo"  -> remove o fundo, mantém o objeto;
            "objeto" -> remove o objeto, mantém o fundo.
    objeto: o que segmentar ("cachorro", "gato"...). Vazio/None = qualquer
            objeto de primeiro plano (só vale para a IA; cores ignoram).
    metodo: "ia", "hsv", "lab" ou "rgb".
    """
    _cancel_event.clear()
    inicio = time.perf_counter()

    if modo not in ("fundo", "objeto"):
        raise ValueError(f"Modo inválido: {modo}")
    if metodo not in METODOS.values():
        raise ValueError(f"Método inválido: {metodo}")

    objeto = (objeto or "").strip() or None

    callback_progresso(10, "Carregando imagem...")
    imagem = load_image(caminho_entrada)
    _checar_cancelamento()

    refinar = int(qualidade) >= QUALIDADE_MINIMA_REFINO
    callback_progresso(
        30,
        "Segmentando..." if metodo != "ia" else "Segmentando com IA (pode demorar na 1ª vez)...",
    )
    parameters = {}
    if metodo == "ia":
        parameters["limiar"] = limiar_da_sensibilidade(sensibilidade)
    resultado = (use_case or _build_remove_use_case()).execute(
        RemovalRequest(
            image=imagem,
            method=metodo,
            object_name=objeto,
            refine=refinar,
            keep_foreground=modo == "fundo",
            parameters=parameters,
        )
    )
    _checar_cancelamento()

    callback_progresso(95, "Finalizando...")
    if modo == "fundo":
        mensagem = "O fundo foi tornado transparente."
    else:
        mensagem = "O objeto foi removido e ficou transparente."

    callback_progresso(100, "Concluído!")
    return ResultadoProcessamento(
        imagem=_bgra_para_pil(resultado.transparent_image),
        caminho_saida="",
        tempo_segundos=time.perf_counter() - inicio,
        mensagem=mensagem,
    )


_remove_background: Optional[RemoveBackground] = None


def _build_remove_use_case() -> RemoveBackground:
    global _remove_background
    if _remove_background is None:
        _remove_background = build_remove_background()
    return _remove_background
