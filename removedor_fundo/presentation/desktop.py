"""Ponto de entrada da interface Tkinter."""

from Front.remover_frontend_v2 import App
from removedor_fundo.infrastructure.logging_config import configure_logging


def main() -> None:
    configure_logging()
    App().mainloop()
