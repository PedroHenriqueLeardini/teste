import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    selected_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, selected_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Nivel de log invalido: {selected_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

