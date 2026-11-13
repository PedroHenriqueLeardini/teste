import re
from typing import Any, Mapping

from removedor_fundo.infrastructure.segmentation.labels import OBJETOS_PT, normalize_text


class RuleBasedCommandInterpreter:
    """Interpretador local deterministico para uso sem credenciais da Groq."""

    def interpret(self, text: str) -> Mapping[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("O comando deve ser um texto nao vazio.")
        normalized = normalize_text(text)
        words = set(re.findall(r"[a-z]+", normalized))

        if "compar" in normalized:
            action = "comparar_metodos"
        elif any(
            word in normalized
            for word in ("remov", "tire", "tirar", "recort", "apag", "isol", "fundo", "transparent")
        ):
            action = "remover_fundo"
        else:
            action = "desconhecida"

        object_name = next((name for name in OBJETOS_PT if name in words), None)
        method = next((name for name in ("hsv", "lab", "rgb") if name in words), "ia")
        refine = not ("sem refin" in normalized or "sem morfolog" in normalized)
        return {
            "acao": action,
            "objeto": object_name,
            "metodo": method,
            "refinar": refine,
        }

