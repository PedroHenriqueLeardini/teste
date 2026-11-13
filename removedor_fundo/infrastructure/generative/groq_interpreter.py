import json
import os
from typing import Any, Mapping


class GroqCommandInterpreter:
    """Interpreta texto; nao recebe nem transmite imagens."""

    def __init__(self, client: Any = None, model: str = "llama-3.1-8b-instant") -> None:
        if client is None:
            from groq import Groq

            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._client = client
        self._model = os.environ.get("MODELO_LINGUAGEM", model)

    def interpret(self, text: str) -> Mapping[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("O comando deve ser um texto nao vazio.")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Interprete pedidos de remocao de fundo. Retorne somente "
                        "JSON com acao (remover_fundo ou comparar_metodos), "
                        "objeto, metodo (ia/rgb/hsv/lab) e refinar (booleano). "
                        "Nao solicite nem manipule pixels."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("O provedor nao retornou um comando.")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("O comando retornado deve ser um objeto JSON.")
        return parsed
