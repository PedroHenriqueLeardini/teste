import os
from typing import Any, Mapping


class GroqResultExplainer:
    """Gera uma explicacao a partir de metadados; nunca recebe dados de imagem."""

    def __init__(self, client: Any = None, model: str = "llama-3.1-8b-instant") -> None:
        if client is None:
            from groq import Groq

            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._client = client
        self._model = os.environ.get("MODELO_LINGUAGEM", model)

    def explain(self, action: str, summary: Mapping[str, Any]) -> str:
        allowed = {"metodo", "refinado"}
        if set(summary) - allowed or any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in summary.values()
        ):
            raise ValueError("A explicacao aceita somente metadados escalares autorizados.")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Explique em portugues, em ate tres frases curtas, "
                        "o resultado informado. Use somente os metadados."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Acao: {action}\nMetadados: {dict(summary)}",
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("O provedor nao retornou uma explicacao.")
        return content
