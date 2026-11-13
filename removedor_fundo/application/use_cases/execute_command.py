from typing import Any, Callable, Mapping, Optional

from ...domain.contracts import CommandInterpreter
from ...domain.errors import InvalidCommand
from ...domain.models import UserCommand


class ExecuteCommand:
    """Converte texto em comando e despacha apenas acoes registradas localmente."""

    ACTIONS = frozenset({"remover_fundo", "comparar_metodos"})
    METHODS = frozenset({"ia", "rgb", "hsv", "lab"})

    def __init__(
        self,
        interpreter: CommandInterpreter,
        handlers: Mapping[str, Callable[..., Any]],
    ) -> None:
        self._interpreter = interpreter
        self._handlers = dict(handlers)

    def execute(self, text: str, **context: Any) -> Any:
        return self.dispatch(self.interpret(text), **context)

    def interpret(self, text: str) -> UserCommand:
        return self.validate(self._interpreter.interpret(text))

    def dispatch(self, command: UserCommand, **context: Any) -> Any:
        handler = self._handlers.get(command.action)
        if command.action not in self.ACTIONS or handler is None:
            raise InvalidCommand(f"Acao nao autorizada: {command.action}")
        return handler(command=command, **context)

    @classmethod
    def validate(cls, raw: Mapping[str, Any]) -> UserCommand:
        if not isinstance(raw, Mapping):
            raise InvalidCommand("O interpretador deve retornar um objeto de comando.")
        unexpected = set(raw) - {"acao", "objeto", "metodo", "refinar"}
        if unexpected:
            raise InvalidCommand(
                f"Parametros nao autorizados: {', '.join(sorted(unexpected))}"
            )
        action = raw.get("acao")
        method = raw.get("metodo", "ia")
        object_name = raw.get("objeto")
        refine = raw.get("refinar", True)

        if not isinstance(action, str) or action not in cls.ACTIONS:
            raise InvalidCommand("A acao solicitada nao e permitida.")
        if not isinstance(method, str) or method not in cls.METHODS:
            raise InvalidCommand(f"Metodo nao autorizado: {method}")
        if object_name is not None and not isinstance(object_name, str):
            raise InvalidCommand("O objeto deve ser informado como texto.")
        if not isinstance(refine, bool):
            raise InvalidCommand("O parametro 'refinar' deve ser booleano.")
        if action == "remover_fundo" and method == "ia" and not object_name:
            raise InvalidCommand("Informe o objeto que deve ser mantido.")

        return UserCommand(
            action=action,
            object_name=object_name.strip() if object_name else None,
            method=method,
            refine=refine,
        )
