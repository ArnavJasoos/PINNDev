"""Shared agent scaffolding: prompt loading and structured LLM invocation.

Agents depend only on the :class:`SupportsStructured` protocol — anything exposing
``with_structured_output(schema).invoke(messages)`` — which every LangChain chat model
satisfies and a test fake can trivially implement. This keeps the langchain/langgraph
packages off the import path for the core node logic and its tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

_PROMPT_DIR = Path(__file__).parent / "prompts"

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class SupportsStructured(Protocol):
    """Minimal interface an LLM must expose for structured-output agent calls."""

    def with_structured_output(self, schema: type[BaseModel]) -> Any:  # pragma: no cover
        ...


def load_prompt(name: str) -> str:
    """Load a versioned prompt template (``prompts/<name>.md``)."""

    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def invoke_structured(
    llm: SupportsStructured,
    schema: type[T],
    system: str,
    human: str,
) -> T:
    """Call ``llm`` for a Pydantic-validated structured result.

    Uses langchain message objects when available, falling back to role/content dicts
    so a lightweight fake needn't import langchain.
    """

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: Any = [SystemMessage(content=system), HumanMessage(content=human)]
    except ImportError:  # pragma: no cover - langchain_core is a core dep
        messages = [{"role": "system", "content": system}, {"role": "user", "content": human}]

    return llm.with_structured_output(schema).invoke(messages)


def contract_header(stage: str, prev: str, nxt: str) -> str:
    """Anti-drift preamble stating this stage's position and I/O contract."""

    return (
        f"You are stage {stage} of the PINN pipeline.\n"
        f"INPUT CONTRACT (guaranteed by the previous stage): {prev}\n"
        f"OUTPUT CONTRACT (the next stage consumes exactly this): {nxt}\n"
        "Emit ONLY the requested schema. Never invent fields or drop required ones."
    )
