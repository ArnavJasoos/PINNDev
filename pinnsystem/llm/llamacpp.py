"""llama-cpp-python backend: run a local GGUF model as the agent LLM.

Exposes the same ``with_structured_output(schema).invoke(messages)`` contract the
agents rely on (see :class:`pinnsystem.agents.base.SupportsStructured`). Structured
output is constrained with llama.cpp's JSON-schema grammar so even small local models
return JSON that validates against the requested Pydantic schema.

``llama_cpp`` is imported lazily inside the constructor so importing this module never
requires the optional ``llamacpp`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _to_chat_messages(messages: Any) -> list[dict[str, str]]:
    """Normalise langchain message objects or role/content dicts to llama.cpp chat."""

    role_map = {"system": "system", "human": "user", "ai": "assistant", "user": "user",
                "assistant": "assistant"}
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role", "user"), m.get("content", "")
        else:  # langchain BaseMessage: .type is "system" | "human" | "ai"
            role, content = getattr(m, "type", "user"), getattr(m, "content", "")
        out.append({"role": role_map.get(role, "user"), "content": str(content)})
    return out


class _StructuredBinding:
    """Result of :meth:`LlamaCppStructuredLLM.with_structured_output`."""

    def __init__(self, llm: "LlamaCppStructuredLLM", schema: type[BaseModel]) -> None:
        self._llm = llm
        self._schema = schema

    def invoke(self, messages: Any) -> BaseModel:
        schema_json = self._schema.model_json_schema()
        chat = _to_chat_messages(messages)
        chat.append(
            {
                "role": "system",
                "content": "Respond with a single JSON object matching this schema. "
                "Emit only the JSON, no prose:\n" + json.dumps(schema_json),
            }
        )
        result = self._llm._client.create_chat_completion(
            messages=chat,
            temperature=self._llm.temperature,
            max_tokens=self._llm.max_tokens,
            response_format={"type": "json_object", "schema": schema_json},
        )
        content = result["choices"][0]["message"]["content"]
        return self._schema.model_validate(json.loads(content))


class LlamaCppStructuredLLM:
    """Local GGUF chat model wrapper conforming to the agent LLM contract."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        temperature: float = 0.0,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        max_tokens: int = 2048,
        **llama_kwargs: Any,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "Provider 'llamacpp' requires `pip install llama-cpp-python`."
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {path}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **llama_kwargs,
        )

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredBinding:
        return _StructuredBinding(self, schema)

    def invoke(self, messages: Any) -> str:
        """Plain-text completion (used for non-structured calls, if any)."""

        result = self._client.create_chat_completion(
            messages=_to_chat_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return result["choices"][0]["message"]["content"]
