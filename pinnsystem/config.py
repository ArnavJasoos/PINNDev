"""Configuration loading and the multi-provider LLM factory.

Provider selection is data-driven: ``config.yaml`` (or env vars) name a provider and
model, and :func:`llm_factory` returns the matching LangChain chat model. Provider
packages are imported lazily so a machine that only has, say, ``langchain-openai``
installed can still run without ``langchain-anthropic`` present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

Provider = Literal["anthropic", "openai", "ollama"]

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "ollama": "llama3.1",
}


class AppConfig(BaseModel):
    """Runtime configuration for the whole system."""

    provider: Provider = "anthropic"
    model: Optional[str] = None
    temperature: float = 0.0

    accuracy_threshold: float = 1e-3
    max_iterations: int = 3
    code_debug_budget: int = 5

    search_backend: Literal["tavily", "serpapi", "duckduckgo", "none"] = "duckduckgo"
    search_api_key: Optional[str] = None

    runs_dir: str = "runs"
    extra: dict[str, Any] = Field(default_factory=dict)

    def resolved_model(self) -> str:
        return self.model or _DEFAULT_MODELS[self.provider]


def _env_override(cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay a few well-known env vars on top of file config."""

    mapping = {
        "PINN_PROVIDER": "provider",
        "PINN_MODEL": "model",
        "PINN_ACCURACY_THRESHOLD": "accuracy_threshold",
        "PINN_MAX_ITERATIONS": "max_iterations",
        "PINN_SEARCH_BACKEND": "search_backend",
        "PINN_SEARCH_API_KEY": "search_api_key",
    }
    for env_key, cfg_key in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            cfg[cfg_key] = val
    return cfg


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load config from a YAML file, overlay env vars, and validate.

    A missing file is fine — defaults + env vars still produce a usable config.
    """

    load_dotenv(override=False)

    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        for candidate in ("config.yaml", "config.example.yaml"):
            p = Path(candidate)
            if p.exists():
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                break

    data = _env_override(data)
    return AppConfig(**data)


def llm_factory(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    *,
    temperature: float = 0.0,
    config: Optional[AppConfig] = None,
    **kwargs: Any,
):
    """Return a LangChain chat model for the requested provider.

    Precedence: explicit args > ``config`` > module defaults. Provider packages are
    imported only when their provider is selected, so unused providers need not be
    installed.
    """

    if config is not None:
        provider = provider or config.provider
        model = model or config.model
        temperature = temperature if temperature is not None else config.temperature

    provider = provider or "anthropic"
    if provider not in _DEFAULT_MODELS:
        raise ValueError(f"Unknown provider: {provider!r}")
    model = model or _DEFAULT_MODELS[provider]

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Provider 'anthropic' requires `pip install langchain-anthropic`."
            ) from exc
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Provider 'openai' requires `pip install langchain-openai`."
            ) from exc
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Provider 'ollama' requires `pip install langchain-ollama`."
            ) from exc
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    raise ValueError(f"Unknown provider: {provider!r}")
