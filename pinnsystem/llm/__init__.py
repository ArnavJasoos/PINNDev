"""Local / self-hosted LLM backends that satisfy the agent ``SupportsStructured``
contract without a cloud provider."""

from .llamacpp import LlamaCppStructuredLLM

__all__ = ["LlamaCppStructuredLLM"]
