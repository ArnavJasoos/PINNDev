"""The four agents as LangGraph-ready node functions.

Each agent is a plain ``node(state, ...) -> partial_state`` function that keeps its
LLM dependency behind the small :class:`~pinnsystem.agents.base.SupportsStructured`
protocol, so nodes are unit-testable with a fake structured LLM and the graph wiring
(Phase 5) stays a thin orchestration layer on top.
"""

from .coding import coding_node
from .feedback import feedback_node
from .parser import parser_node
from .research import research_node

__all__ = ["parser_node", "research_node", "coding_node", "feedback_node"]
