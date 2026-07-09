# PINN Multi-Agent System

A multi-agent [LangGraph](https://langchain-ai.github.io/langgraph/) system that turns
a natural-language physics request into a **trained, validated Physics-Informed Neural
Network**, with a feedback-driven improvement loop, MCP tool servers, and a NiceGUI
front-end. Design is inspired by Lang-PINN (arXiv:2510.05158).

## Status

Built in phases (see `.claude/plan/pinn-multiagent-system.md`, §9 Build Order).

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Skeleton + state (`state.py`, `config.py`, `llm_factory`) | ✅ done |
| 2 | PINN core (solvers, architectures, losses, train, evaluate) | ✅ done |
| 3 | Execution + MCP servers | ✅ done |
| 4 | Agents (parser / research / coding / feedback) | ✅ done |
| 5 | Graph + loops | ⏳ pending |
| 6 | NiceGUI front-end | ⏳ pending |
| 7 | Polish (failure memory, rollback, knowledge base) | ⏳ pending |

## What works now

The PINN core is provable end-to-end without any LLM:

```bash
python -m pytest            # 31 tests: PINN solve, execution, tools, MCP, agents
```

```python
from pinnsystem.pinn.problems import poisson_1d
from pinnsystem.pinn.train import train_pinn
from pinnsystem.pinn.evaluate import evaluate
from pinnsystem.state import HyperParams, SamplingPlan

problem = poisson_1d()                       # u_xx = -pi^2 sin(pi x), u(0)=u(1)=0
result = train_pinn(problem, HyperParams(epochs=3000, lr=5e-3),
                    SamplingPlan(collocation_points=256), boundary_weight=10.0)
report = evaluate(problem, result)
print(report.rel_l2)                         # ~1e-4
```

## Install

```bash
pip install -e .                 # core (state + PINN numerics)
pip install -e ".[agents,gui]"   # once phases 3-6 land
```

Configure via `config.yaml` (copy `config.example.yaml`) or env vars
(`PINN_PROVIDER`, `PINN_MODEL`, `PINN_ACCURACY_THRESHOLD`, ...).

## Layout

```
pinnsystem/
  state.py        # shared PINNState + Pydantic models (single source of truth)
  config.py       # multi-provider llm_factory (anthropic / openai / ollama)
  pinn/           # architectures, solvers, losses, train, evaluate, reference problems
  execution/      # per-run workspaces + isolated subprocess venv runner
  tools/          # pure tool logic: symbolic, offline-safe search, PINN ops, plotting
  mcp/            # 3 stdio FastMCP servers (research / compute / pinn) + client wiring
  knowledge/      # phi(E).psi(A) architecture-matching priors
  agents/         # parser / research / coding / feedback node fns + prompt contracts
tests/            # state, solver, execution, tools, MCP, knowledge, agent, e2e tests
```

MCP servers are launchable standalone (`python -m pinnsystem.mcp.compute_server`);
agents are plain `node(state, llm, ...)` functions behind a tiny structured-LLM
protocol, so they unit-test with a fake LLM (no network).
