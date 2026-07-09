# PINN Multi-Agent System — Test Suite Checklist

What to verify in a fresh session. Nothing here has been run yet in the build session;
all code byte-compiles clean but is otherwise unexercised this session.

## 0. Environment setup

```bash
cd C:/Users/arnav.jade/Documents/PINNDev
pip install -e .                    # core: torch, numpy, scipy, sympy, pydantic, mcp, langchain-core
# Optional stacks (only for the extras-gated checks below):
pip install -e ".[agents,gui]"      # langgraph(+sqlite), langchain-*, mcp adapters, nicegui, matplotlib, plotly
```

Present at build time (already installed): `mcp`, `langchain_core`, `matplotlib`,
`sympy`, `torch`, `numpy`, `scipy`.
Missing at build time (install if testing those paths): `langgraph`,
`langgraph-checkpoint-sqlite`, `langchain` (full), `langchain-mcp-adapters`,
`nicegui`, `plotly`, `duckduckgo_search`.

## 1. Run everything

```bash
python -m pytest -q                 # expect 43 passed
python -m pytest -q --durations=10  # see the slow ones
```

`tests/test_end_to_end_smoke.py` trains a real PINN → **~100–125 s**; it dominates the
run. For a fast inner loop, exclude it:

```bash
python -m pytest -q --ignore=tests/test_end_to_end_smoke.py   # expect 42 passed, ~30–40 s
```

## 2. Per-file coverage map

| Test file | Count | Covers | Notes |
|-----------|-------|--------|-------|
| `test_state.py` | ~? | Pydantic models, `new_state`, config plumbing | Phase 1 (prior) |
| `test_solvers.py` | ~? | reference solvers, architectures, losses | Phase 2 (prior) |
| `test_end_to_end_smoke.py` | 1 | full Poisson/oscillator train+eval convergence | **SLOW ~2 min** |
| `test_execution.py` | 4 | `RunWorkspace` layout; `VenvRunner` stdout/error/timeout | spawns subprocesses |
| `test_tools.py` | 7 | sympy parse/equivalence, offline search, build/train/eval/plot roundtrip, **`fetch_url` scheme reject**, **sympify unsafe-token reject** | train roundtrip trains a small net (~sec) |
| `test_mcp.py` | 2 | 3 servers import + register; client `server_specs` shape | `importorskip` on `mcp` |
| `test_knowledge.py` | 3 | φ(E)·ψ(A) arch match, empty→MLP, forbidden exclusion | pure |
| `test_agents.py` | 7 | parser/research/coding/feedback nodes via **FakeLLM** | coding node runs subprocess writing metrics.json |
| `test_graph_routing.py` | 6 | entry / clarify / feedback / final routing + iteration cap | pure, no langgraph |
| `test_gui_bridge.py` | 4 | `initial_state_from_input`, `event_to_transcript` mapping | pure, no nicegui |
| `test_history.py` | 3 | sqlite `HistoryStore` record/best; `select_best_iteration` | writes tmp sqlite |

Totals: 11 prior + 10 (Phase 3) + 10 (Phase 4) + 6+4+3 (Phase 5–7) = **43**.

## 3. Things to watch / likely failure points (first real run)

These paths were written but never executed — check them first if anything fails:

- **`test_tools.py::test_build_train_evaluate_plot_roundtrip`** — asserts
  `final_loss < 1.0` and a finite `rel_l2`. Training is short (200 epochs); if it
  flakes, bump epochs or loosen the bound. Confirms `HyperParams`/`SamplingPlan`
  kwargs (`epochs`, `width`, `depth`, `collocation_points`) match the real Phase-2
  signatures.
- **`test_agents.py::test_coding_*`** — subprocess runs `sys.executable` on a
  generated script with `cwd=workspace.root`. The good-case script uses **stdlib only**
  (writes `metrics.json`), so no `pinnsystem` import needed in the child. Verify
  `metrics.json` lands in `workspace.root` and `_collect_metrics` finds it.
- **`test_tools.py::test_fetch_url_rejects_non_web_schemes`** — uses `Path.as_uri()`
  (`file://`). Confirms the SSRF/local-file guard.
- **`_GENERATORS` names in `tools/pinn_ops.py`** — `poisson_1d`, `harmonic_oscillator`,
  `finite_difference_poisson` must match actual function names in `pinn/solvers.py`.
  If `build_dataset`/`train_run` raise `KeyError`/`TypeError`, reconcile names + kwargs.
- **`REFERENCE_PROBLEMS` keys** — `train_run`/`evaluate_run` look up `problem_name`
  in `pinn/problems.py::REFERENCE_PROBLEMS`. Confirm `"poisson_1d"` is a key.
- **`build_network` signature** — `pinn_ops` calls
  `build_network(arch, in_dim, out_dim, width=, depth=, activation=, seed=)`. Verify
  it accepts these kwargs (esp. `seed`).
- **`evaluate`/`quality_score` / `EvalReport` fields** — `evaluate_run` reads
  `report.mse/rel_l2/loss_smoothness/convergence_iters/test_inputs/prediction/reference`.
  Confirm attribute names on the real `EvalReport`.

## 4. Extras-gated checks (only after `pip install ".[agents,gui]"`)

Not covered by the 43 unit tests — verify manually once deps are in:

1. **Graph compiles**
   ```python
   from pinnsystem.graph.builder import GraphDeps, build_graph
   from pinnsystem.execution import new_workspace
   from pinnsystem.config import load_config
   # fake or real llm; build_graph(GraphDeps(llm, ws, cfg), checkpoint_path=...)
   ```
   Expect a compiled graph; check nodes `parser/research/coding/feedback/human_clarify/
   human_approve_final` and the conditional edges resolve (names match `routing.*`).
2. **Interrupt/resume** — run `graph.astream(state, {"configurable":{"thread_id":...}})`;
   confirm a `__interrupt__` fires at `human_clarify`, and `Command(resume={"approved":True})`
   advances to `research`.
3. **SQLite checkpointer** — pass `checkpoint_path=runs/checkpoints.sqlite`; confirm
   `SqliteSaver.from_conn_string` is used (falls back to `MemorySaver` if the sqlite
   extra is missing — check which path you hit).
4. **MCP servers launch standalone**
   ```bash
   python -m pinnsystem.mcp.compute_server   # should start a stdio server (Ctrl-C to stop)
   ```
   Then `client.build_client(workdir=...)` should construct a `MultiServerMCPClient`.
5. **GUI** — `python -m pinnsystem` opens the NiceGUI desktop window. Manual:
   - submit a query → transcript streams parser→research→coding→feedback
   - clarify dialog appears; Approve advances; Send-answer appends and re-parses
   - final approval bar shows score/decision; Accept ends, Iterate loops
   - `matplotlib` present → feedback plot renders (else `.npz` fallback)
6. **Live LLM end-to-end** — set `PINN_PROVIDER`/`PINN_MODEL` + API key; run one full
   loop on a real problem (e.g. "solve the 1D Poisson equation u_xx = -pi^2 sin(pi x)").

## 5. Security regression (already unit-tested, re-confirm)

- `test_tools.py::test_fetch_url_rejects_non_web_schemes` — no `file://` read.
- `test_tools.py::test_symbolic_rejects_unsafe_tokens` — dunder/`import` rejected.
- Manual: `run_python(filename="../../evil.py")` and coding module key `"../x.py"`
  must write **inside** the workdir only (basename-stripped). Not auto-tested — spot check.

## 6. Green-bar definition

- `python -m pytest -q` → **43 passed**, 0 failed, 0 error.
- No `ImportError` from core-only install (langgraph/nicegui stay lazy).
- `python -m pinnsystem` without GUI extras → clean install message, exit 1 (not a traceback).
