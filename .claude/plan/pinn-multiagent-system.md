# Implementation Plan: PINN Multi-Agent System (Lang-PINN–inspired)

> Multi-agent LangGraph system that turns a natural-language physics request into a
> trained, validated Physics-Informed Neural Network, with a feedback-driven improvement
> loop, MCP tool servers, and an aesthetic NiceGUI front-end.

## Task Type
- [x] Fullstack (agent backend + GUI)
- Backend: LangGraph orchestration, agents, MCP tool servers, PINN training
- Frontend: NiceGUI desktop-runnable app

## Locked Decisions (from user)
| Decision | Choice |
|----------|--------|
| GUI framework | **NiceGUI** (native async, streams LangGraph events, embeds Plotly) |
| LLM provider | **Configurable/multi** — Anthropic + OpenAI + Ollama, selected by env/runtime |
| Code execution | **Local subprocess venv** (isolated interpreter, no PowerShell dependency) |
| Language | Python only, launchable via `python -m pinnsystem` |

---

## 1. How the paper maps to the user's 4 agents

The user specifies 4 agents; Lang-PINN (paper) also has 4 but split differently. Mapping:

| User agent | Paper agent(s) | What we take from the paper |
|-----------|----------------|------------------------------|
| **1. NL→Problem Statement parser** | PDE Agent (part) | K-sample CoT trajectories → normalized descriptions → candidate PDEs → **consensus voting** (symbolic + semantic). We ADD the user's interactive clarification loop (keep questioning until user accepts). Optional multimodal/OCR. |
| **2. Research agent** | PDE Agent (formalization) + **PINN Agent** (architecture selection) | Knowledge-guided architecture matching: PDE feature vector φ(E)=[periodicity, geometry complexity, multi-scale demand] vs architecture capability ψ(A), weighted cosine similarity, `A* = argmax`. History reuse cache H + knowledge base K. Research also picks loss terms, sampling, hyperparameters. **Failure memory** so it never re-proposes a failed approach. |
| **3. Coding agent** | **Code Agent** | **Modular** generation (6 modules: model / pde_loss / data / train_loop / validation / main), interface contracts between modules, **PDE-loss symbolic verification** (parse generated loss back to a PDE, check equivalence with Research's E). Internal debug loop until scripts run clean. |
| **4. Feedback agent** | **Feedback Agent** | Error localization → attribute failure to a module → ask Coding agent to regenerate only that module; escalate upstream (Research) when the problem is architectural. Multi-dimensional quality score S(C)=Σ wᵢ·m̂ᵢ over effectiveness (MSE), efficiency (convergence/params), robustness (loss smoothness / gradient health). **Accept-or-rollback** by comparing S(Cᵗ) vs S(Cᵗ⁻¹). |

### User-specific requirements NOT in the paper (must build)
1. **Hard-computed datasets**: numerically solve the ODE/PDE to produce train/test data (paper reuses PINNacle data). Solvers chosen per problem: `scipy.integrate.solve_ivp` (ODE/IVP), finite-difference or `py-pde` (PDEs), analytical solution when known.
2. **Data-dependency branching** (entry routing):
   - User gives **dataset** → skip data-generation, parse problem against data, go straight to training.
   - User gives **formulas + procedures** → Research skips physics research, researches only training/architecture.
   - User gives **both** → Research does only training research; Coding starts from given data directly.
3. **Dual exit condition**: `user approval` **OR** `accuracy ≥ threshold`.
4. **Explicit shared state / role awareness**: every agent knows prev+next stage and reads/writes one typed state object (anti-drift).
5. **MCP tool servers** consumed by LangGraph agents.
6. **Aesthetic NiceGUI** with live agent transcript, clarification prompts, approval buttons, and result graphs.

---

## 2. Architecture Overview

```
                         ┌────────────────────────── NiceGUI (async) ───────────────────────────┐
                         │  chat transcript · clarification prompts · approval btns · Plotly plots │
                         └───────────────▲───────────────────────────────┬───────────────────────┘
                                         │ stream events (astream)        │ user input / approvals
                                         │                                 ▼
   ┌─────────────────────────────────  LangGraph StateGraph  ─────────────────────────────────┐
   │                                                                                            │
   │  [entry router] ──▶ Parser ──(interrupt: clarify/approve)──▶ Research ──▶ Coding ──▶ Feedback
   │        │ (dataset/formulas flags)                 ▲                          │        │      │
   │        └── skip branches ──────────────────────────┘         internal debug loop│        │      │
   │                                                                                  │        │      │
   │                        feedback verdict ── revert-to-Research ◀──────────────────┘◀───────┘      │
   │                                       └── accept / user-approval ──▶ END                          │
   └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                         │ MCP client (langchain-mcp-adapters)
             ┌───────────────────────────┼───────────────────────────────────────┐
             ▼                           ▼                                         ▼
   MCP: research_tools           MCP: compute_tools                       MCP: pinn_tools
   (web_search, fetch,           (run_python venv subprocess,             (build_dataset, train_pinn,
    sympy_parse, arxiv)           write_file, read_file, symbolic_check)   evaluate, plot_results)
```

- **Orchestration**: one `StateGraph` with a shared `PINNState` (TypedDict). Nodes = agents. Conditional edges implement the loops and skip-branches. `interrupt()` (LangGraph human-in-the-loop) drives clarification & approval through the GUI. A `checkpointer` (SQLite) persists state so the GUI can resume across interrupts.
- **LLM abstraction**: `llm_factory(provider, model)` returns a LangChain chat model (`langchain-anthropic` / `langchain-openai` / `langchain-ollama`). Provider from `config.yaml`/env.
- **MCP**: three MCP servers (stdio) exposing tools; agents load them via `langchain-mcp-adapters` `MultiServerMCPClient`. Keeps tool logic decoupled and reusable (satisfies "LangGraph MCP with appropriate tools").
- **Anti-drift**: each agent prompt is templated with (a) its role, (b) the previous stage's output contract, (c) the next stage's expected input contract, (d) the relevant slice of shared state only.

---

## 3. Shared State Schema (`PINNState`)

Single source of truth passed between all agents. Pydantic models serialized into the LangGraph state.

```python
class ProblemSpec(BaseModel):
    raw_query: str
    normalized_statement: str          # parser output, user-approved
    pde: SymbolicPDE | None            # operators, coeffs, domain, BC/IC
    domain: DomainSpec                  # bounds, dims, geometry flags
    quantities: list[str]
    user_provided_formulas: bool
    user_provided_dataset: bool
    dataset_path: str | None
    approved_by_user: bool

class SymbolicPDE(BaseModel):
    latex: str
    sympy_src: str                      # reconstructable expression
    operators: list[str]                # e.g. ["d/dt","d2/dx2","nonlinear_prod"]
    boundary_conditions: list[str]
    initial_conditions: list[str]
    feature_vector: dict                # periodicity, geometry_complexity, multiscale

class ResearchReport(BaseModel):
    architecture: str                   # MLP | Fourier-MLP | CNN | GNN | Transformer | SIREN
    arch_rationale: str                 # from φ(E)·ψ(A) matching
    loss_terms: list[LossTerm]          # residual, bc, ic, data
    sampling: SamplingPlan              # collocation counts, adaptive?
    hyperparams: HyperParams            # width, depth, lr, epochs, optimizer, act fn
    data_generation: DataGenPlan        # solver lib + params, or "user_dataset"
    citations: list[str]
    forbidden_approaches: list[str]     # failure memory (never reuse)

class CodeArtifacts(BaseModel):
    modules: dict[str,str]              # module_name -> filepath
    dataset_path: str
    model_path: str
    metrics_path: str
    last_run_stdout: str
    last_run_error: str | None

class FeedbackVerdict(BaseModel):
    quality_score: float                # S(C) weighted sum
    metrics: QualityMetrics             # mse, rel_l2, convergence_iters, loss_smoothness
    passed_threshold: bool
    decision: Literal["accept","revise_code","revert_research","await_user"]
    faulty_module: str | None           # error localization target
    directive: str                      # actionable instruction to the target agent
    plots: list[str]

class PINNState(TypedDict):
    spec: ProblemSpec
    research: ResearchReport | None
    code: CodeArtifacts | None
    feedback: FeedbackVerdict | None
    history: list[dict]                 # per-iteration (research+code+score) for rollback & memory
    iteration: int
    max_iterations: int                 # cap (paper uses 3)
    accuracy_threshold: float
    messages: list                      # transcript for GUI
    pending_user_action: str | None     # "clarify" | "approve_statement" | "approve_final"
```

---

## 4. LangGraph Graph Definition

Nodes and edges:

```
add_node: entry_router, parser, research, coding, feedback
add_node: human_clarify (interrupt), human_approve_final (interrupt)

entry:            START -> entry_router
entry_router:     conditional ->
                    - "need_parse"      -> parser        (no dataset, no formulas)
                    - "have_formulas"   -> research      (formulas given; skip physics research inside)
                    - "have_dataset"    -> parser(light) -> research(train-only)
                    - "have_both"       -> research(train-only, points to given data) -> coding

parser:           -> human_clarify   (interrupt whenever pending question)
human_clarify:    conditional -> parser (more Qs) | research (statement approved)
research:         -> coding
coding:           self-loop until scripts execute clean OR internal_debug_budget hit -> feedback
feedback:         conditional ->
                    - "accept" & !user_gate      -> human_approve_final
                    - "accept" & threshold_only  -> END
                    - "revise_code"              -> coding   (regenerate faulty_module only)
                    - "revert_research"          -> research (with forbidden_approaches updated)
                    - "await_user"               -> human_approve_final
                    - iteration>=max_iterations  -> human_approve_final (report best-so-far)
human_approve_final: conditional -> END (approved) | research (user wants better)
```

Loop-engineering details:
- **Coding internal loop** = paper's self-debug: run script → on error, LLM patches the specific module → re-run, up to `code_debug_budget` (e.g. 5) tries; then hand error to Feedback for module attribution.
- **Outer loop** = Feedback → Research/Coding, capped by `max_iterations` (default 3, configurable).
- **Failure memory**: before each Research revision, append the failed `(architecture, loss, hyperparams, score)` to `forbidden_approaches`; Research prompt is instructed to avoid them.
- **Rollback**: if `S(Cᵗ) < S(Cᵗ⁻¹)`, restore the previous `CodeArtifacts`/`ResearchReport` from `history` before next attempt (paper's accept/revert).

---

## 5. Agents — responsibilities, I/O contract, prompt strategy

Each agent = a prompt template + bound MCP tools + a structured-output parser (Pydantic). Prompts explicitly state the **input contract** (what the previous stage guarantees) and **output contract** (exact schema the next stage consumes) to prevent context loss/drift.

### 5.1 Parser Agent (NL → Problem Statement)
- **Tools**: `sympy_parse`, (optional) `ocr_image`, `web_search` (term disambiguation).
- **Logic**: sample K CoT trajectories → normalize → candidate PDEs → consensus vote (symbolic equivalence via Sympy AST score + semantic similarity via embeddings) → pick canonical PDE. Then **clarify loop**: list every assumption; ask user via `interrupt`; do not proceed until `approved_by_user=True`.
- **Output**: `ProblemSpec` (+ `SymbolicPDE`).
- **Prompt anchors**: "You are stage 1 of 4. Next stage (Research) needs a canonical PDE + domain + BC/IC. Never assume; enumerate assumptions and ask. Emit ONLY the ProblemSpec schema."

### 5.2 Research Agent (Reason)
- **Tools**: `web_search`, `fetch_url`, `arxiv_search`, `sympy_parse`, `read_file` (to read Coding's scripts during revert).
- **Logic**: knowledge-guided architecture matching (φ(E)·ψ(A) weighted cosine, prioritize multi-scale > geometry > periodicity), history-cache reuse, choose loss terms/sampling/hyperparams, choose data-gen solver (or point to user dataset). On revert: read failed scripts + `forbidden_approaches`, propose a *different* approach.
- **Output**: `ResearchReport` (self-contained enough for Coding to build everything from it alone).
- **Prompt anchors**: "Input contract: ProblemSpec with canonical PDE. Output contract: ResearchReport that a coder can implement with zero extra research. If reverted, you MUST change approach vs forbidden_approaches."

### 5.3 Coding Agent (Action)
- **Tools**: `write_file`, `read_file`, `run_python` (venv subprocess), `symbolic_check` (verify PDE loss), `web_search` (only on data loss).
- **Logic**: emit 6 modules honoring interface contracts → verify `pde_loss` symbolically against `SymbolicPDE` → run `build_dataset` → run `train_pinn` → run `evaluate` → collect metrics. Internal debug loop on any failure (regenerate only the faulty module).
- **Output**: `CodeArtifacts` (+ metrics json).
- **Prompt anchors**: "Generate modules independently against these interfaces {…}. Do not couple modules. pde_loss must pass symbolic_check against the provided PDE."

### 5.4 Feedback Agent (Observe)
- **Tools**: `read_file`, `plot_results`, `evaluate`.
- **Logic**: if run failed → localize faulty module from stderr, set `decision=revise_code` or escalate `revert_research`. If succeeded → compute S(C), compare to previous, accept/rollback, decide threshold pass, generate comparison plots (pred vs ground-truth), route to user approval or END.
- **Output**: `FeedbackVerdict` (+ plot paths).
- **Prompt anchors**: "You decide loop continuation. Map errors to exactly one module. Never rewrite code yourself — emit a directive for Coding or Research."

---

## 6. MCP Tool Servers

Three stdio MCP servers (FastMCP), loaded by `MultiServerMCPClient`:

| Server | Tools | Notes |
|--------|-------|-------|
| `research_tools` | `web_search`, `fetch_url`, `arxiv_search`, `sympy_parse`, `symbolic_equivalence` | web via configured search API; sympy for AST + equivalence score |
| `compute_tools` | `run_python(code, timeout)`, `write_file`, `read_file`, `list_workdir` | `run_python` spawns the isolated venv interpreter as a subprocess with cwd = per-run workdir, timeout + captured stdout/stderr. **No shell; direct `subprocess.run([venv_python, script])`.** |
| `pinn_tools` | `build_dataset(plan)`, `train_pinn(config)`, `evaluate(model,data)`, `plot_results` | thin wrappers that generate+invoke the standardized scripts; also usable directly for the skip-branches |

Design note: agents mostly call `run_python` with generated scripts (paper's modular codegen), while `pinn_tools` provide deterministic fallbacks and the graphing used by Feedback.

---

## 7. Project Structure

```
PINNDev/
├─ pyproject.toml                 # deps, entry point `pinnsystem`
├─ README.md
├─ config.example.yaml            # provider, model, threshold, max_iterations, search api key
├─ pinnsystem/
│  ├─ __main__.py                 # launches NiceGUI app
│  ├─ config.py                   # load env/yaml, llm_factory
│  ├─ state.py                    # PINNState + Pydantic models (§3)
│  ├─ graph.py                    # StateGraph wiring (§4), checkpointer
│  ├─ agents/
│  │  ├─ parser.py  research.py  coding.py  feedback.py
│  │  └─ prompts/  *.md           # versioned prompt templates w/ I/O contracts
│  ├─ mcp/
│  │  ├─ research_server.py  compute_server.py  pinn_server.py
│  │  └─ client.py                # MultiServerMCPClient setup
│  ├─ pinn/
│  │  ├─ solvers.py               # scipy/py-pde/analytical data generation
│  │  ├─ architectures.py         # MLP, Fourier-MLP/SIREN, CNN, GNN, Transformer
│  │  ├─ losses.py                # residual/bc/ic/data loss builders (autograd)
│  │  ├─ train.py  evaluate.py    # standardized training/validation
│  │  └─ interfaces.py            # module I/O contracts (dataclasses)
│  ├─ execution/
│  │  ├─ venv_runner.py           # create/reuse isolated venv, run scripts
│  │  └─ workdir.py               # per-run scratch dirs, artifact paths
│  ├─ knowledge/
│  │  ├─ arch_knowledge.py        # ψ(A) capability priors (knowledge base K)
│  │  └─ history.py               # PDE→arch outcome cache H (sqlite)
│  └─ gui/
│     ├─ app.py                   # NiceGUI layout, theme, async event pump
│     ├─ components.py            # transcript, clarify dialog, approval bar, plot panel
│     └─ bridge.py                # graph.astream ↔ UI; interrupt resume
├─ runs/                          # per-session artifacts (datasets, models, plots, metrics)
└─ tests/
   ├─ test_state.py test_graph_routing.py test_solvers.py test_symbolic.py
   └─ test_end_to_end_smoke.py    # 1D ODE happy-path, mocked LLM
```

---

## 8. Key Files (build targets)

| File | Operation | Description |
|------|-----------|-------------|
| `pinnsystem/state.py` | Create | Pydantic + TypedDict state (§3) — build first, everything depends on it |
| `pinnsystem/config.py` | Create | multi-provider `llm_factory`, config load |
| `pinnsystem/mcp/*_server.py` | Create | 3 MCP servers (§6) |
| `pinnsystem/pinn/*` | Create | solvers, architectures, losses, train/eval, interfaces |
| `pinnsystem/execution/venv_runner.py` | Create | isolated subprocess execution |
| `pinnsystem/agents/*.py` + prompts | Create | 4 agents w/ structured output + contracts (§5) |
| `pinnsystem/graph.py` | Create | StateGraph, conditional edges, interrupts, checkpointer (§4) |
| `pinnsystem/gui/*` | Create | NiceGUI app + astream bridge |
| `tests/*` | Create | routing + solver + symbolic + smoke tests |

---

## 9. Build Order (phased, each phase runnable/testable)

1. **Skeleton + state** — `pyproject`, `config`, `state.py`, `llm_factory`. Test: models validate, providers instantiate (mocked).
2. **PINN core (no agents)** — `solvers`, `architectures`, `losses`, `train`, `evaluate`. Prove a hand-written 1D problem (e.g. damped harmonic oscillator ODE or 1D Poisson) trains to threshold end-to-end. De-risks the hardest technical piece before wiring LLMs.
3. **Execution + MCP** — `venv_runner`, 3 MCP servers, client. Test tools standalone.
4. **Agents** — parser→research→coding→feedback, each with contract prompts + structured output; unit-test with mocked LLM + real tools.
5. **Graph + loops** — wire StateGraph, conditional routing, skip-branches, interrupts, checkpointer. Test routing table with a fake LLM.
6. **GUI** — NiceGUI layout, astream bridge, clarify/approval interrupts, live plots. 
7. **Polish** — failure-memory, rollback, knowledge base K + history H, config surface, README, packaging (`python -m pinnsystem`).

---

## 10. Dependencies (pyproject)
`langgraph`, `langchain`, `langchain-anthropic`, `langchain-openai`, `langchain-ollama`, `langchain-mcp-adapters`, `mcp` (FastMCP), `pydantic>=2`, `torch`, `deepxde` (optional/reference), `numpy`, `scipy`, `sympy`, `py-pde`, `plotly`, `nicegui`, `pyyaml`, `python-dotenv`, `pytest`. Optional multimodal: `pillow`, `pytesseract`.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Generated code executing on host (local venv) | Dedicated venv + per-run workdir, subprocess timeout, no shell string exec, package surface limited to installed venv; document that Coding runs model-written code |
| PINN training is slow / non-convergent (gradient pathologies noted in paper) | Start with cheap 1D problems for smoke tests; Feedback's robustness metric (loss smoothness) catches pathologies; iteration cap + best-so-far reporting |
| LLM structured-output drift breaks contracts | Pydantic-validated structured output + retry-on-parse-fail; explicit I/O contracts embedded in every prompt |
| NiceGUI ↔ LangGraph `interrupt` resume complexity | Use LangGraph SQLite checkpointer + thread_id per session; GUI resumes via `Command(resume=...)`; isolate in `gui/bridge.py` |
| Web search API/key variability | Abstract search behind one tool; support a pluggable backend (Tavily/SerpAPI/DuckDuckGo) via config; degrade gracefully when offline (formulas-provided branch) |
| Multi-provider quirks (tool-calling differs) | Route tool-use through LangChain's unified bind_tools; feature-flag providers lacking reliable tool calling (e.g. some Ollama models) |
| Symbolic PDE-loss verification false negatives (paper notes strict symbolic checks over-reject) | Pair symbolic equivalence with semantic check + tolerance, mirroring the paper's composite score |

---

## 12. Success Criteria
- `python -m pinnsystem` launches the NiceGUI app with no PowerShell scripts.
- A plain-language request (e.g. "solve a 1D heat equation on [0,1]") flows parser→research→coding→feedback and yields a trained model + pred-vs-truth plot.
- Loop demonstrably revises on low accuracy and exits on threshold **or** user approval.
- Skip-branches work: providing a dataset skips data-gen; providing formulas skips physics research.
- Provider switch (Anthropic↔OpenAI↔Ollama) via config only.

---

## SESSION_ID (for /ccg:execute)
- CODEX_SESSION: n/a (codeagent-wrapper not installed on this machine)
- GEMINI_SESSION: n/a
- Planning done Claude-only; no external-model sessions to resume.
