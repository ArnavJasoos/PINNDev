"""NiceGUI application entry point (``python -m pinnsystem``).

Builds the layout, wires the input form to a :class:`~pinnsystem.gui.bridge.PinnRunner`,
and streams LangGraph events into the transcript, surfacing interrupts as dialogs.
nicegui/langgraph are imported here (lazily via :func:`run`) so the package imports
without the GUI extra.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import AppConfig, llm_factory, load_config
from ..execution import VenvRunner, new_workspace
from ..graph.builder import GraphDeps, build_graph
from .bridge import PinnRunner, initial_state_from_input


def _build_runner(config: AppConfig) -> PinnRunner:
    llm = llm_factory(config=config)
    workspace = new_workspace(config.runs_dir)
    deps = GraphDeps(llm=llm, workspace=workspace, config=config, runner=VenvRunner())
    graph = build_graph(deps, checkpoint_path=f"{config.runs_dir}/checkpoints.sqlite")
    return PinnRunner(graph=graph, thread_id=workspace.run_id)


def run(config: Optional[AppConfig] = None) -> int:
    """Launch the NiceGUI desktop app."""

    from nicegui import ui

    cfg = config or load_config()

    ui.colors(primary="#6d5dfa", secondary="#22d3ee")
    ui.query("body").style("background: linear-gradient(135deg,#0f172a,#1e1b4b)")

    state: dict[str, Any] = {"runner": None}

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-3"):
        ui.label("PINN Multi-Agent System").classes("text-h4 text-white")
        ui.label(
            f"provider={cfg.provider} · model={cfg.resolved_model()} · threshold={cfg.accuracy_threshold}"
        ).classes("text-caption text-grey-4")

        query = ui.textarea("Describe the physics problem").classes("w-full").props("dark outlined")
        with ui.row().classes("w-full items-center gap-3"):
            dataset = ui.input("Dataset path (optional)").props("dark outlined dense")
            formulas = ui.checkbox("Formulas provided")
        transcript = ui.column().classes("w-full gap-2")

        async def start() -> None:
            from .components import approval_bar, clarify_dialog, transcript_entry

            transcript.clear()
            runner = _build_runner(cfg)
            state["runner"] = runner
            init = initial_state_from_input(
                query.value or "",
                dataset_path=dataset.value or None,
                formulas_given=formulas.value,
                accuracy_threshold=cfg.accuracy_threshold,
                max_iterations=cfg.max_iterations,
            )

            async def pump(payload: Any) -> None:
                async for item in runner.stream(payload):
                    if "interrupt" in item:
                        intr = item["interrupt"]
                        data = intr[0].value if isinstance(intr, (list, tuple)) else intr
                        cb = lambda p: ui.timer(  # noqa: E731 - inline resume
                            0.01, lambda: pump(runner.resume_command(p)), once=True
                        )
                        if data.get("type") == "approve_final":
                            approval_bar(data, cb)
                        else:
                            clarify_dialog(data, cb)
                        return
                    with transcript:
                        transcript_entry(item)

            await pump(init)

        ui.button("Run", on_click=start).props("color=primary size=lg").classes("w-full")

    ui.run(native=True, title="PINN Multi-Agent System", reload=False)
    return 0
