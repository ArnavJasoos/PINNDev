"""NiceGUI application entry point (``python -m pinnsystem``).

Builds a two-pane layout — a session sidebar and a run panel — wires the input form
and the follow-up chat to a :class:`~pinnsystem.gui.bridge.PinnRunner`, and streams
LangGraph events into the transcript, surfacing interrupts as dialogs. Sessions persist
via :class:`~pinnsystem.gui.sessions.SessionStore`; each keeps its own checkpoint
thread so a follow-up can resume it. nicegui/langgraph are imported lazily in
:func:`run` so the package imports without the GUI extra.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..config import AppConfig, llm_factory, load_config
from ..execution import VenvRunner, new_workspace
from ..graph.builder import GraphDeps, build_graph
from .bridge import PinnRunner, initial_state_from_input
from .sessions import SessionStore


def _runner_for(config: AppConfig, *, session_id: Optional[str] = None) -> PinnRunner:
    """Build a runner. New session → fresh workspace; existing → reuse its dirs+thread.

    The checkpoint DB is shared across sessions (keyed by ``thread_id``), so reusing a
    session's run id resumes its full graph state.
    """

    llm = llm_factory(config=config)
    workspace = new_workspace(config.runs_dir, run_id=session_id)
    deps = GraphDeps(llm=llm, workspace=workspace, config=config, runner=VenvRunner())
    graph = build_graph(deps, checkpoint_path=f"{config.runs_dir}/checkpoints.sqlite")
    return PinnRunner(graph=graph, thread_id=workspace.run_id)


def run(config: Optional[AppConfig] = None) -> int:
    """Launch the NiceGUI desktop app."""

    from nicegui import ui

    from .components import approval_bar, clarify_dialog, transcript_entry

    cfg = config or load_config()
    store = SessionStore(cfg.runs_dir)

    ui.colors(primary="#6d5dfa", secondary="#22d3ee")
    ui.query("body").style("background: linear-gradient(135deg,#0f172a,#1e1b4b)")

    # Mutable per-page handles shared across the nested view builders. Runners are
    # cached per session and built lazily — never during initial page construction,
    # where there is no running event loop for the async SQLite checkpointer.
    ctx: dict[str, Any] = {"session_id": None, "runners": {}}

    def ensure_runner(session_id: Optional[str]) -> PinnRunner:
        """Return the runner for a session, building (and caching) it on first use.

        Must be called from within an async handler: the AsyncSqliteSaver binds to the
        running event loop at construction time.
        """

        cache = ctx["runners"]
        runner = cache.get(session_id) if session_id else None
        if runner is None:
            runner = _runner_for(cfg, session_id=session_id)
            cache[runner.thread_id] = runner
        return runner

    # ------------------------------------------------------------------ layout
    with ui.row().classes("w-full no-wrap gap-0"):
        sidebar = ui.column().classes("w-64 q-pa-sm gap-1 h-screen overflow-auto").style(
            "background: rgba(15,23,42,0.6); border-right: 1px solid rgba(255,255,255,0.08)"
        )
        main = ui.column().classes("flex-grow q-pa-md gap-3 h-screen overflow-auto")

    # ----------------------------------------------------------- streaming core
    def pump_factory(runner: PinnRunner, transcript, persist: bool):
        """Build the recursive stream pump bound to one transcript + session."""

        async def pump(payload: Any) -> None:
            def record(entry: dict) -> None:
                with transcript:
                    transcript_entry(entry)
                if persist and ctx["session_id"]:
                    store.append_entry(ctx["session_id"], entry)

            try:
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
                    record(item)
            except Exception as exc:  # noqa: BLE001 - surface any agent failure to the UI
                import traceback

                record(
                    {
                        "stage": "error",
                        "label": f"Run failed: {type(exc).__name__}",
                        "detail": str(exc).splitlines()[0] if str(exc) else "",
                        "body": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        )[-2000:],
                    }
                )

        return pump

    # --------------------------------------------------------------- views
    def open_session(session_id: str) -> None:
        """Show an existing session: its saved transcript + a follow-up chat box."""

        sess = store.get(session_id)
        if sess is None:
            return
        ctx["session_id"] = session_id
        render_sidebar()

        main.clear()
        with main:
            ui.label(sess.title).classes("text-h5 text-white")
            ui.label(f"session {session_id}").classes("text-caption text-grey-5")
            transcript = ui.column().classes("w-full gap-2")
            with transcript:
                for entry in sess.transcript:
                    transcript_entry(entry)

            async def send_followup() -> None:
                text = (followup.value or "").strip()
                if not text:
                    return
                followup.value = ""
                runner = ensure_runner(session_id)  # lazy: loop is running here
                await pump_factory(runner, transcript, persist=True)({"followup": text})

            with ui.row().classes("w-full no-wrap items-end gap-2 q-mt-sm"):
                followup = ui.textarea("Ask for a change (e.g. 'make the network deeper')").classes(
                    "flex-grow"
                ).props("dark outlined autogrow")
                ui.button(icon="send", on_click=send_followup).props("color=primary round")

    def new_session_view() -> None:
        """Show the blank input form for starting a fresh run."""

        ctx["session_id"] = None
        render_sidebar()

        main.clear()
        with main:
            ui.label("PINN Multi-Agent System").classes("text-h4 text-white")
            ui.label(
                f"provider={cfg.provider} · model={cfg.resolved_model()} · "
                f"threshold={cfg.accuracy_threshold}"
            ).classes("text-caption text-grey-4")

            query = ui.textarea("Describe the physics problem").classes("w-full").props(
                "dark outlined"
            )
            with ui.row().classes("w-full items-center gap-3"):
                dataset = ui.input("Dataset path (optional)").props("dark outlined dense")
                formulas = ui.checkbox("Formulas provided")

            transcript = ui.column().classes("w-full gap-2")

            async def start() -> None:
                q = query.value or ""
                if not q.strip():
                    ui.notify("Describe a problem first.", type="warning")
                    return
                runner = _runner_for(cfg)  # async context: checkpointer loop is live
                ctx["runners"][runner.thread_id] = runner
                title = q.strip().splitlines()[0][:80]
                store.create(runner.thread_id, workspace_dir=f"{cfg.runs_dir}/{runner.thread_id}",
                             title=title)
                ctx["session_id"] = runner.thread_id
                render_sidebar()

                init = initial_state_from_input(
                    q,
                    dataset_path=dataset.value or None,
                    formulas_given=formulas.value,
                    accuracy_threshold=cfg.accuracy_threshold,
                    max_iterations=cfg.max_iterations,
                )
                # Swap the input form for the live session view, keeping this transcript.
                run_button.disable()
                await pump_factory(runner, transcript, persist=True)(init)
                # After the first turn, reopen as a full session (adds follow-up chat).
                open_session(runner.thread_id)

            run_button = ui.button("Run", on_click=start).props("color=primary size=lg").classes(
                "w-full"
            )

    def render_sidebar() -> None:
        sidebar.clear()
        with sidebar:
            ui.button("+ New Session", on_click=new_session_view).props("flat color=primary").classes(
                "w-full"
            )
            ui.separator()
            for sess in store.list():
                active = sess.session_id == ctx["session_id"]
                ui.button(
                    sess.title,
                    on_click=lambda _=None, sid=sess.session_id: open_session(sid),
                ).props(f"flat align=left {'color=primary' if active else 'color=white'}").classes(
                    "w-full text-left ellipsis"
                )

    # Initial view: most recent session if any, else the new-run form.
    existing = store.list()
    if existing:
        open_session(existing[0].session_id)
    else:
        new_session_view()

    native = os.environ.get("PINN_GUI_NATIVE", "0") == "1"
    ui.run(native=native, title="PINN Multi-Agent System", reload=False, show=not native)
    return 0
