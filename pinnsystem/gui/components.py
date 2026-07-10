"""Reusable NiceGUI widgets for the PINN app.

Imported only from :mod:`pinnsystem.gui.app` at launch time, so nicegui is never a
hard dependency of the package. Each factory returns the created element(s) so the
app can update them as the run streams.
"""

from __future__ import annotations

from typing import Any, Callable

from nicegui import ui

_STAGE_ICON = {
    "parser": "psychology",
    "research": "science",
    "coding": "code",
    "feedback": "fact_check",
    "human_clarify": "help",
    "human_approve_final": "how_to_reg",
    "error": "error",
}


def _basename(path: str) -> str:
    from pathlib import PurePath

    return PurePath(path).name or path


def transcript_entry(entry: dict[str, Any]) -> None:
    """Render one stage as a collapsible "Thinking" step: header = stage + one-liner,
    expanded body = verbose detail + the files the stage produced.
    """

    stage = entry.get("stage", "")
    icon = _STAGE_ICON.get(stage, "chevron_right")
    detail = entry.get("detail", "")
    files = entry.get("files") or []
    body = entry.get("body", "")
    is_error = stage == "error"

    with ui.card().classes("w-full q-pa-none").style("backdrop-filter: blur(6px)"):
        with ui.expansion(value=is_error).classes("w-full") as exp:
            with exp.add_slot("header"):
                with ui.row().classes("items-center no-wrap w-full gap-2"):
                    ui.icon(icon).classes("text-negative" if is_error else "text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label(entry.get("label", "")).classes("text-weight-medium")
                        if detail:
                            ui.label(detail).classes("text-caption text-grey-7 ellipsis")

            with ui.column().classes("w-full gap-1 q-px-md q-pb-sm"):
                if body:
                    ui.markdown(f"```\n{body}\n```").classes("w-full text-caption")
                if files:
                    ui.label("Files created").classes("text-caption text-grey-6")
                    for path in files:
                        with ui.row().classes("items-center no-wrap gap-1"):
                            ui.icon("description").classes("text-secondary text-sm")
                            ui.label(_basename(path)).classes("text-caption").tooltip(path)
                if not body and not files and not detail:
                    ui.label("—").classes("text-caption text-grey-6")


def clarify_dialog(payload: dict, on_submit: Callable[[dict], None]) -> Any:
    """Statement-confirmation gate: show the parsed problem and let the user approve it
    or send a correction (which re-runs the parser). ``on_submit`` gets the resume payload.
    """

    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-[36rem] max-w-full gap-2"):
        ui.label("Confirm the problem statement").classes("text-h6")
        ui.label("The model creation loop starts once you approve this.").classes(
            "text-caption text-grey-6"
        )

        ui.markdown(payload.get("statement", "") or "_(no statement produced)_")

        if payload.get("pde_latex"):
            ui.label("PDE / ODE").classes("text-caption text-grey-6 q-mt-sm")
            ui.markdown(f"$$ {payload['pde_latex']} $$")

        domain = payload.get("domain")
        if domain:
            vars_ = ", ".join(domain.get("variables", [])) or "—"
            bounds = ", ".join(f"{k}∈[{lo}, {hi}]" for k, (lo, hi) in domain.get("bounds", {}).items())
            ui.label(
                f"Domain — {domain.get('dims')}D · variables: {vars_}"
                + (f" · {bounds}" if bounds else "")
            ).classes("text-caption text-grey-7")

        if payload.get("quantities"):
            ui.label("Solve for: " + ", ".join(payload["quantities"])).classes(
                "text-caption text-grey-7"
            )
        for key, title in (("boundary_conditions", "BCs"), ("initial_conditions", "ICs")):
            if payload.get(key):
                ui.label(f"{title}: " + "; ".join(payload[key])).classes(
                    "text-caption text-grey-7"
                )

        answer = ui.textarea("Correction (optional — what's wrong or missing)").classes("w-full")

        with ui.row().classes("w-full justify-end q-mt-sm"):
            ui.button("Send correction", on_click=lambda: (
                on_submit({"approved": False, "answer": answer.value}),
                dialog.close(),
            )).props("flat")
            ui.button("Approve & run", on_click=lambda: (
                on_submit({"approved": True}),
                dialog.close(),
            )).props("color=primary")
    dialog.open()
    return dialog


def approval_bar(payload: dict, on_submit: Callable[[dict], None]) -> Any:
    """Final accept/reject dialog, showing the Feedback verdict."""

    fb = payload.get("feedback") or {}
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-96"):
        ui.label("Final result").classes("text-h6")
        if fb:
            ui.label(f"Quality score: {fb.get('quality_score', 0):.3f}")
            ui.label(f"Decision: {fb.get('decision')}")
            ui.label(fb.get("directive", ""))
        with ui.row().classes("w-full justify-end"):
            ui.button("Iterate again", on_click=lambda: (
                on_submit({"approved": False}),
                dialog.close(),
            )).props("flat")
            ui.button("Accept", on_click=lambda: (
                on_submit({"approved": True}),
                dialog.close(),
            )).props("color=positive")
    dialog.open()
    return dialog
