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
}


def transcript_entry(entry: dict[str, Any]) -> None:
    """Render one transcript line as a themed card row."""

    icon = _STAGE_ICON.get(entry.get("stage", ""), "chevron_right")
    with ui.card().classes("w-full q-pa-sm").style("backdrop-filter: blur(6px)"):
        with ui.row().classes("items-center no-wrap w-full"):
            ui.icon(icon).classes("text-primary")
            with ui.column().classes("gap-0"):
                ui.label(entry.get("label", "")).classes("text-weight-medium")
                if entry.get("detail"):
                    ui.label(entry["detail"]).classes("text-caption text-grey-7")


def clarify_dialog(payload: dict, on_submit: Callable[[dict], None]) -> Any:
    """Dialog for a clarification/approval interrupt; calls ``on_submit`` with a resume payload."""

    dialog = ui.dialog()
    with dialog, ui.card().classes("w-96"):
        ui.label("Clarify / approve the problem statement").classes("text-h6")
        ui.markdown(payload.get("statement", "") or "_(no statement)_")
        answer = ui.textarea("Answer / correction (optional)").classes("w-full")

        with ui.row().classes("w-full justify-end"):
            ui.button("Send answer", on_click=lambda: (
                on_submit({"approved": False, "answer": answer.value}),
                dialog.close(),
            )).props("flat")
            ui.button("Approve", on_click=lambda: (
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
