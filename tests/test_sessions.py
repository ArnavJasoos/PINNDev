"""Persistent session store for the GUI sidebar."""

from __future__ import annotations

from pinnsystem.gui.sessions import SessionStore


def test_create_list_and_persist(tmp_path):
    store = SessionStore(tmp_path)
    store.create("run_a", workspace_dir=str(tmp_path / "run_a"), title="Heat eq")
    store.create("run_b", workspace_dir=str(tmp_path / "run_b"), title="Oscillator")

    titles = [s.title for s in store.list()]
    assert set(titles) == {"Heat eq", "Oscillator"}

    # Reload from disk: a new store instance sees the same sessions.
    reloaded = SessionStore(tmp_path)
    assert {s.session_id for s in reloaded.list()} == {"run_a", "run_b"}


def test_append_entry_survives_reload(tmp_path):
    store = SessionStore(tmp_path)
    store.create("run_a", workspace_dir=str(tmp_path / "run_a"), title="t")
    store.append_entry("run_a", {"stage": "parser", "label": "Parser", "detail": "u_tt+u=0"})
    store.append_entry("run_a", {"stage": "coding", "label": "Coding", "files": ["main.py"]})

    reloaded = SessionStore(tmp_path)
    transcript = reloaded.get("run_a").transcript
    assert len(transcript) == 2
    assert transcript[0]["detail"] == "u_tt+u=0"
    assert transcript[1]["files"] == ["main.py"]


def test_set_title_truncates_and_persists(tmp_path):
    store = SessionStore(tmp_path)
    store.create("run_a", workspace_dir=str(tmp_path / "run_a"))
    store.set_title("run_a", "x" * 200)
    assert len(SessionStore(tmp_path).get("run_a").title) == 80
