"""Execution layer: workspace layout and the subprocess venv runner."""

from __future__ import annotations

from pinnsystem.execution import VenvRunner, new_workspace


def test_workspace_creates_standard_subdirs(tmp_path):
    ws = new_workspace(tmp_path, run_id="run_test")
    assert ws.root.exists()
    for sub in (ws.scripts, ws.data, ws.models, ws.plots, ws.metrics):
        assert sub.is_dir()

    nested = ws.path("scripts", "model.py")
    assert nested.parent.is_dir()


def test_run_code_captures_stdout(tmp_path):
    runner = VenvRunner()  # current interpreter — no venv build
    outcome = runner.run_code("print('hello pinn')", workdir=tmp_path, timeout=30)
    assert outcome.ok
    assert "hello pinn" in outcome.stdout
    assert outcome.error is None


def test_run_code_reports_error(tmp_path):
    runner = VenvRunner()
    outcome = runner.run_code("raise ValueError('boom')", workdir=tmp_path, timeout=30)
    assert not outcome.ok
    assert outcome.returncode != 0
    assert "boom" in (outcome.error or "")


def test_run_code_times_out(tmp_path):
    runner = VenvRunner()
    outcome = runner.run_code(
        "import time; time.sleep(5)", workdir=tmp_path, timeout=0.5
    )
    assert outcome.timed_out
    assert not outcome.ok
    assert "Timeout" in (outcome.error or "")
