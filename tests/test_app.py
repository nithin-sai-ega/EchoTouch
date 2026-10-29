"""Smoke-test the Streamlit dashboard headlessly (no browser, no audio hardware)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_real_mode(isolated_repo, fake_sd):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "EchoTouch" in text
    assert "Enrollment Studio" in [h.value for h in at.subheader]


def test_app_renders_simulate_mode(isolated_repo, fake_sd):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.toggle(key="simulate").set_value(True).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "SIMULATED MODE" in text
    assert (isolated_repo / "sim" / "demo" / "manifest.json").exists()
    assert not (isolated_repo / "profiles").exists() or not any((isolated_repo / "profiles").iterdir())
