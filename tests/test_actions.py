import sys
import types
from pathlib import Path

import pytest

import actions as actions_mod
from actions import ActionBackendError, ActionConfigError, ActionExecutor, load_actions
from config import DEFAULT_ACTIONS_FILE


def test_repo_actions_file():
    acts = load_actions(DEFAULT_ACTIONS_FILE)
    assert acts["idle"].type == "none"
    assert acts["push"].type == "key"
    assert acts["swipe_left"].describe() == "browser back"


@pytest.mark.parametrize(
    "body,match",
    [
        ("actions:\n  push: {type: teleport}\n", "unknown type"),
        ("actions:\n  push: {type: key}\n", "key"),
        ("actions:\n  push: {type: hotkey, keys: []}\n", "hotkey"),
        ("actions:\n  push: {type: scroll, amount: 0}\n", "scroll"),
        ("actions:\n  push: {type: volume, direction: sideways}\n", "volume"),
        # Muting the system would also mute EchoTouch's own carrier tone.
        ("actions:\n  pull: {type: volume, direction: mute}\n", "volume"),
        ("actions:\n  pull: {type: key, key: volumemute}\n", "mute EchoTouch"),
        ("actions:\n  pull: {type: hotkey, keys: [ctrl, VolumeMute]}\n", "mute EchoTouch"),
        ("actions:\n  idle: {type: key, key: a}\n", "idle"),
        ("nothing: here\n", "actions"),
    ],
)
def test_invalid_actions(tmp_path: Path, body: str, match: str):
    p = tmp_path / "a.yaml"
    p.write_text(body)
    with pytest.raises(ActionConfigError, match=match):
        load_actions(p)


def test_command_parsing(tmp_path: Path):
    p = tmp_path / "a.yaml"
    p.write_text(
        "actions:\n"
        "  wave: {type: command, command: [notepad.exe, file.txt]}\n"
        "  push: {type: command, command: 'calc'}\n"
    )
    acts = load_actions(p)
    assert acts["wave"].params["command"] == ["notepad.exe", "file.txt"]
    assert acts["push"].params["command"] == ["calc"]


def test_control_off_by_default(capsys: pytest.CaptureFixture[str]):
    ex = ActionExecutor(load_actions())
    assert not ex.enabled
    assert "control OFF" in ex.execute("push")
    assert capsys.readouterr().out == ""
    assert "no action" in ex.execute("idle")


def test_dry_run_prints(capsys: pytest.CaptureFixture[str]):
    ex = ActionExecutor(load_actions(), enabled=True, dry_run=True)
    line = ex.execute("swipe_up")
    assert line.startswith("[dry-run] swipe_up: scroll +5")
    assert "[dry-run]" in capsys.readouterr().out


def make_fake_gui() -> types.ModuleType:
    gui = types.ModuleType("pyautogui")
    gui.FAILSAFE = False
    gui.calls = []
    gui.press = lambda k: gui.calls.append(("press", k))
    gui.hotkey = lambda *k: gui.calls.append(("hotkey", k))
    gui.scroll = lambda n: gui.calls.append(("scroll", n))
    return gui


def test_real_execution_keeps_failsafe(monkeypatch: pytest.MonkeyPatch):
    gui = make_fake_gui()
    monkeypatch.setitem(sys.modules, "pyautogui", gui)
    monkeypatch.setattr(actions_mod, "gui_environment_problem", lambda: None)
    ex = ActionExecutor(load_actions(), enabled=False, dry_run=False)
    ex.set_enabled(True)
    assert gui.FAILSAFE is True
    ex.execute("push")
    ex.execute("swipe_left")
    ex.execute("swipe_down")
    ex.execute("pull")
    assert gui.calls == [("press", "playpause"), ("hotkey", ("alt", "left")), ("scroll", -5), ("press", "prevtrack")]
    assert gui.FAILSAFE is True


def test_default_actions_never_silence_the_carrier():
    keys = []
    for a in load_actions().values():
        keys += [a.params.get("key", "")] + list(a.params.get("keys", []))
        assert not (a.type == "volume" and a.params["direction"] == "down"), a
    assert "volumemute" not in keys and "volumedown" not in keys


def test_force_dry_run_blocks_real_actions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    gui = make_fake_gui()
    monkeypatch.setitem(sys.modules, "pyautogui", gui)
    ex = ActionExecutor(load_actions(), enabled=True, dry_run=False, force_dry_run=True)
    assert ex.execute("push").startswith("[dry-run]")
    assert gui.calls == []


def test_headless_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(actions_mod, "gui_environment_problem", lambda: "No display found (headless session)")
    ex = ActionExecutor(load_actions(), dry_run=False)
    with pytest.raises(ActionBackendError, match="headless"):
        ex.set_enabled(True)
    assert not ex.enabled


def test_gui_environment_problem_linux(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(actions_mod.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert "Wayland" in actions_mod.gui_environment_problem()
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert "headless" in actions_mod.gui_environment_problem()
    monkeypatch.setenv("DISPLAY", ":0")
    assert actions_mod.gui_environment_problem() is None
