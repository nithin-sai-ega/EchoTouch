"""Gesture -> OS action mapping (from ``actions.yaml``) and a safe executor.

Supported action types::

    key      {key: playpause}                 press one key
    hotkey   {keys: [alt, left]}              press a key combination
    scroll   {amount: 5}                      scroll (positive = up)
    volume   {direction: up|down}             media volume keys
    command  {command: ["notepad.exe"]}       run a program (argument list, no shell)
    none     {}                               do nothing

Safety: OS control is OFF unless explicitly enabled; ``dry_run`` only prints;
pyautogui's FAILSAFE (slam the mouse into a screen corner to abort) stays ON.
Actions that mute the system are refused: the carrier tone plays through the same
output, so muting would make EchoTouch deaf until the user unmutes by hand.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from config import DEFAULT_ACTIONS_FILE

logger = logging.getLogger(__name__)

ACTION_TYPES = ("key", "hotkey", "scroll", "volume", "command", "none")
_VOLUME_KEYS = {"up": "volumeup", "down": "volumedown"}
#: Keys that silence the system output, and with it EchoTouch's own carrier tone.
_CARRIER_KILLING_KEYS = {"volumemute"}


class ActionConfigError(ValueError):
    """Invalid ``actions.yaml`` content."""


class ActionBackendError(RuntimeError):
    """The OS-control backend (pyautogui) is unavailable in this environment."""


@dataclass(frozen=True)
class Action:
    gesture: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def describe(self) -> str:
        if self.description:
            return self.description
        p = self.params
        if self.type == "key":
            return f"press {p['key']}"
        if self.type == "hotkey":
            return "hotkey " + "+".join(p["keys"])
        if self.type == "scroll":
            return f"scroll {p['amount']:+d}"
        if self.type == "volume":
            return f"volume {p['direction']}"
        if self.type == "command":
            return "run " + " ".join(p["command"])
        return "no action"


def _check_carrier_safe(gesture: str, keys: list[str]) -> None:
    """Refuse mute keys (they would silence the carrier); warn on volume-down."""
    if _CARRIER_KILLING_KEYS & set(keys):
        raise ActionConfigError(
            f"actions.{gesture}: muting the system would also mute EchoTouch's own tone, so it "
            "could not hear any further gestures. Map this gesture to something else."
        )
    if "volumedown" in keys:
        logger.warning(
            "actions.%s lowers the system volume, which also weakens EchoTouch's tone; "
            "recognition may stop at low volumes.",
            gesture,
        )


def _parse_action(gesture: str, raw: Any) -> Action:
    if raw is None:
        return Action(gesture, "none")
    if not isinstance(raw, dict) or "type" not in raw:
        raise ActionConfigError(f"actions.{gesture}: expected a mapping with a 'type' field")
    typ = str(raw["type"])
    desc = str(raw.get("description", ""))
    if typ not in ACTION_TYPES:
        raise ActionConfigError(f"actions.{gesture}: unknown type {typ!r} (allowed: {', '.join(ACTION_TYPES)})")
    if typ == "key":
        if not isinstance(raw.get("key"), str):
            raise ActionConfigError(f"actions.{gesture}: 'key' action needs key: <name>")
        _check_carrier_safe(gesture, [raw["key"].lower()])
        return Action(gesture, typ, {"key": raw["key"].lower()}, desc)
    if typ == "hotkey":
        keys = raw.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            raise ActionConfigError(f"actions.{gesture}: 'hotkey' action needs keys: [k1, k2, ...]")
        _check_carrier_safe(gesture, [k.lower() for k in keys])
        return Action(gesture, typ, {"keys": [k.lower() for k in keys]}, desc)
    if typ == "scroll":
        amount = raw.get("amount")
        if not isinstance(amount, int) or amount == 0:
            raise ActionConfigError(f"actions.{gesture}: 'scroll' action needs a non-zero integer amount")
        return Action(gesture, typ, {"amount": amount}, desc)
    if typ == "volume":
        direction = raw.get("direction")
        if direction not in _VOLUME_KEYS:
            raise ActionConfigError(f"actions.{gesture}: volume direction must be up|down")
        _check_carrier_safe(gesture, [_VOLUME_KEYS[direction]])
        return Action(gesture, typ, {"direction": direction}, desc)
    if typ == "command":
        cmd = raw.get("command")
        if isinstance(cmd, str):
            cmd = shlex.split(cmd, posix=os.name != "nt")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
            raise ActionConfigError(f"actions.{gesture}: 'command' needs a command list or string")
        return Action(gesture, typ, {"command": cmd}, desc)
    return Action(gesture, "none", {}, desc)


def load_actions(path: Path | str | None = None) -> dict[str, Action]:
    """Load and validate the gesture->action mapping."""
    path = Path(path) if path is not None else DEFAULT_ACTIONS_FILE
    if not path.exists():
        raise ActionConfigError(f"Actions file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    mapping = data.get("actions")
    if not isinstance(mapping, dict):
        raise ActionConfigError(f"{path}: expected a top-level 'actions:' mapping")
    actions = {str(g): _parse_action(str(g), raw) for g, raw in mapping.items()}
    if "idle" in actions and actions["idle"].type != "none":
        raise ActionConfigError("The 'idle' gesture must not trigger an action")
    return actions


def gui_environment_problem() -> str | None:
    """Return a human-readable reason OS control cannot work here, or None if it probably can."""
    system = platform.system()
    if system == "Linux":
        if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            return (
                "Wayland session without XWayland DISPLAY: pyautogui cannot inject input on pure Wayland. "
                "Log into an X11/Xorg session, or run with --dry-run."
            )
        if not os.environ.get("DISPLAY"):
            return "No display found (headless session): OS control is impossible. Use --dry-run."
    return None


def environment_notes() -> list[str]:
    notes = []
    system = platform.system()
    if system == "Darwin":
        notes.append(
            "macOS: grant your terminal/IDE 'Accessibility' permission (System Settings > Privacy & "
            "Security > Accessibility) or key presses are silently ignored."
        )
    if system == "Linux" and os.environ.get("WAYLAND_DISPLAY"):
        notes.append("Wayland: pyautogui only reaches XWayland apps; native Wayland windows may ignore input.")
    return notes


class ActionExecutor:
    """Executes mapped actions; prints instead in dry-run mode; does nothing when disabled."""

    def __init__(
        self, actions: dict[str, Action], enabled: bool = False, dry_run: bool = True, force_dry_run: bool = False
    ) -> None:
        self.actions = actions
        self.enabled = enabled
        self.dry_run = dry_run
        #: When True (simulated models), real actions are impossible regardless of ``dry_run``.
        self.force_dry_run = force_dry_run
        self._gui: ModuleType | None = None

    @property
    def effective_dry_run(self) -> bool:
        return self.dry_run or self.force_dry_run

    def set_enabled(self, enabled: bool) -> None:
        if enabled and not self.effective_dry_run:
            self._backend()  # fail early with a clear message
        self.enabled = enabled

    def _backend(self) -> ModuleType:
        if self._gui is None:
            problem = gui_environment_problem()
            if problem:
                raise ActionBackendError(problem)
            try:
                import pyautogui
            except Exception as exc:  # pyautogui raises various errors when no display exists
                raise ActionBackendError(f"pyautogui is unavailable: {exc}. Use --dry-run.") from exc
            pyautogui.FAILSAFE = True  # never disable: moving the mouse to a corner aborts
            self._gui = pyautogui
        return self._gui

    def execute(self, gesture: str) -> str:
        """Run (or describe) the action for ``gesture``. Returns a log line."""
        action = self.actions.get(gesture)
        if action is None or action.type == "none":
            return f"{gesture}: (no action mapped)"
        desc = action.describe()
        if not self.enabled:
            return f"{gesture}: {desc} (control OFF)"
        if self.effective_dry_run:
            line = f"[dry-run] {gesture}: {desc}"
            print(line, flush=True)
            return line
        gui = self._backend()
        gui.FAILSAFE = True
        p = action.params
        if action.type == "key":
            gui.press(p["key"])
        elif action.type == "hotkey":
            gui.hotkey(*p["keys"])
        elif action.type == "scroll":
            gui.scroll(p["amount"])
        elif action.type == "volume":
            gui.press(_VOLUME_KEYS[p["direction"]])
        elif action.type == "command":
            subprocess.Popen(  # noqa: S603 - argument list from the user's own actions.yaml, no shell
                p["command"],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=sys.platform != "win32",
            )
        logger.info("Executed %s -> %s", gesture, desc)
        return f"{gesture}: {desc}"
