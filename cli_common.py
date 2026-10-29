"""Shared CLI plumbing: common arguments, logging, and the simulated-mode banner."""

from __future__ import annotations

import argparse
import logging
import sys

from config import DEFAULT_PROFILE, SIMULATED_BANNER


def add_common_args(parser: argparse.ArgumentParser, audio: bool = False) -> None:
    """Add ``--profile``, ``--simulate``, ``--gestures-file``, ``-v`` (and audio options)."""
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile name (default: %(default)s)")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="use the acoustic simulator and the sim/ directory (tests/demos; never real data)",
    )
    parser.add_argument("--gestures-file", default=None, help="gesture definitions YAML (default: gestures.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    if audio:
        parser.add_argument("--input-device", default=None, help="input device index or name substring")
        parser.add_argument("--output-device", default=None, help="output device index or name substring")
        parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def banner(text: str) -> str:
    bar = "=" * (len(text) + 8)
    return f"{bar}\n=== {text} ===\n{bar}"


def print_banner_if_simulated(simulate: bool) -> None:
    if simulate:
        print(banner(SIMULATED_BANNER), file=sys.stdout, flush=True)


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Prompt on the terminal; returns ``default`` when stdin is not interactive."""
    if not sys.stdin or not sys.stdin.isatty():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")
