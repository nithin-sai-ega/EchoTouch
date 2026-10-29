"""Guided enrollment wizard: record YOUR acoustic gesture signature.

    python enroll.py --profile default            # interactive wizard (resumable)
    python enroll.py --simulate --fast            # demo: fills sim/<profile> from the simulator

The :class:`EnrollmentController` holds all the logic (recording, quality checks,
saving, undo, round planning) and is shared with the Streamlit Enrollment Studio;
this file adds only the terminal UI on top of it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from audio_stream import AudioDeviceError, AudioSource
from config import (
    IDLE_SUBTYPES,
    ROUND_VARIATIONS,
    SIMULATED_BANNER,
    DSPConfig,
    EnrollConfig,
    GestureSpec,
    enabled_gestures,
    load_gestures,
)
from dsp import QualityReport, motion_score_db, quality_check, snr_at_frequency, stft_crop
from profiles import ProfileStore, SimulatedDataError

logger = logging.getLogger(__name__)

INTRO = """
Welcome to EchoTouch enrollment!

You are about to record YOUR personal acoustic gesture signature. Your laptop plays an
inaudible-ish tone and listens to its echo off your hand: the Doppler shift of that echo is
what EchoTouch learns. The exact signal depends on YOUR laptop's speaker/mic placement, mic
sensitivity, your room and your hand, so no generic dataset exists: the model is trained ONLY
on what you record here.

How it works:
  * You record in ROUNDS (default 25 clips per gesture). Each round asks you to vary speed,
    distance and hand position, so the model generalizes instead of memorizing one motion.
  * For every clip: "Get ready ... 3, 2, 1, GO!" then ~1.5 s of recording. Do the gesture once,
    right after GO.
  * Bad clips (no motion detected, clipping) are rejected automatically and re-recorded.
  * 'idle' is just as important: you will be asked to sit still, type, move your head, have
    someone walk by, etc. These teach the model what NOT to react to.
  * Start with ~50 clips per gesture to check feasibility; then build up to 150-200 over
    several sessions (different times of day / postures) for a robust model.
  * Progress is saved after every clip. Quit any time and re-run to resume.
"""

COMMANDS_HELP = "[Enter]=next  u=undo last  r=redo this  n=next idle type  s=skip gesture  q=quit"


@dataclass
class ClipAttempt:
    gesture: str
    audio: np.ndarray
    quality: QualityReport
    idle_subtype: str | None = None


class EnrollmentController:
    """Shared enrollment logic for the CLI wizard and the Streamlit Enrollment Studio."""

    def __init__(
        self,
        store: ProfileStore,
        engine: AudioSource,
        gestures: Sequence[GestureSpec],
        cfg: EnrollConfig | None = None,
    ) -> None:
        if bool(engine.simulated) != bool(store.simulated):
            raise SimulatedDataError(
                "Audio source and profile disagree about simulation: simulated audio may only be "
                "stored in sim/ profiles and real audio only in profiles/."
            )
        self.store = store
        self.engine = engine
        self.gestures = list(gestures)
        self.cfg = cfg or EnrollConfig()
        self.dsp: DSPConfig = store.dsp_config()
        self.session_id: str | None = None
        self.round = 0
        self._rng = np.random.default_rng()

    # -- planning ------------------------------------------------------------- #
    def begin_session(self) -> str:
        self.session_id = self.store.new_session()
        self.round = 0
        return self.session_id

    def begin_round(self) -> tuple[int, str]:
        """Start a new round; returns ``(round_no, variation instructions)``."""
        if self.session_id is None:
            self.begin_session()
        self.round = self.store.start_round(self.session_id)
        return self.round, self.variation(self.round)

    @staticmethod
    def variation(round_no: int) -> str:
        return ROUND_VARIATIONS[(round_no - 1) % len(ROUND_VARIATIONS)]

    def idle_subtype(self, clip_index: int, clips_per_round: int | None = None) -> tuple[str, str]:
        """Idle sub-type for the ``clip_index``-th idle clip of a round (contiguous blocks)."""
        per_round = clips_per_round or self.cfg.clips_per_round
        block = max(1, int(np.ceil(per_round / len(IDLE_SUBTYPES))))
        return IDLE_SUBTYPES[min(clip_index // block, len(IDLE_SUBTYPES) - 1)]

    def spec(self, gesture: str) -> GestureSpec:
        for g in self.gestures:
            if g.name == gesture:
                return g
        raise KeyError(gesture)

    def counts(self) -> dict[str, int]:
        c = self.store.counts()
        return {g.name: int(c.get(g.name, 0)) for g in self.gestures}

    def progress_line(self, target: int) -> str:
        return "  ".join(f"{g}: {n}/{target}" for g, n in self.counts().items())

    # -- recording ------------------------------------------------------------ #
    def record(self, gesture: str, idle_subtype: str | None = None) -> ClipAttempt:
        """Record one clip right now (the UI must already have shown the countdown)."""
        spec = self.spec(gesture)
        if self.engine.simulated:
            # The simulator "performs" the requested gesture shortly after GO.
            delay = float(self._rng.uniform(0.15, 0.35))
            self.engine.perform(gesture, delay=delay, idle_subtype=idle_subtype if gesture == "idle" else None)  # type: ignore[attr-defined]
        audio = self.engine.record(self.dsp.clip_seconds)
        q = quality_check(audio, self.dsp, expects_motion=spec.motion, thresholds=self.cfg)
        return ClipAttempt(gesture, audio, q, idle_subtype if gesture == "idle" else None)

    def save(self, attempt: ClipAttempt) -> dict[str, Any]:
        if self.session_id is None or self.round == 0:
            self.begin_round()
        extra = {"idle_subtype": attempt.idle_subtype} if attempt.idle_subtype else None
        return self.store.add_clip(
            attempt.audio,
            attempt.gesture,
            self.session_id,
            self.round,
            self.store.mode,
            attempt.quality.to_dict(),
            extra,
        )

    def undo_last(self) -> dict[str, Any] | None:
        """Undo the most recent clip of the current session."""
        return self.store.undo_last(self.session_id) if self.session_id else None

    def meter(self, seconds: float = 0.5) -> tuple[float, float]:
        """Live carrier SNR (dB) and motion score (dB) over the latest audio."""
        if self.engine.simulated and not getattr(self.engine, "realtime", True):
            self.engine.advance(seconds)  # type: ignore[attr-defined]
        audio = self.engine.get_latest(max(seconds, self.dsp.window_seconds))
        if len(audio) < self.dsp.n_fft * 2:
            return float("nan"), float("nan")
        snr = snr_at_frequency(audio, self.dsp.sample_rate, self.dsp.carrier_hz)
        return snr, motion_score_db(stft_crop(audio, self.dsp), self.dsp)


# --------------------------------------------------------------------------- #
# Terminal UI helpers
# --------------------------------------------------------------------------- #
class KeyReader:
    """Non-blocking single-key reads on Windows and POSIX terminals."""

    def __init__(self) -> None:
        self.interactive = bool(sys.stdin) and sys.stdin.isatty()
        self._old: Any = None

    def __enter__(self) -> KeyReader:
        if self.interactive and os.name != "nt":
            import termios
            import tty

            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc: object) -> None:
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            self._old = None

    def get(self, timeout: float) -> str | None:
        """Return a key pressed within ``timeout`` seconds, else None."""
        if not self.interactive:
            return None
        deadline = time.monotonic() + timeout
        if os.name == "nt":
            import msvcrt

            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    return "\n" if ch in ("\r", "\n") else ch.lower()
                time.sleep(0.02)
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
        if ready:
            ch = sys.stdin.read(1)
            return "\n" if ch in ("\r", "\n") else ch.lower()
        return None

    def flush(self) -> None:
        while self.get(0.0) is not None:
            pass


def meter_bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    if np.isnan(value):
        return "[" + "?" * width + "]"
    frac = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def countdown(gesture: str, seconds: int, fast: bool) -> None:
    label = gesture.upper().replace("_", " ")
    if fast:
        print(f"  {label}: GO!", end=" ", flush=True)
        return
    print(f"  Get ready: {label} ...", end=" ", flush=True)
    for i in range(seconds, 0, -1):
        print(f"{i},", end=" ", flush=True)
        time.sleep(0.6)
    print("GO!", end=" ", flush=True)


def feedback_line(q: QualityReport) -> str:
    status = "OK " if q.accepted else "REJECTED"
    parts = [
        f"{status} SNR {q.snr_db:4.0f} dB {meter_bar(q.snr_db, 0, 60, 10)}",
        f"motion {q.motion_db:4.1f} dB {meter_bar(q.motion_db, 0, 20, 10)}",
    ]
    line = " | ".join(parts)
    if q.messages:
        line += "\n      " + "\n      ".join(q.messages)
    return line


def run_wizard(
    ctrl: EnrollmentController, gestures: list[str], rounds: int, clips: int, target: int, fast: bool, step: bool
) -> None:
    """The interactive round loop. Saves after every clip; returns on quit or completion."""
    keys = KeyReader()
    with keys:
        snr, motion = ctrl.meter(1.0)
        print(
            f"\nLive meter: carrier SNR {snr:5.1f} dB {meter_bar(snr, 0, 60)}   motion {motion:4.1f} dB "
            f"{meter_bar(motion, 0, 20)}"
        )
        if not np.isnan(snr) and snr < ctrl.cfg.low_snr_db:
            print("  WARNING: low carrier SNR. Consider re-running doctor.py.")
        for _ in range(rounds):
            rnd, variation = ctrl.begin_round()
            print(f"\n=== Round {rnd} (session {ctrl.session_id}) ===\nThis round: {variation}")
            for g in gestures:
                spec = ctrl.spec(g)
                print(f"\n--- {spec.label.upper()} ---  {spec.instruction}")
                if spec.experimental:
                    print("    (experimental gesture: accuracy may be limited)")
                done, rejects, last_sub = 0, 0, None
                sub_offset = 0
                while done < clips:
                    sub = ctrl.idle_subtype(done + sub_offset, clips) if g == "idle" else None
                    if sub and sub[0] != last_sub:
                        print(f"\n    IDLE type: {sub[0].upper()} - {sub[1]}")
                        last_sub = sub[0]
                    if not fast and not step:
                        time.sleep(0.4)
                    if step:
                        input("    press Enter when ready...")
                    countdown(g, ctrl.cfg.countdown_seconds, fast)
                    attempt = ctrl.record(g, sub[0] if sub else None)
                    print("recorded.")
                    print("    " + feedback_line(attempt.quality))
                    saved = None
                    if attempt.quality.accepted:
                        saved = ctrl.save(attempt)
                        done += 1
                        rejects = 0
                    else:
                        rejects += 1
                        if rejects >= 5:
                            print(
                                "    Many rejects in a row: move closer/faster, or check doctor.py. "
                                "Press k to keep this clip anyway."
                            )
                    print(f"    {g}: {ctrl.counts()[g]}/{target} total, {done}/{clips} this round.  {COMMANDS_HELP}")
                    key = None if fast else keys.get(ctrl.cfg.review_seconds)
                    if key == "k" and saved is None:
                        ctrl.save(attempt)
                        done += 1
                        print("    kept anyway.")
                    elif key == "u":
                        rec = ctrl.undo_last()
                        if rec:
                            print(f"    undid {rec['gesture']} clip {rec['filename']}")
                            if rec["gesture"] == g and rec["round"] == rnd:
                                done = max(0, done - 1)
                    elif key == "r" and saved is not None:
                        ctrl.undo_last()
                        done -= 1
                        print("    discarded; recording again.")
                    elif key == "n" and g == "idle":
                        block = max(1, int(np.ceil(clips / len(IDLE_SUBTYPES))))
                        sub_offset += block - ((done + sub_offset) % block)
                    elif key == "s":
                        print(f"    skipping the rest of {g} this round.")
                        break
                    elif key == "q":
                        print("\nQuitting. Progress is saved; re-run enroll.py to continue.")
                        return
                    keys.flush()
            print(f"\nRound {rnd} complete. {ctrl.progress_line(target)}")


def main(argv: list[str] | None = None) -> int:
    from cli_common import add_common_args, ask_yes_no, banner, print_banner_if_simulated, setup_logging
    from engine import EngineError, create_audio, open_store

    parser = argparse.ArgumentParser(description="EchoTouch guided enrollment wizard")
    add_common_args(parser, audio=True)
    parser.add_argument("--rounds", type=int, default=1, help="rounds to record now (default 1)")
    parser.add_argument("--clips", type=int, default=EnrollConfig.clips_per_round, help="clips per gesture per round")
    parser.add_argument("--gestures", default=None, help="comma-separated subset of enabled gestures")
    parser.add_argument("--target", type=int, default=150, help="total clips per gesture you are aiming for")
    parser.add_argument("--fast", action="store_true", help="no countdown/review pauses (mainly for --simulate)")
    parser.add_argument("--step", action="store_true", help="wait for Enter before every clip")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    print_banner_if_simulated(args.simulate)
    if args.list_devices:
        from audio_stream import list_devices_text

        print(list_devices_text())
        return 0

    specs = enabled_gestures(load_gestures(args.gestures_file))
    names = [s.name for s in specs]
    chosen = names if not args.gestures else [g.strip() for g in args.gestures.split(",") if g.strip()]
    unknown = [g for g in chosen if g not in names]
    if unknown:
        print(f"ERROR: not enabled in gestures.yaml: {', '.join(unknown)} (enabled: {', '.join(names)})")
        return 2

    print(banner("EchoTouch enrollment"))
    print(INTRO)
    store = open_store(args.profile, args.simulate)
    if args.simulate and not store.exists():
        store.create(DSPConfig(), 0.3, {"input": "simulated mic", "output": "simulated speaker"})
        store.set_calibration(
            {"carrier_hz": DSPConfig().carrier_hz, "simulated": True, "tool": "enroll.py"}, DSPConfig(), 0.3
        )
    if not store.exists() or not store.calibration:
        print(f"Profile '{args.profile}' is not calibrated yet. Calibration (doctor.py) must run first.")
        if ask_yes_no("Run doctor.py now?", True):
            import doctor

            extra = ["--simulate"] if args.simulate else []
            if args.input_device:
                extra += ["--input-device", str(args.input_device)]
            if args.output_device:
                extra += ["--output-device", str(args.output_device)]
            if doctor.main(["--profile", args.profile, *extra]) != 0:
                return 2
            store.reload()
        else:
            return 2
    cal = store.calibration or {}
    dsp = store.dsp_config()
    print(
        f"Using calibration: carrier {dsp.carrier_hz:.0f} Hz, SNR {cal.get('snr_db', '?')} dB, "
        f"volume {store.volume}, updated {cal.get('updated', '?')}."
    )
    if not args.simulate and not ask_yes_no("Is this the same laptop/setup as the calibration?", True):
        print("Please re-run doctor.py (ideally with a new --profile) first.")
        return 2
    counts = store.counts()
    if counts:
        print("Existing clips: " + ", ".join(f"{g}={counts.get(g, 0)}" for g in names) + " (resuming)")
    print(
        f"Plan: {args.rounds} round(s) x {args.clips} clips x {len(chosen)} gestures = "
        f"{args.rounds * args.clips * len(chosen)} clips."
    )

    engine = create_audio(
        dsp,
        store.volume,
        args.simulate,
        args.input_device,
        args.output_device,
        realtime=not (args.simulate and args.fast),
    )
    try:
        with engine:
            ctrl = EnrollmentController(store, engine, specs)
            ctrl.begin_session()
            if not args.simulate and not args.fast:
                input("\nPosition yourself in front of the laptop and press Enter to start...")
            run_wizard(ctrl, chosen, args.rounds, args.clips, args.target, args.fast or False, args.step)
    except (AudioDeviceError, EngineError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Progress so far is saved.")
    final = store.counts()
    print("\nPer-gesture totals:")
    for g in names:
        n = final.get(g, 0)
        hint = "  (start: 50 to test feasibility; aim for 150-200)" if n < 50 else ""
        print(f"  {g:<12} {n:>4} {meter_bar(n, 0, args.target, 20)}{hint}")
    print(
        f"\nNext: python train.py --profile {args.profile}" + (" --simulate --allow-simulated" if args.simulate else "")
    )
    if args.simulate:
        print(SIMULATED_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
