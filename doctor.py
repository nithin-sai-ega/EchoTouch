"""Hardware check and carrier calibration. RUN THIS FIRST.

    python doctor.py                      # full check + carrier sweep, saves to profiles/default
    python doctor.py --list-devices
    python doctor.py --simulate           # demo against the simulator (saves to sim/)

Checks: devices exist; the mic hears the carrier (SNR vs noise floor); sweeps
17-21 kHz and picks the best (preferring higher, less audible frequencies);
mic level and clipping; carrier stability (drop-outs hint at OS noise suppression,
"audio enhancements" or Bluetooth). Prints PASS/WARN/FAIL with actionable fixes.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audio_stream import AudioDeviceError, AudioSource
from config import SIMULATED_BANNER, AudioConfig, CalibrationConfig, ConfigError, DSPConfig
from dsp import as_channels_first, clipping_fraction, rms_dbfs, snr_at_frequency

logger = logging.getLogger(__name__)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_BT_HINTS = ("bluetooth", "hands-free", "handsfree", "airpods", "headset", "bt ", "buds", "a2dp", "hfp")

AUDIBILITY_WARNING = (
    "Some people (especially children) and pets can hear 17-19 kHz tones. Keep the volume LOW "
    "(default 0.3) and prefer the highest carrier that still gives good SNR."
)
INVALIDATION_WARNING = (
    "Changing the carrier frequency, volume, sample rate or devices later invalidates recorded "
    "clips and trained models: you would need to re-record and retrain."
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)
    carrier_hz: float | None = None
    snr_db: float | None = None
    sweep: list[tuple[float, float]] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str, fix: str = "") -> CheckResult:
        res = CheckResult(name, status, detail, fix)
        self.results.append(res)
        return res

    @property
    def overall(self) -> str:
        statuses = {r.status for r in self.results}
        return FAIL if FAIL in statuses else WARN if WARN in statuses else PASS

    def calibration(self) -> dict[str, Any]:
        return {
            "carrier_hz": self.carrier_hz,
            "snr_db": None if self.snr_db is None else round(self.snr_db, 2),
            "sweep": [{"carrier_hz": f, "snr_db": round(s, 2)} for f, s in self.sweep],
            "status": self.overall,
            "checks": [{"name": r.name, "status": r.status, "detail": r.detail} for r in self.results],
            **self.measurements,
            "tool": "doctor.py",
        }

    def format(self) -> str:
        lines = ["", "EchoTouch doctor report", "-" * 60]
        for r in self.results:
            lines.append(f"[{r.status}] {r.name}: {r.detail}")
            if r.fix and r.status != PASS:
                lines.append(f"       fix: {r.fix}")
        if self.sweep:
            lines.append("\nCarrier sweep (SNR at mic):")
            best = max(s for _, s in self.sweep)
            for f, s in self.sweep:
                bar = "#" * int(max(0.0, s) / 3)
                mark = "  <== selected" if self.carrier_hz is not None and abs(f - self.carrier_hz) < 1 else ""
                lines.append(
                    f"  {f:7.0f} Hz  {s:6.1f} dB  {bar}{mark}" + ("  (best SNR)" if s == best and not mark else "")
                )
        lines.append("-" * 60)
        lines.append(f"OVERALL: {self.overall}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual checks (engine-agnostic so they run on the simulator too)
# --------------------------------------------------------------------------- #
def bluetooth_suspect(name: str) -> bool:
    n = f" {name.lower()} "
    return any(h in n for h in _BT_HINTS)


def check_devices(report: DoctorReport, input_name: str, output_name: str) -> None:
    report.add("devices", PASS, f"input='{input_name}', output='{output_name}'")
    for kind, name in (("input", input_name), ("output", output_name)):
        if bluetooth_suspect(name):
            report.add(
                f"{kind} device type",
                WARN,
                f"'{name}' looks like a Bluetooth device",
                "Bluetooth audio usually cuts frequencies above ~16 kHz and adds latency. Use the "
                "laptop's built-in speaker and microphone.",
            )


def carrier_level_series(audio: np.ndarray, sr: int, freq: float, seg_seconds: float = 0.1) -> np.ndarray:
    """Carrier amplitude (dBFS) per segment via a single-bin windowed DFT."""
    x = as_channels_first(audio).mean(axis=0)
    seg = max(256, int(sr * seg_seconds))
    n = len(x) // seg
    if n == 0:
        return np.zeros(0)
    frames = x[: n * seg].reshape(n, seg)
    w = np.hanning(seg)
    ph = np.exp(-2j * np.pi * freq * np.arange(seg) / sr)
    amp = np.abs((frames * w) @ ph) / (w.sum() / 2)
    return 20 * np.log10(amp + 1e-12)


def measure_snr(engine: AudioSource, freq: float, cal: CalibrationConfig) -> tuple[float, np.ndarray]:
    engine.set_frequency(freq)
    engine.record(cal.settle_seconds)  # let the tone (and any OS processing) settle
    audio = engine.record(cal.measure_seconds)
    return snr_at_frequency(audio, engine.sample_rate, freq), audio


def choose_carrier(sweep: list[tuple[float, float]], cal: CalibrationConfig) -> tuple[float, str]:
    """Highest carrier whose SNR is within ``prefer_high_within_db`` of the best."""
    best = max(s for _, s in sweep)
    ok = [f for f, s in sweep if s >= best - cal.prefer_high_within_db and s >= cal.snr_warn_db]
    if ok:
        f = max(ok)
        return f, f"highest carrier within {cal.prefer_high_within_db:.0f} dB of the best SNR (less audible)"
    f = max(sweep, key=lambda fs: fs[1])[0]
    return f, "best available SNR (all candidates are weak)"


def check_stability(report: DoctorReport, audio: np.ndarray, sr: int, freq: float, cal: CalibrationConfig) -> None:
    levels = carrier_level_series(audio, sr, freq)
    if len(levels) < 4:
        report.add("carrier stability", WARN, "recording too short to judge stability")
        return
    med = float(np.median(levels))
    dropouts = int(np.sum(levels < med - cal.dropout_db))
    third = max(1, len(levels) // 3)
    drift = float(np.median(levels[:third]) - np.median(levels[-third:]))
    report.measurements["carrier_level_dbfs"] = round(med, 2)
    report.measurements["carrier_dropouts"] = dropouts
    report.measurements["carrier_drift_db"] = round(drift, 2)
    if dropouts:
        report.add(
            "carrier stability",
            WARN,
            f"the carrier disappeared in {dropouts}/{len(levels)} segments (>{cal.dropout_db:.0f} dB drops)",
            "Disable OS noise suppression / 'audio enhancements' (Windows: Sound settings > device > "
            "Advanced > turn off Audio enhancements; also disable 'Voice clarity' / Teams/Zoom noise "
            "suppression), and avoid Bluetooth. See docs/TROUBLESHOOTING.md.",
        )
    elif abs(drift) > cal.drift_db:
        report.add(
            "carrier stability",
            WARN,
            f"carrier level drifted by {drift:.1f} dB over a few seconds",
            "Automatic gain control or noise suppression is adapting. Disable audio enhancements / AGC.",
        )
    else:
        report.add("carrier stability", PASS, f"steady (level {med:.1f} dBFS, drift {drift:+.1f} dB)")


def run_doctor(
    engine: AudioSource,
    dsp: DSPConfig,
    cal: CalibrationConfig | None = None,
    sweep: bool = True,
    progress: Any = None,
) -> DoctorReport:
    """Run all checks on an already-started engine. ``progress(str)`` gets status lines."""
    cal = cal or CalibrationConfig()
    say = progress or (lambda _m: None)
    report = DoctorReport()
    sr = engine.sample_rate
    check_devices(report, getattr(engine, "input_name", "?"), getattr(engine, "output_name", "?"))

    # 1) Ambient noise floor with the tone off.
    say("Measuring ambient noise (tone off)...")
    engine.set_tone_enabled(False)
    engine.record(0.2)
    ambient = engine.record(1.0)
    noise_dbfs = rms_dbfs(ambient)
    report.measurements["ambient_rms_dbfs"] = round(noise_dbfs, 2)
    if noise_dbfs < -90:
        report.add(
            "microphone",
            FAIL,
            f"input is silent ({noise_dbfs:.0f} dBFS)",
            "The mic may be muted, disabled, or blocked by OS privacy settings (Windows: Settings > "
            "Privacy > Microphone; macOS: System Settings > Privacy > Microphone).",
        )
    elif clipping_fraction(ambient) > 0.001:
        report.add("microphone", WARN, "ambient sound is clipping", "Lower the microphone input gain.")
    else:
        report.add("microphone", PASS, f"ambient level {noise_dbfs:.1f} dBFS")
    engine.set_tone_enabled(True)

    # 2) Carrier sweep (or single frequency).
    candidates = [f for f in cal.candidate_carriers if f + dsp.crop_hz < sr / 2 - 200]
    if not sweep:
        candidates = [dsp.carrier_hz]
    say(f"Sweeping {len(candidates)} carrier frequencies...")
    for f in candidates:
        snr, _ = measure_snr(engine, f, cal)
        report.sweep.append((f, snr))
        say(f"  {f:.0f} Hz: SNR {snr:.1f} dB")
    carrier, why = choose_carrier(report.sweep, cal)
    report.carrier_hz = carrier
    report.snr_db = dict(report.sweep)[carrier]
    if report.snr_db < cal.snr_fail_db:
        report.add(
            "carrier detection",
            FAIL,
            f"mic barely hears the carrier (best SNR {report.snr_db:.1f} dB)",
            "Raise --volume a little, use the built-in speaker+mic (not Bluetooth), disable audio "
            "enhancements, and make sure the speaker is not muted.",
        )
    elif report.snr_db < cal.snr_warn_db:
        report.add(
            "carrier detection",
            WARN,
            f"weak carrier: {carrier:.0f} Hz at {report.snr_db:.1f} dB SNR",
            "Gestures may be unreliable. Try a slightly higher --volume or a quieter room.",
        )
    else:
        report.add("carrier detection", PASS, f"{carrier:.0f} Hz at {report.snr_db:.1f} dB SNR ({why})")

    # 3) Level / clipping and 4) stability at the chosen carrier.
    say(f"Checking stability at {carrier:.0f} Hz for {cal.stability_seconds:.0f} s (keep still)...")
    engine.set_frequency(carrier)
    engine.record(cal.settle_seconds)
    steady = engine.record(cal.stability_seconds)
    peak = float(np.max(np.abs(steady))) if steady.size else 0.0
    report.measurements["peak"] = round(peak, 4)
    if clipping_fraction(steady, cal.clip_level) > 0.0005:
        report.add(
            "clipping",
            FAIL,
            f"input clips with the tone on (peak {peak:.2f})",
            "Lower --volume and/or the microphone gain in OS sound settings.",
        )
    elif peak > 0.7:
        report.add("clipping", WARN, f"input is hot (peak {peak:.2f})", "Consider lowering --volume or mic gain.")
    else:
        report.add("clipping", PASS, f"peak {peak:.2f}")
    check_stability(report, steady, sr, carrier, cal)
    report.add(
        "audibility",
        WARN if carrier < 18500 else PASS,
        f"carrier {carrier:.0f} Hz" + (" may be audible to some people/pets" if carrier < 18500 else ""),
        AUDIBILITY_WARNING,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    from cli_common import add_common_args, ask_yes_no, banner, print_banner_if_simulated, setup_logging
    from engine import create_audio, open_store
    from profiles import ProfileError
    from simulator import SCENARIOS

    parser = argparse.ArgumentParser(description="EchoTouch hardware check and carrier calibration (run first)")
    add_common_args(parser, audio=True)
    parser.add_argument("--volume", type=float, default=AudioConfig().volume, help="carrier volume 0..1 (keep low)")
    parser.add_argument("--sample-rate", type=int, default=DSPConfig().sample_rate, choices=(44100, 48000))
    parser.add_argument("--carrier", type=float, default=None, help="test only this carrier (no sweep)")
    parser.add_argument("--channels", type=int, default=1, help="microphone channels to record")
    parser.add_argument("--no-save", action="store_true", help="do not save calibration to the profile")
    parser.add_argument("--force", action="store_true", help="overwrite calibration even if clips exist")
    parser.add_argument("--scenario", default="normal", choices=SCENARIOS, help="simulator scenario (with --simulate)")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    print_banner_if_simulated(args.simulate)

    if args.list_devices:
        from audio_stream import list_devices_text

        try:
            print(list_devices_text())
        except AudioDeviceError as exc:
            print(f"ERROR: {exc}")
            return 2
        return 0

    print(banner("EchoTouch doctor"))
    print(f"WARNING: {AUDIBILITY_WARNING}")
    print(f"NOTE: {INVALIDATION_WARNING}\n")
    try:
        dsp = DSPConfig(
            sample_rate=args.sample_rate, channels=args.channels, carrier_hz=args.carrier or DSPConfig().carrier_hz
        )
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    engine = create_audio(
        dsp,
        args.volume,
        args.simulate,
        args.input_device,
        args.output_device,
        realtime=not args.simulate,
        scenario=args.scenario,
    )
    try:
        with engine:
            report = run_doctor(engine, dsp, sweep=args.carrier is None, progress=print)
            status = engine.status()
    except AudioDeviceError as exc:
        print(f"\nFAIL: {exc}")
        print("See docs/TROUBLESHOOTING.md.")
        return 2
    print(report.format())
    if status.get("input_overflows") or status.get("output_underflows"):
        print(
            f"NOTE: audio xruns during the test (in {status['input_overflows']}, out {status['output_underflows']}). "
            "Close heavy apps; a few are harmless."
        )
    if report.overall == FAIL or args.no_save or report.carrier_hz is None:
        if report.overall == FAIL:
            print("\nCalibration NOT saved because of failures. Fix them and run doctor.py again.")
        return 1 if report.overall == FAIL else 0

    store = open_store(args.profile, args.simulate)
    chosen = dsp.with_overrides(carrier_hz=report.carrier_hz)
    devices = {"input": status.get("input", "?"), "output": status.get("output", "?")}
    old = store.calibration if store.exists() else None
    if old and abs(float(old.get("carrier_hz") or 0) - report.carrier_hz) > 1e-6 and not args.force:
        print(f"\nExisting calibration uses {old.get('carrier_hz')} Hz; new best is {report.carrier_hz:.0f} Hz.")
        print(INVALIDATION_WARNING)
        if not ask_yes_no("Keep the EXISTING carrier (recommended if you already recorded data)?", True):
            args.force = True
        else:
            chosen = dsp.with_overrides(carrier_hz=float(old["carrier_hz"]))
            report.carrier_hz = chosen.carrier_hz
            report.snr_db = dict(report.sweep).get(chosen.carrier_hz, report.snr_db)
    if args.simulate:
        report.measurements["simulated"] = True
    try:
        warnings = store.set_calibration(report.calibration(), chosen, args.volume, devices, force=args.force)
    except ProfileError as exc:
        print(f"\nERROR: {exc}")
        return 2
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"\nSaved calibration (carrier {chosen.carrier_hz:.0f} Hz, volume {args.volume}) to {store.manifest_path}")
    print(f"Next step: python enroll.py --profile {args.profile}" + (" --simulate" if args.simulate else ""))
    if args.simulate:
        print(SIMULATED_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
