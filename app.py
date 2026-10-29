"""EchoTouch dashboard (Streamlit).

    streamlit run app.py

Streamlit re-runs this script on every interaction. Audio, inference and training
therefore live in a long-lived :class:`engine.AppRuntime` (``st.cache_resource``)
on background threads; this script only *reads* their state. Live views use
``st.fragment(run_every=...)`` so only small parts of the page refresh.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from actions import ActionBackendError, ActionConfigError, environment_notes, load_actions
from config import (
    IDLE_SUBTYPES,
    SIMULATED_BANNER,
    DSPConfig,
    EnrollConfig,
    GestureSpec,
    InferenceConfig,
    TrainConfig,
    enabled_gestures,
    load_gestures,
)
from dsp import display_spectrogram, snr_at_frequency
from engine import AppRuntime, EngineError, LiveSession, open_store, profile_dsp
from profiles import ProfileError, ProfileStore, list_profiles

ACCENT = "#7C5CFF"
ACCENT_2 = "#22D3EE"
GOOD, WARN_C, BAD = "#34D399", "#FBBF24", "#F87171"
PLOT_BG = "rgba(0,0,0,0)"
#: Spectrogram colour range (dB above the static background); noise below ~8 dB stays dark.
SPEC_ZMIN, SPEC_ZMAX = 8.0, 35.0
#: Inferno-like scale, explicit so no theme can swap its colours; starts at the page background.
SPEC_COLORS = [
    [0.0, "#0B0D12"],
    [0.15, "#1B0C41"],
    [0.35, "#6A176E"],
    [0.55, "#BB3754"],
    [0.75, "#F98E09"],
    [1.0, "#FCFFA4"],
]

st.set_page_config(page_title="EchoTouch", page_icon="🖐️", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
h1, h2, h3 {{letter-spacing: -0.02em;}}
.et-hero {{font-size: 2.1rem; font-weight: 750; margin-bottom: 0;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}); -webkit-background-clip: text; color: transparent;}}
.et-sub {{opacity: .7; margin-top: -.2rem; margin-bottom: 1rem;}}
.et-card {{border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: .9rem 1.1rem;
  background: rgba(255,255,255,.03);}}
.et-pill {{display:inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 650; font-size: .9rem;}}
.et-sim {{background: repeating-linear-gradient(45deg, #3b2f00, #3b2f00 10px, #4a3b00 10px, #4a3b00 20px);
  color: #FDE68A; border-radius: 10px; padding: .55rem 1rem; font-weight: 700;
  text-align:center; margin-bottom: .8rem;}}
.et-empty {{border: 1px dashed rgba(255,255,255,.18); border-radius: 14px; padding: 2rem;
  text-align: center; opacity: .85;}}
.et-event {{font-family: ui-monospace, monospace; font-size: .85rem;}}
</style>
"""


@st.cache_resource
def get_runtime() -> AppRuntime:
    """One runtime per server process; survives reruns."""
    return AppRuntime()


def pill(text: str, color: str) -> str:
    return f'<span class="et-pill" style="background:{color}22;color:{color};border:1px solid {color}66">{text}</span>'


def empty_state(text: str) -> None:
    st.markdown(f'<div class="et-empty">{text}</div>', unsafe_allow_html=True)


def dark_layout(fig: go.Figure, height: int, **kw: Any) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"size": 12},
        **kw,
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def input_output_devices() -> tuple[list[str], list[str], str | None]:
    try:
        from audio_stream import import_sounddevice

        sd = import_sounddevice()
        devs = sd.query_devices()
        ins = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        outs = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d["max_output_channels"] > 0]
        return ins, outs, None
    except Exception as exc:  # no PortAudio / no devices
        return [], [], str(exc).splitlines()[0]


def ensure_sim_profile(store: ProfileStore) -> None:
    if not store.exists():
        dsp = DSPConfig()
        store.create(dsp, 0.3, {"input": "simulated mic", "output": "simulated speaker"})
        store.set_calibration({"carrier_hz": dsp.carrier_hz, "simulated": True, "tool": "app.py"}, dsp, 0.3)


def sidebar(rt: AppRuntime) -> dict[str, Any]:
    sb = st.sidebar
    sb.markdown("### 🖐️ EchoTouch")
    simulate = sb.toggle("Simulate mode", key="simulate", help="Use the acoustic simulator and sim/ (demo only).")
    if simulate:
        sb.markdown(f'<div class="et-sim">{SIMULATED_BANNER}</div>', unsafe_allow_html=True)
    existing = list_profiles(simulated=simulate)
    default = "demo" if simulate else "default"
    options = sorted(set(existing) | {default})
    profile = sb.selectbox("Profile", options, index=options.index(default) if default in options else 0)
    new_name = sb.text_input("…or create a new profile", placeholder="e.g. office-laptop")
    if new_name.strip():
        profile = new_name.strip()
    try:
        store = open_store(profile, simulate)
        if simulate:
            ensure_sim_profile(store)
    except (ProfileError, ValueError) as exc:
        sb.error(str(exc))
        st.stop()

    in_dev = out_dev = None
    if not simulate:
        ins, outs, err = input_output_devices()
        if err:
            sb.error(f"Audio unavailable: {err}")
        else:
            in_sel = sb.selectbox("Microphone", ["System default", *ins])
            out_sel = sb.selectbox("Speaker", ["System default", *outs])
            in_dev = None if in_sel == "System default" else int(in_sel.split(":")[0])
            out_dev = None if out_sel == "System default" else int(out_sel.split(":")[0])

    dsp: DSPConfig | None = None
    try:
        dsp = profile_dsp(store)
    except EngineError as exc:
        sb.warning(str(exc))

    sb.divider()
    running = rt.engine is not None and rt.engine.running
    c1, c2 = sb.columns(2)

    def start_audio() -> None:
        try:
            rt.ensure_audio(dsp, store.volume, simulate, in_dev, out_dev)  # type: ignore[arg-type]
        except EngineError as exc:
            st.session_state["audio_error"] = str(exc)

    # Buttons use on_click callbacks: they run even if a page fragment's timer rerun wins a race.
    c1.button("▶ Start audio", disabled=dsp is None or running, width="stretch", on_click=start_audio)
    c2.button("■ Stop", disabled=not running, width="stretch", on_click=rt.stop_audio)
    if err := st.session_state.pop("audio_error", None):
        sb.error(err)
    if running and rt.engine_key is not None and rt.engine_key[4] != simulate:
        sb.warning("Audio engine mode differs from the toggle: restart audio.")

    def snr_meter() -> None:
        """Static meter (refreshes on interaction). The Live page shows a continuously updated SNR."""
        eng = rt.engine
        if eng is None or not eng.running:
            st.caption("Audio stopped")
            return
        audio = eng.get_latest(1.0)
        snr = snr_at_frequency(audio, eng.sample_rate, eng.carrier_hz) if len(audio) >= 8192 else float("nan")
        color = GOOD if snr >= 20 else WARN_C if snr >= 10 else BAD
        label = f"{snr:.0f} dB" if np.isfinite(snr) else "…"
        st.markdown(f"**Carrier SNR** {pill(label, color)}", unsafe_allow_html=True)
        st.progress(float(np.clip((snr if np.isfinite(snr) else 0) / 60, 0, 1)))
        s = eng.status()
        st.caption(
            f"{'🟢' if s['running'] else '🔴'} {s['carrier_hz']:.0f} Hz · in: {s['input']} · "
            f"xruns {s['input_overflows'] + s['output_underflows']}"
        )

    with sb:
        snr_meter()
    sb.divider()
    cal = store.calibration if store.exists() else None
    sb.caption(
        f"Profile **{profile}** ({'sim/' if simulate else 'profiles/'}) · "
        + (f"carrier {dsp.carrier_hz:.0f} Hz · " if dsp else "")
        + (f"calibrated {str(cal.get('updated', ''))[:10]}" if cal else "not calibrated")
    )
    try:
        specs = enabled_gestures(load_gestures())
    except ValueError as exc:
        sb.error(f"gestures.yaml: {exc}")
        st.stop()
    return {
        "simulate": simulate,
        "profile": profile,
        "store": store,
        "dsp": dsp,
        "specs": specs,
        "in_dev": in_dev,
        "out_dev": out_dev,
    }


# --------------------------------------------------------------------------- #
# Tab 1: Enrollment Studio
# --------------------------------------------------------------------------- #
def get_controller(rt: AppRuntime, ctx: dict[str, Any]) -> Any:
    from enroll import EnrollmentController

    ctrl = rt.enrollment
    key = (ctx["profile"], ctx["simulate"], id(rt.engine))
    if ctrl is None or getattr(ctrl, "_ui_key", None) != key:
        ctrl = EnrollmentController(ctx["store"], rt.engine, ctx["specs"])
        ctrl._ui_key = key  # type: ignore[attr-defined]
        rt.enrollment = ctrl
    return ctrl


def counts_chart(counts: dict[str, int], specs: tuple[GestureSpec, ...]) -> go.Figure:
    names = [s.name for s in specs]
    vals = [counts.get(n, 0) for n in names]
    colors = [ACCENT if s.tier == "core" else WARN_C for s in specs]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, text=vals, textposition="outside"))
    for y, label in ((50, "feasibility (50)"), (150, "robust (150)")):
        fig.add_hline(y=y, line_dash="dot", line_color="rgba(255,255,255,.35)", annotation_text=label)
    return dark_layout(fig, 280, yaxis_title="clips", showlegend=False)


def clip_preview(audio: np.ndarray, dsp: DSPConfig) -> go.Figure:
    x = audio[:, 0] if audio.ndim == 2 else audio
    step = max(1, len(x) // 3000)
    t = np.arange(len(x))[::step] / dsp.sample_rate
    spec = display_spectrogram(audio, dsp)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("waveform", "Doppler spectrogram"))
    fig.add_trace(go.Scattergl(x=t, y=x[::step], line={"color": ACCENT_2, "width": 1}), 1, 1)
    frames_t = np.arange(spec.shape[1]) * dsp.hop / dsp.sample_rate
    fig.add_trace(
        go.Heatmap(
            z=spec,
            x=frames_t,
            y=dsp.relative_freqs(),
            colorscale=SPEC_COLORS,
            zmin=SPEC_ZMIN,
            zmax=SPEC_ZMAX,
            showscale=False,
        ),
        1,
        2,
    )
    fig.update_yaxes(title_text="Hz vs carrier", row=1, col=2)
    return dark_layout(fig, 260, showlegend=False)


def tab_enrollment(rt: AppRuntime, ctx: dict[str, Any]) -> None:
    store, specs, dsp = ctx["store"], ctx["specs"], ctx["dsp"]
    st.subheader("Enrollment Studio")
    st.caption("Record YOUR acoustic signature. Vary speed, distance and hand position between rounds.")
    if dsp is None:
        empty_state(
            "This profile is not calibrated.<br>Run <code>python doctor.py --profile "
            f"{ctx['profile']}</code> first (or turn on Simulate mode)."
        )
        return
    if rt.engine is None or not rt.engine.running:
        empty_state("Audio is stopped. Press <b>▶ Start audio</b> in the sidebar.")
        counts = store.counts() if store.exists() else {}
        if counts:
            st.plotly_chart(counts_chart(dict(counts), specs), width="stretch", theme=None, key="counts_chart_idle")
        return
    try:
        ctrl = get_controller(rt, ctx)
    except ProfileError as exc:
        st.error(str(exc))
        return
    preview_on = st.toggle("Live spectrogram preview", key="enroll_preview")

    # Everything interactive lives inside this fragment, so the optional live-preview
    # timer can never swallow a click on Record/Undo.
    @st.fragment(run_every=0.5 if preview_on else None)
    def enroll_panel() -> None:
        left, right = st.columns([0.42, 0.58], gap="large")
        counts = store.counts()
        with left:
            labels = {s.label: s.name for s in specs}
            gesture = labels[st.selectbox("Gesture", list(labels), key="enroll_gesture")]
            spec = ctrl.spec(gesture)
            st.markdown(f"**How:** {spec.instruction}")
            if spec.experimental:
                st.warning("Experimental gesture: Doppler sees radial motion only, so accuracy may be limited.")
            sub = None
            if gesture == "idle":
                auto = ctrl.idle_subtype(int(counts.get("idle", 0)) % EnrollConfig.clips_per_round)
                names = [n for n, _ in IDLE_SUBTYPES]
                sub = st.selectbox(
                    "Idle type",
                    names,
                    index=names.index(auto[0]),
                    format_func=lambda n: f"{n} – {dict(IDLE_SUBTYPES)[n]}",
                )
            rcol1, rcol2 = st.columns(2)
            if rcol1.button("⟳ New round", width="stretch") or ctrl.round == 0:
                ctrl.begin_round()
            st.info(f"Session `{ctrl.session_id}` · round **{ctrl.round}** — {ctrl.variation(ctrl.round)}")
            record = rcol2.button("● Record clip", type="primary", width="stretch")
            status = st.empty()
            if record:
                for i in range(EnrollConfig.countdown_seconds, 0, -1):
                    status.markdown(f"### Get ready: {gesture.upper()} … {i}")
                    time.sleep(0.6)
                status.markdown("### GO!")
                attempt = ctrl.record(gesture, sub)
                if attempt.quality.accepted:
                    ctrl.save(attempt)
                st.session_state["last_attempt"] = attempt
                status.empty()
                counts = store.counts()
            if st.button("↶ Undo last clip", width="stretch"):
                rec = ctrl.undo_last()
                st.session_state.pop("last_attempt", None)
                st.toast(f"Removed {rec['gesture']} clip" if rec else "Nothing to undo in this session")
                counts = store.counts()
            attempt = st.session_state.get("last_attempt")
            if attempt is not None:
                q = attempt.quality
                st.markdown(
                    pill("SAVED" if q.accepted else "REJECTED", GOOD if q.accepted else BAD)
                    + f" &nbsp; **{attempt.gesture}** · SNR {q.snr_db:.0f} dB · motion {q.motion_db:.1f} dB",
                    unsafe_allow_html=True,
                )
                for m in q.messages:
                    st.caption("⚠️ " + m)
        with right:
            if not counts:
                empty_state("No clips yet. Pick a gesture and press <b>Record clip</b>.")
            st.plotly_chart(counts_chart(dict(counts), specs), width="stretch", theme=None, key="counts_chart")
            if attempt is not None:
                st.plotly_chart(clip_preview(attempt.audio, dsp), width="stretch", theme=None, key="clip_preview")
            if preview_on and rt.engine is not None and rt.engine.running:
                audio = rt.engine.get_latest(3.0)
                if len(audio) >= dsp.n_fft * 2:
                    fig = spectrogram_figure(audio, dsp, 220, "live preview")
                    st.plotly_chart(fig, width="stretch", theme=None, key="enroll_live_spec")

    enroll_panel()


# --------------------------------------------------------------------------- #
# Tab 2: Model Training
# --------------------------------------------------------------------------- #
def curves_figure(history: list[dict[str, Any]]) -> go.Figure:
    ep = [h["epoch"] for h in history]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("loss", "accuracy"))
    fig.add_trace(
        go.Scatter(x=ep, y=[h["train_loss"] for h in history], name="train loss", line={"color": ACCENT}), 1, 1
    )
    fig.add_trace(go.Scatter(x=ep, y=[h["val_loss"] for h in history], name="val loss", line={"color": ACCENT_2}), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=[h["train_acc"] for h in history], name="train acc", line={"color": ACCENT}), 1, 2)
    fig.add_trace(go.Scatter(x=ep, y=[h["val_acc"] for h in history], name="val acc", line={"color": ACCENT_2}), 1, 2)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    return dark_layout(fig, 300, legend={"orientation": "h", "y": -0.2})


def confusion_figure(cm: list[list[int]], classes: list[str]) -> go.Figure:
    arr = np.asarray(cm, dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=classes,
            y=classes,
            colorscale=[[0.0, "#141824"], [1.0, ACCENT]],
            zmin=0,
            zmax=1,
            text=arr.astype(int),
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed", title="true")
    fig.update_xaxes(title="predicted")
    return dark_layout(fig, 360)


def show_report(report: dict[str, Any]) -> None:
    m = report["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{(m['accuracy'] or 0):.1%}")
    c2.metric("Balanced accuracy", f"{(m['balanced_accuracy'] or 0):.1%}")
    c3.metric("Validation clips", report["n_val"], help=f"split by {report['split']['strategy']}")
    st.caption(f"Evaluated on {report['evaluated_on']} · source: {report['source']} · gate {report['gate_db']} dB")
    left, right = st.columns([0.55, 0.45])
    left.plotly_chart(
        confusion_figure(m["confusion_matrix"], report["classes"]),
        width="stretch",
        theme=None,
        key=f"cm_{report.get('best_epoch')}_{report['n_val']}",
    )
    rows = [
        {
            "class": c,
            "tier": v["tier"],
            "precision": v["precision"],
            "recall": v["recall"],
            "support": v["support"],
            "weak": "⚠️" if c in m["weak_classes"] else "",
        }
        for c, v in m["per_class"].items()
    ]
    right.dataframe(rows, width="stretch", hide_index=True)
    for w in report.get("warnings", []):
        right.warning(w)


def tab_training(rt: AppRuntime, ctx: dict[str, Any]) -> None:
    store, specs, simulate = ctx["store"], ctx["specs"], ctx["simulate"]
    st.subheader("Model Training")
    counts = store.counts() if store.exists() else {}
    if not counts:
        empty_state(
            "No data yet: go to <b>Enrollment Studio</b>"
            + (" or generate a simulated demo dataset below." if simulate else ".")
        )
    if simulate:
        with st.expander("Generate a simulated demo dataset (sim/ only)"):
            n = st.slider("Clips per gesture per round", 5, 40, 15)
            if st.button("Generate"):
                from simulator import simulate_dataset

                with st.spinner("Simulating…"):
                    simulate_dataset(
                        store, [s.name for s in specs], n, sessions=2, rounds=2, seed=int(time.time()) % 10000
                    )
                st.rerun()
    if not counts:
        return
    sessions = {c["session_id"] for c in store.clips()}
    st.caption(
        f"{sum(counts.values())} clips · {len(sessions)} session(s) · "
        + " · ".join(f"{s.name}: {counts.get(s.name, 0)}" for s in specs)
    )
    with st.expander("Training settings"):
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Max epochs", 5, 100, TrainConfig.epochs)
        batch = c2.select_slider("Batch size", [8, 16, 32, 64], TrainConfig.batch_size)
        lr = c3.select_slider("Learning rate", [3e-4, 1e-3, 3e-3], TrainConfig.lr)
    prog = rt.training.progress()
    running = prog["status"] == "running"

    def start_training() -> None:
        try:
            cfg = replace(TrainConfig(), epochs=epochs, batch_size=batch, lr=lr)
            rt.training.start(store, specs, cfg, allow_simulated=simulate)
        except EngineError as exc:
            st.session_state["train_error"] = str(exc)

    st.button("🚀 Start training", type="primary", disabled=running, on_click=start_training)
    if err := st.session_state.pop("train_error", None):
        st.error(err)

    @st.fragment(run_every=1.0 if running else None)
    def progress_view() -> None:
        p = rt.training.progress()
        epochs_ev = [e for e in p["events"] if e["type"] == "epoch"]
        if p["status"] == "running" and st.button("■ Stop training"):
            rt.training.stop()
        if p["status"] == "running":
            total = epochs_ev[-1]["epochs"] if epochs_ev else 1
            st.progress(len(epochs_ev) / max(total, 1), text=f"epoch {len(epochs_ev)}/{total}")
        for e in p["events"]:
            if e["type"] == "warning":
                st.warning(e["message"])
            elif e["type"] == "info":
                st.caption(e["message"])
        if epochs_ev:
            st.plotly_chart(curves_figure(epochs_ev), width="stretch", theme=None, key="curves")
        if p["status"] == "error":
            st.error(f"Training failed: {p['error']}")
        if p["status"] in ("done", "stopped") and p["result"] is not None:
            st.success(f"Saved model to {p['result'].model_path}")
            show_report(p["result"].report)
        if running and p["status"] != "running":
            st.rerun()  # leave polling mode

    progress_view()
    if prog["status"] == "idle" and store.report_path.exists():
        import json

        with open(store.report_path, encoding="utf-8") as fh:
            st.markdown("#### Last training report")
            show_report(json.load(fh))


# --------------------------------------------------------------------------- #
# Tab 3: Live Control Center
# --------------------------------------------------------------------------- #
def spectrogram_figure(audio: np.ndarray, dsp: DSPConfig, height: int, title: str = "") -> go.Figure:
    spec = display_spectrogram(audio, dsp)
    t = (np.arange(spec.shape[1]) * dsp.hop + dsp.n_fft / 2) / dsp.sample_rate - len(audio) / dsp.sample_rate
    fig = go.Figure(
        go.Heatmap(
            z=spec,
            x=t,
            y=dsp.relative_freqs(),
            colorscale=SPEC_COLORS,
            zmin=SPEC_ZMIN,
            zmax=SPEC_ZMAX,
            zsmooth="best",
            colorbar={"title": "dB", "thickness": 10},
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,.25)", line_dash="dot")
    fig.update_xaxes(title="seconds")
    fig.update_yaxes(title="Hz vs carrier")
    return dark_layout(fig, height, title={"text": title, "font": {"size": 13}})


def probs_figure(probs: dict[str, float], threshold: float) -> go.Figure:
    names = list(probs) or ["—"]
    vals = [probs.get(n, 0.0) for n in names]
    top = int(np.argmax(vals)) if probs else -1
    colors = [ACCENT_2 if i == top else "rgba(124,92,255,.55)" for i in range(len(names))]
    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors, text=[f"{v:.0%}" for v in vals], textposition="auto"
        )
    )
    fig.add_vline(x=threshold, line_dash="dot", line_color="rgba(255,255,255,.5)")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return dark_layout(fig, 300, showlegend=False)


def tab_live(rt: AppRuntime, ctx: dict[str, Any]) -> None:
    store, dsp, simulate = ctx["store"], ctx["dsp"], ctx["simulate"]
    st.subheader("Live Control Center")
    if dsp is None:
        empty_state("Calibrate this profile first: <code>python doctor.py</code>.")
        return
    if not store.model_path.exists():
        empty_state("No model yet: record clips in <b>Enrollment Studio</b>, then train in <b>Model Training</b>.")
        return
    if rt.engine is None or not rt.engine.running:
        empty_state("Audio is stopped. Press <b>▶ Start audio</b> in the sidebar.")
        return
    if rt.live is None:

        def start_live() -> None:
            try:
                rt.live = LiveSession.open(
                    ctx["profile"], simulate, load_actions(), InferenceConfig(), engine=rt.engine
                )
                rt.live.start()
            except (EngineError, ActionConfigError, ActionBackendError) as exc:
                st.session_state["live_error"] = str(exc)

        # on_click callbacks survive concurrent fragment reruns (plain button returns may not).
        st.button("▶ Start recognition", type="primary", on_click=start_live)
        if err := st.session_state.pop("live_error", None):
            st.error(err)
        return

    # Controls live inside the auto-refreshing fragment: widget clicks then rerun only the
    # fragment and are never lost to a concurrent timer rerun.
    @st.fragment(run_every=0.3)
    def live_panel() -> None:
        eng, sess = rt.engine, rt.live
        if eng is None or sess is None or not eng.running:
            st.rerun(scope="app")
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
        if c1.button("■ Stop recognition", width="stretch"):
            rt.stop_live()
            st.rerun(scope="app")
        control = c2.toggle(
            "OS control",
            value=False,
            key="os_control",
            help="OFF by default. When ON, gestures trigger the actions in actions.yaml.",
        )
        dry = c3.toggle(
            "Dry-run (print only)",
            value=True,
            key="dry_run",
            disabled=simulate,
            help="Simulate mode always uses dry-run.",
        )
        sess.worker.executor.dry_run = dry or simulate
        if control != sess.worker.executor.enabled:
            try:
                sess.worker.set_control(control)
            except ActionBackendError as exc:
                st.error(str(exc))
        if control and not dry:
            for note in environment_notes():
                st.caption("ℹ️ " + note)
        if simulate:
            with c4:
                eng.auto_demo = st.toggle("Auto demo gestures", key="auto_demo")  # type: ignore[union-attr]
            pcols = st.columns(len(sess.class_names))
            for col, g in zip(pcols, sess.class_names):
                if g != "idle" and col.button(f"🎭 perform {g}", key=f"perf_{g}", width="stretch"):
                    eng.perform(g)  # type: ignore[union-attr]
        s = sess.worker.snapshot()
        audio = eng.get_latest(4.0)
        top, side = st.columns([0.68, 0.32], gap="medium")
        with top:
            if len(audio) >= dsp.n_fft * 2:
                st.plotly_chart(
                    spectrogram_figure(audio, dsp, 380, "Doppler spectrogram (dB above static)"),
                    width="stretch",
                    theme=None,
                    key="live_spec",
                )
        with side:
            gate = pill("● MOTION", GOOD) if s.gate_open else pill("○ quiet", "#94A3B8")
            cool = " " + pill("cooldown", WARN_C) if s.cooldown else ""
            ctrl_pill = (
                pill("CONTROL ON" + (" (dry-run)" if s.dry_run else ""), BAD if not s.dry_run else WARN_C)
                if s.control_enabled
                else pill("control off", "#94A3B8")
            )
            st.markdown(f"{gate}{cool}<br><br>{ctrl_pill}", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Motion", f"{s.motion_db:.1f} dB", help=f"gate at {sess.recognizer.gate.threshold_db:.1f} dB")
            m2.metric("SNR", f"{s.snr_db:.0f} dB" if s.snr_db is not None else "…")
            st.caption(f"step {s.avg_step_ms:.1f} ms · classified {s.classified_steps}/{s.steps}")
            if s.error:
                st.error(s.error)
        p1, p2 = st.columns([0.5, 0.5], gap="medium")
        with p1:
            st.markdown("**Class probabilities**")
            st.plotly_chart(
                probs_figure(s.probs or {c: 0.0 for c in sess.class_names}, sess.recognizer.smoother.confidence),
                width="stretch",
                theme=None,
                key="live_probs",
            )
        with p2:
            st.markdown("**Recent gestures**")
            if not s.events:
                st.caption("none yet — try a push!")
            for ev in reversed(s.events[-12:]):
                st.markdown(
                    f'<div class="et-event">{ev.wall_time} &nbsp; <b>{ev.gesture}</b> '
                    f"({ev.confidence:.0%}) → {ev.action}</div>",
                    unsafe_allow_html=True,
                )

    live_panel()


# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    rt = get_runtime()
    ctx = sidebar(rt)
    st.markdown(
        '<div class="et-hero">EchoTouch</div><div class="et-sub">Touchless gestures from your laptop\'s '
        "speaker and mic · Doppler sonar</div>",
        unsafe_allow_html=True,
    )
    if ctx["simulate"]:
        st.markdown(
            f'<div class="et-sim">⚠ {SIMULATED_BANNER} Clips and models live in sim/ and can never '
            "control your computer.</div>",
            unsafe_allow_html=True,
        )
    # Page navigation (not st.tabs): only the active page renders, so only its live
    # fragments refresh in the background.
    pages = {
        "🎙️ Enrollment Studio": tab_enrollment,
        "🧠 Model Training": tab_training,
        "📡 Live Control Center": tab_live,
    }
    page = st.segmented_control("Page", list(pages), default=list(pages)[0], key="page", label_visibility="collapsed")
    pages[page or list(pages)[0]](rt, ctx)


main()
