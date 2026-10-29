# Troubleshooting

Start with `python doctor.py`. It prints PASS/WARN/FAIL for each check, with a suggested fix.
This page explains those fixes in more detail.

## "Carrier detection FAIL" / low SNR

The microphone can't hear the near-ultrasonic tone clearly.

1. **Use the built-in speaker and microphone.** Don't use headphones, a Bluetooth headset or HDMI audio.
2. **Unmute and raise the system volume a bit.** Then raise `--volume` gradually:
   `python doctor.py --volume 0.4`. Avoid going above ~0.5, because higher levels are more likely to be audible.
3. **Disable audio enhancements** (see below). They often filter out anything above 16 kHz.
4. **Try 48 kHz:** `python doctor.py --sample-rate 48000`. Some drivers band-limit 44.1 kHz streams.
5. **Pick devices explicitly:** `python doctor.py --list-devices`, then
   `--input-device "Microphone Array" --output-device "Speakers"`.
6. Some laptops simply roll off above ~18 kHz. The sweep will choose the best frequency your hardware supports.

## Windows: "Audio enhancements", noise suppression, Voice Clarity

Windows and many OEM audio suites (Realtek, Dolby, MaxxAudio, Waves, Nahimic) process the
microphone signal and can remove a steady tone. The doctor reports this as "carrier disappeared"
or "carrier level drifted".

- **Settings → System → Sound →** select your microphone → **Audio enhancements: Off**.
  Do the same for the speakers.
- Disable **Voice Clarity** / **Voice focus** / **Studio effects** (Windows Studio Effects on
  Copilot+ PCs) for the microphone.
- In OEM apps (Realtek Audio Console, Dolby Access, MaxxAudio, etc.), turn off noise
  suppression, beamforming and "conference" modes.
- Close Teams, Zoom, Discord and similar apps, or disable their noise suppression. They can
  switch the device into communications mode.
- Under **Sound Control Panel → Communications**, select **Do nothing** so Windows doesn't duck audio.
- Some drivers only allow raw capture in **exclusive mode**. Under **Sound Control Panel →
  Recording → Properties → Advanced**, enable *Allow applications to take exclusive control*.

## Bluetooth

Bluetooth audio isn't supported. The hands-free profile (used whenever the mic is active)
samples at 8–16 kHz, so there is no ultrasonic band. A2DP adds 100–300 ms of latency and often
low-passes at 16–20 kHz. The doctor warns when a device name looks like Bluetooth. Switch to
the laptop's built-in devices.

## Clipping

`clipping FAIL` means the input reaches full scale while the tone plays.

- Lower the tone: `python doctor.py --volume 0.2`.
- Lower the **microphone level** in the OS sound settings (70–80 % is a good start).
- Turn off "microphone boost".

## The tone is audible / bothers people or pets

Some people (especially children and young adults) and many pets can hear 17–19 kHz.

- Keep `--volume` low (default 0.3).
- Prefer a higher carrier. The doctor already picks the highest frequency within 3 dB of the best SNR.
- Stop EchoTouch when you're not using it.

Changing the carrier or volume later **invalidates your recorded clips and model**. Start a new
`--profile` if you need to change them.

## "Model was created with different audio/DSP settings"

The model's carrier, sample rate, FFT size, crop width or channel count doesn't match the
profile's current calibration. This is intentional: features computed with different settings
look different to the network. Either restore the original calibration or retrain (`python train.py`).

## "trained on SIMULATED data and cannot be used for real live control"

Simulated models exist only for tests and demos. Record real data with `enroll.py` and train on it.

## macOS permissions

- **Microphone:** System Settings → Privacy & Security → Microphone → enable your terminal
  (Terminal, iTerm, VS Code…). If the app was never prompted, run `tccutil reset Microphone` and try again.
- **Accessibility** (needed for key presses): System Settings → Privacy & Security →
  Accessibility → add/enable your terminal. Without it, pyautogui calls silently do nothing.
- Built-in MacBook speakers and mics generally reach 18–20 kHz, but "Voice Isolation" mic modes
  (Control Center → Mic Mode) must be set to **Standard**.

## Linux: Wayland, headless

- `pyautogui` uses X11. On a **pure Wayland** session it can't send input to native Wayland
  windows. Log in with an "Xorg/X11" session, or use `--dry-run`.
- Headless (SSH, CI): there's no display, so OS control is impossible. Recognition still works,
  so use `--dry-run`.
- `PortAudio could not be loaded`: `sudo apt install libportaudio2`.
- PipeWire/PulseAudio "echo cancellation" modules remove the carrier. Disable `module-echo-cancel`.

## Xruns (input overflow / output underflow)

A few xruns are harmless. Many xruns mean the CPU is busy or the buffer is too small. Close heavy
apps, keep the laptop on AC power, and avoid running training and live control at the same time.

## Gestures are not recognized / too many false triggers

- Check the **Live Control Center** spectrogram. A push should light up *above* the carrier
  line and a pull *below* it. If nothing lights up, it's an SNR or OS-processing problem (see above).
- Record **more varied rounds** (speed, distance, hand, posture) and **more idle** clips
  (typing, head movement, people walking by). Then retrain.
- Tune `--confidence` (default 0.55), `--smoothing` (3) and `--cooldown` (1.2 s) in `inference.py`,
  or override the learned motion gate with `--gate-db`.
- Advanced swipes are experimental. Disable them in `gestures.yaml` if they confuse the core gestures.
