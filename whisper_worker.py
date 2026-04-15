#!/usr/bin/env python3
"""
whisper_worker.py — Persistent faster-whisper daemon for smart_voice.sh.
Keeps large-v3 loaded in GPU memory. Watches for audio files to transcribe.
Includes energy gate (rejects silence) and audio normalization before transcription.

Protocol:
  smart_voice.sh writes audio path to /tmp/whisper_request
  Worker transcribes it, writes result to /tmp/whisper_result
  Worker removes /tmp/whisper_request when done
"""

import time, os, wave, struct, math
import numpy as np

REQUEST_FILE = "/tmp/whisper_request"
RESULT_FILE  = "/tmp/whisper_result"
READY_FILE   = "/tmp/whisper_worker_ready"

# Minimum RMS energy to attempt transcription (below = silence/noise → skip)
MIN_RMS_THRESHOLD = 150   # out of 32768 — tune up if hallucinating, down if missing speech

# Known Whisper hallucination phrases to discard
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thanks for watching!", "thank you.",
    "you", "bye", "goodbye", "bye.", "see you next time", "take care",
    "thanks", "like and subscribe", "subscribe", "the end", "silence",
    "you you you", ".", "", " "
}

print("[WhisperWorker] Loading large-v3 model...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
print("[WhisperWorker] Model ready.", flush=True)

open(READY_FILE, "w").close()

def load_and_normalize(audio_path):
    """Load WAV, check energy, normalize to consistent level for Whisper."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Energy gate: compute RMS
        rms = math.sqrt(np.mean(samples ** 2))
        print(f"[WhisperWorker] Audio RMS: {rms:.1f} (min: {MIN_RMS_THRESHOLD})", flush=True)

        if rms < MIN_RMS_THRESHOLD:
            print("[WhisperWorker] Audio too quiet — skipping (silence/noise)", flush=True)
            return None

        # Normalize to -3dB (0.707 of full scale) so Whisper gets consistent input
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples * (0.707 * 32767.0 / peak)

        return samples / 32768.0   # Whisper expects float32 in [-1, 1]

    except Exception as e:
        print(f"[WhisperWorker] WAV load error: {e}", flush=True)
        return None


while True:
    if os.path.exists(REQUEST_FILE):
        try:
            audio_path = open(REQUEST_FILE).read().strip()
            os.remove(REQUEST_FILE)

            print(f"[WhisperWorker] Transcribing: {audio_path}", flush=True)

            audio_data = load_and_normalize(audio_path)

            if audio_data is None:
                with open(RESULT_FILE, "w") as f:
                    f.write("")
            else:
                segments, _ = model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=5,
                    vad_filter=False,   # PTT: user controls start/stop — whole recording is speech
                    without_timestamps=True
                )
                text = "".join(s.text for s in segments).strip()

                # Filter hallucinations
                if text.lower().rstrip(".,!?") in HALLUCINATION_PHRASES:
                    print(f"[WhisperWorker] Hallucination filtered: '{text}'", flush=True)
                    text = ""

                print(f"[WhisperWorker] Result: '{text}'", flush=True)
                with open(RESULT_FILE, "w") as f:
                    f.write(text)

        except Exception as e:
            print(f"[WhisperWorker] Error: {e}", flush=True)
            with open(RESULT_FILE, "w") as f:
                f.write("")

    time.sleep(0.2)
