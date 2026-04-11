#!/usr/bin/env python3
"""
Nexus Voice Daemon v2.0
Always-listening voice command system for Nexus Local

System-wide voice control that works whether or not the Nexus app is running.
Uses Whisper large-v3 for STT, Piper (Obadiah) for TTS, and Ollama for AI.

Trigger phrases:
  - "write this down" → Records speech, saves as knowledge card to ~/.nexus/cards/
  - "gogo ollama" → Wakes Hermes (light model), who can escalate to bigger models
  - "run that" → Auto-approve for 90 seconds, execute plan
  - "nevermind" → Cancel current operation
  - "stand by" / "hold on" → Pause listening until next wake phrase

Mouse side button → Dictation mode (types speech into focused window via xdotool)

Requirements:
  pip install openai-whisper sounddevice numpy webrtcvad requests --break-system-packages

Usage:
  python nexus-voice-daemon.py
  
Or as systemd service:
  systemctl --user enable nexus-voice-daemon
  systemctl --user start nexus-voice-daemon
"""

import os
import sys
import json
import time
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    import whisper
except ImportError:
    print("ERROR: openai-whisper not installed")
    print("Run: pip install openai-whisper --break-system-packages")
    sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("ERROR: webrtcvad not installed")
    print("Run: pip install webrtcvad --break-system-packages")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    print("Run: pip install requests --break-system-packages")
    sys.exit(1)


# ============================================
# Configuration
# ============================================

NEXUS_CARDS_DIR = Path.home() / ".nexus" / "cards"

CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "frame_duration_ms": 30,
    "silence_threshold_sec": 2.0,
    "vad_aggressiveness": 0,
    "max_recording_sec": 120.0,
    
    # Whisper settings - using faster-whisper for speed with large-v3
    "whisper_model": "large-v3",
    "whisper_device": "cuda",
    
    # Energy gating - reject audio below this threshold (prevents hallucinations)
    "min_audio_energy": 0.0001,
    "min_audio_length_sec": 0.5,
    
    "ollama_endpoint": "http://localhost:11434",
    
    # Model roster - Hermes is the gatekeeper/dispatcher
    "models": {
        "gatekeeper": "hermes3:8b",       # Light model, always-on, decides if big kid needed
        "coding": "devstral-small-2:24b",
        "coding_fast": "qwen2.5-coder:7b",
        "reasoning": "deepseek-r1:32b",
        "vision": "qwen3-vl:30b",
        "general": "qwen2.5:32b",
    },
    
    # Keywords that trigger escalation from Hermes to a specialist
    "escalation_rules": {
        "coding": ["code", "function", "bug", "refactor", "debug", "script", "variable", "class", 
                   "method", "api", "endpoint", "database", "query", "sql", "typescript", "python", 
                   "rust", "javascript", "import", "export", "module", "component", "error", "fix",
                   "implement", "create a", "write a", "build a"],
        "reasoning": ["plan", "analyze", "think through", "complex", "strategy", "compare", 
                      "evaluate", "design", "architect", "decide", "trade-off", "pros and cons", 
                      "should i", "help me understand", "explain why", "break down", "step by step"],
        "vision": ["look at", "screenshot", "image", "picture", "photo", "what do you see", 
                   "analyze this image", "read this"],
    },
    
    # Piper TTS settings - Obadiah voice
    "tts_enabled": True,
    "tts_backend": "piper",
    "piper_path": "/usr/bin/piper",
    "piper_model": str(Path.home() / ".local/share/piper/en_GB-semaine-medium.onnx"),
    "piper_speaker": "2",           # obadiah=2, prudence=0, spike=1, poppy=3
    "piper_length_scale": 0.75,     # Speed: 1.0 / 1.33 ≈ 0.75 (faster)
    "piper_noise_scale": 0.33,      # Tone/variance
    # Fallback espeak settings
    "tts_speed": 1.33,
    "tts_pitch": -33,
    
    # IPC files - how daemon communicates with Nexus app
    "command_file": "/tmp/nexus_voice_command.json",
    "state_file": "/tmp/nexus_voice_state.json",
    "auto_approve_duration_sec": 90,
    
    # Dictation trigger file (written by xbindkeys when mouse button pressed)
    "dictate_trigger_file": "/tmp/nexus_dictate_trigger",
    
    # Sound settings - only chirp in active conversation mode
    "sound_enabled": True,
    "chirp_on_passive_listen": False,
    "chirp_on_active_listen": True,
    "sounds": {
        "listening": "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga",
        "success": "/usr/share/sounds/freedesktop/stereo/complete.oga",
        "cancel": "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga",
        "error": "/usr/share/sounds/freedesktop/stereo/dialog-error.oga",
        "standby": "/usr/share/sounds/freedesktop/stereo/service-logout.oga",
        "dictate_start": "/usr/share/sounds/freedesktop/stereo/message.oga",
        "dictate_end": "/usr/share/sounds/freedesktop/stereo/bell.oga",
    },
}


# ============================================
# State Management
# ============================================

class DaemonState:
    def __init__(self):
        self.in_conversation = False
        self.current_model = None
        self.conversation_history = []
        self.auto_approve_until = 0
        self.last_plan = None
        self.paused = False
        self.dictation_mode = False
        self.load()
    
    def load(self):
        try:
            if os.path.exists(CONFIG["state_file"]):
                with open(CONFIG["state_file"], "r") as f:
                    data = json.load(f)
                    self.auto_approve_until = data.get("auto_approve_until", 0)
        except:
            pass
    
    def save(self):
        try:
            with open(CONFIG["state_file"], "w") as f:
                json.dump({
                    "auto_approve_until": self.auto_approve_until,
                    "in_conversation": self.in_conversation,
                    "current_model": self.current_model,
                    "paused": self.paused,
                }, f)
        except:
            pass
    
    def is_auto_approve_active(self):
        return time.time() < self.auto_approve_until
    
    def activate_auto_approve(self):
        self.auto_approve_until = time.time() + CONFIG["auto_approve_duration_sec"]
        self.save()
        print(f"[State] Auto-approve active for {CONFIG['auto_approve_duration_sec']} seconds")
    
    def start_conversation(self, model):
        self.in_conversation = True
        self.current_model = model
        self.conversation_history = []
        self.paused = False
        self.save()
    
    def end_conversation(self):
        self.in_conversation = False
        self.current_model = None
        self.conversation_history = []
        self.last_plan = None
        self.save()
    
    def enter_standby(self):
        self.paused = True
        self.in_conversation = False
        self.save()
        print("[State] Entering standby mode")
    
    def exit_standby(self):
        self.paused = False
        self.save()
        print("[State] Exiting standby mode")


# ============================================
# Audio Recording
# ============================================

class AudioRecorder:
    def __init__(self):
        self.sample_rate = CONFIG["sample_rate"]
        self.frame_duration_ms = CONFIG["frame_duration_ms"]
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(CONFIG["vad_aggressiveness"])
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._interrupt_flag = False
        
    def interrupt(self):
        self._interrupt_flag = True
        
    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def record_until_silence(self, verbose=True):
        self.is_recording = True
        self._interrupt_flag = False
        self.audio_queue = queue.Queue()
        
        frames = []
        silence_frames = 0
        silence_threshold = int(CONFIG["silence_threshold_sec"] * 1000 / self.frame_duration_ms)
        max_frames = int(CONFIG["max_recording_sec"] * 1000 / self.frame_duration_ms)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CONFIG["channels"],
                dtype=np.int16,
                blocksize=self.frame_size,
                callback=self._audio_callback
            ):
                if verbose:
                    print(f"[Recording] Listening... (stop after {CONFIG['silence_threshold_sec']}s silence)")
                
                frame_count = 0
                speech_detected = False
                
                while frame_count < max_frames:
                    if self._interrupt_flag:
                        break
                        
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        frames.append(audio_chunk)
                        frame_count += 1
                        
                        audio_bytes = audio_chunk.flatten().tobytes()
                        # Use BOTH VAD and energy threshold - either one says speech = speech
                        try:
                            vad_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
                        except:
                            vad_speech = False
                        
                        # Energy threshold as fallback/ дополнение
                        energy = np.abs(audio_chunk).mean()
                        energy_speech = energy > 300  # Lower threshold
                        
                        # Accept if either VAD or energy detects speech
                        is_speech = vad_speech or energy_speech
                        
                        if is_speech:
                            silence_frames = 0
                            speech_detected = True
                        else:
                            silence_frames += 1
                        
                        if speech_detected and silence_frames >= silence_threshold:
                            if verbose:
                                print(f"[Recording] Silence detected, stopping...")
                            break
                            
                    except queue.Empty:
                        continue
                        
        finally:
            self.is_recording = False
        
        if frames:
            return np.concatenate(frames)
        return None


# ============================================
# Whisper Transcription
# ============================================

class Transcriber:
    def __init__(self):
        print(f"[Whisper] Loading faster-whisper model '{CONFIG['whisper_model']}' on {CONFIG['whisper_device']}...")
        from faster_whisper import WhisperModel
        
        try:
            self.model = WhisperModel(
                CONFIG["whisper_model"], 
                device=CONFIG["whisper_device"], 
                compute_type="int8_float16" if CONFIG["whisper_device"] == "cuda" else "int8"
            )
            print(f"[Whisper] Model loaded on GPU ✓")
        except Exception as e:
            print(f"[Whisper] GPU failed ({e}), falling back to CPU")
            self.model = WhisperModel(CONFIG["whisper_model"], device="cpu", compute_type="int8")
        
    def transcribe(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        audio_float = audio_data.astype(np.float32).flatten() / 32768.0
        
        # Energy gating
        audio_energy = np.abs(audio_float).mean()
        if audio_energy < CONFIG["min_audio_energy"]:
            print(f"[Whisper] Audio energy too low ({audio_energy:.4f}), skipping")
            return ""
        
        # Length check
        audio_length_sec = len(audio_float) / CONFIG["sample_rate"]
        if audio_length_sec < CONFIG["min_audio_length_sec"]:
            print(f"[Whisper] Audio too short ({audio_length_sec:.2f}s), skipping")
            return ""
        
        segments, info = self.model.transcribe(
            audio_float,
            language="en",
            beam_size=5,
            vad_filter=True,
            without_timestamps=True
        )
        
        text = "".join([segment.text for segment in segments]).strip()
        
        # Filter out common hallucinations on near-silence
        hallucination_phrases = [
            "thank you", "thanks", "you", "bye", "goodbye", 
            "thanks for watching", "subscribe", "like and subscribe",
            "see you next time", "take care", "have a good day",
            "you you you", "the end", "silence",
        ]
        if text.lower() in hallucination_phrases:
            print(f"[Whisper] Filtered hallucination: '{text}'")
            return ""
        
        return text


# ============================================
# Text-to-Speech (Piper with fallback to espeak)
# ============================================

class Speaker:
    def __init__(self):
        self.piper_available = self._check_piper()
        if self.piper_available:
            print(f"[TTS] Using Piper TTS (Obadiah) ✓")
        else:
            print(f"[TTS] Piper not available, falling back to espeak")
    
    def _check_piper(self):
        if CONFIG["tts_backend"] != "piper":
            return False
        
        piper_paths = [
            CONFIG["piper_path"],
            "/usr/bin/piper",
            "/usr/local/bin/piper",
            str(Path.home() / ".local/bin/piper"),
        ]
        
        piper_found = None
        for p in piper_paths:
            if os.path.exists(p):
                piper_found = p
                break
        
        if not piper_found:
            print(f"[TTS] Piper binary not found in: {piper_paths}")
            return False
        
        CONFIG["piper_path"] = piper_found
        
        if not os.path.exists(CONFIG["piper_model"]):
            print(f"[TTS] Piper model not found: {CONFIG['piper_model']}")
            return False
        
        print(f"[TTS] Found Piper at: {piper_found}")
        print(f"[TTS] Using voice: {CONFIG['piper_model']}")
        return True
    
    def speak(self, text):
        if not CONFIG["tts_enabled"] or not text:
            return
        
        # Truncate very long responses for speech
        if len(text) > 500:
            text = text[:500] + "... I'll stop there."
        
        print(f"[TTS] Speaking: {text[:80]}...")
        
        try:
            if self.piper_available:
                self._speak_piper(text)
            else:
                self._speak_espeak(text)
        except Exception as e:
            print(f"[TTS] Error: {e}")
    
    def _speak_piper(self, text):
        wav_path = f"/tmp/nexus_speech_{int(time.time() * 1000)}.wav"
        
        process = subprocess.Popen(
            [
                CONFIG["piper_path"],
                "--model", CONFIG["piper_model"],
                "--speaker", CONFIG["piper_speaker"],
                "--length-scale", str(CONFIG["piper_length_scale"]),
                "--noise-scale", str(CONFIG["piper_noise_scale"]),
                "--output_file", wav_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=text.encode('utf-8'))
        
        if os.path.exists(wav_path):
            subprocess.run(["paplay", wav_path], capture_output=True)
            os.remove(wav_path)
    
    def _speak_espeak(self, text):
        speed = int(175 * CONFIG["tts_speed"])
        pitch = int(50 + CONFIG["tts_pitch"])
        subprocess.run(
            ["espeak", "-s", str(speed), "-p", str(pitch), text],
            capture_output=True
        )


# ============================================
# Model Router with Hermes as Gatekeeper
# ============================================

class ModelRouter:
    def __init__(self):
        self.endpoint = CONFIG["ollama_endpoint"]
    
    def select_model(self, text):
        """Hermes-first routing: start with the gatekeeper, escalate if needed."""
        text_lower = text.lower()
        
        # Check for explicit escalation keywords
        for keyword in CONFIG["escalation_rules"]["vision"]:
            if keyword in text_lower:
                model = CONFIG["models"]["vision"]
                print(f"[Router] Vision task → {model}")
                return model
        
        for keyword in CONFIG["escalation_rules"]["reasoning"]:
            if keyword in text_lower:
                model = CONFIG["models"]["reasoning"]
                print(f"[Router] Reasoning task → {model}")
                return model
        
        for keyword in CONFIG["escalation_rules"]["coding"]:
            if keyword in text_lower:
                model = CONFIG["models"]["coding_fast"] if len(text) < 100 else CONFIG["models"]["coding"]
                print(f"[Router] Coding task → {model}")
                return model
        
        # Default: start with Hermes (gatekeeper). He's fast and can handle simple stuff.
        # If the task is too complex, Hermes will say so and we can escalate.
        model = CONFIG["models"]["gatekeeper"]
        print(f"[Router] Gatekeeper (Hermes) → {model}")
        return model
    
    def chat(self, model, messages):
        try:
            response = requests.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "repeat_penalty": 1.3,
                        "num_predict": 2048,
                    }
                },
                timeout=300
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            print(f"[Ollama] Error: {response.status_code}")
            return None
        except Exception as e:
            print(f"[Ollama] Connection error: {e}")
            return None
    
    def is_available(self):
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def ensure_model_loaded(self, model_name):
        """Pre-load a model into Ollama if not already loaded."""
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": "5m"},
                timeout=120
            )
            return response.status_code == 200
        except:
            return False


# ============================================
# Command Handler
# ============================================

class CommandHandler:
    def __init__(self, state, router, speaker):
        self.state = state
        self.router = router
        self.speaker = speaker
        # Ensure cards directory exists (shared with Nexus app)
        NEXUS_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    
    def play_sound(self, sound_name):
        if CONFIG["sound_enabled"]:
            sound_path = CONFIG["sounds"].get(sound_name)
            if sound_path and os.path.exists(sound_path):
                subprocess.run(["paplay", sound_path], capture_output=True)
    
    def detect_trigger(self, text):
        text_lower = text.lower().strip()
        
        triggers = {
            # Knowledge capture
            "write this down": "knowledge",
            "write that down": "knowledge",
            "note this": "knowledge",
            "remember this": "knowledge",
            
            # Wake/conversation start - "gogo ollama" and variants
            "gogo ollama": "wake",
            "go go ollama": "wake",
            "go go all llama": "wake",
            "go go olama": "wake",
            "gogo all llama": "wake",
            "coco ollama": "wake",       # Common mishearing
            "cocoa ollama": "wake",
            "go go llama": "wake",
            "hey ollama": "wake",
            "okay ollama": "wake",
            # Legacy wake phrases (still work)
            "go go gadget llama": "wake",
            "gadget llama": "wake",
            
            # Execute
            "run that": "execute",
            "run this": "execute",
            "do that": "execute",
            "do it": "execute",
            "execute": "execute",
            "approved": "execute",
            
            # Cancel
            "nevermind": "cancel",
            "never mind": "cancel",
            "cancel": "cancel",
            "stop": "cancel",
            "abort": "cancel",
            
            # Standby/pause
            "stand by": "standby",
            "standby": "standby",
            "hold on": "standby",
            "one moment": "standby",
            "pause": "standby",
            "wait": "standby",
            "give me a minute": "standby",
            "give me a moment": "standby",
            "brb": "standby",
            "be right back": "standby",
        }
        
        for phrase, action in triggers.items():
            if phrase in text_lower:
                idx = text_lower.find(phrase)
                remaining = text[idx + len(phrase):].strip()
                for filler in [",", ".", ":", "-", "that", "um", "uh", "so"]:
                    remaining = remaining.lstrip(filler).strip()
                return action, remaining
        
        if self.state.in_conversation:
            return "continue", text
        
        return None, text
    
    def handle_knowledge(self, text):
        """Save a voice note as a knowledge card in ~/.nexus/cards/"""
        if not text:
            self.speaker.speak("I didn't catch what you wanted me to write down.")
            return False
        
        title = text.split('\n')[0][:50]
        if len(title) < len(text.split('\n')[0]):
            title += "..."
        
        card = {
            "id": f"voice_{int(time.time() * 1000)}_{os.urandom(3).hex()}",
            "title": title,
            "content": text,
            "tags": ["voice-note"],
            "active": True,
            "cardType": "team",  # Voice notes default to team-visible
            "assignedModels": [],
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
            "createdBy": "voice-daemon",
            "suggestedBy": "voice-daemon",
            "status": "active",
            "priority": 2,
        }
        
        # Save to shared cards directory
        card_path = NEXUS_CARDS_DIR / f"{card['id']}.json"
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        
        # Also write command file for Nexus app to pick up
        command = {"type": "knowledge", "card": card, "timestamp": time.time()}
        with open(CONFIG["command_file"], "w") as f:
            json.dump(command, f, indent=2)
        
        self.play_sound("success")
        self.speaker.speak(f"Got it. I've noted: {title}")
        print(f"[Knowledge] Saved card: {card['id']} - {title}")
        return True
    
    def handle_wake(self, initial_text=""):
        """Wake up Hermes (gatekeeper). He handles simple stuff or escalates."""
        if self.state.paused:
            self.state.exit_standby()
        
        if not self.router.is_available():
            self.speaker.speak("Ollama isn't running. Start it with: ollama serve")
            self.play_sound("error")
            return False
        
        self.play_sound("listening")
        
        if not initial_text:
            self.speaker.speak("Go ahead, I'm listening.")
            return "await_input"
        
        # Route to the right model based on the request
        model = self.router.select_model(initial_text)
        self.state.start_conversation(model)
        
        model_short = model.split(':')[0]
        if model == CONFIG["models"]["gatekeeper"]:
            self.speaker.speak("Hermes here. What do you need?")
        else:
            self.speaker.speak(f"Routing to {model_short}. Thinking...")
        
        # Build system prompt with knowledge card context
        knowledge_context = self._load_team_knowledge()
        
        system_prompt = f"""You are Hermes, the gatekeeper of the Nexus - a local AI workbench built by TAO.

You are currently running as model: {model}

ENVIRONMENT:
- Voice daemon with Whisper STT and Piper TTS (Obadiah voice)
- Ollama with multiple models: {json.dumps(CONFIG['models'], indent=2)}
- The user can say "run that" to approve commands, "stand by" to pause, "nevermind" to cancel
- You can propose shell commands for the user to approve

VOICE GUIDELINES:
- Keep responses under 100 words - they're spoken aloud
- No markdown, no bullet points, no numbered lists - just natural speech
- Be direct and helpful, not apologetic
- If you need a bigger model for the task, say so clearly

ESCALATION:
- If the task requires deep coding, say "This needs a coding specialist. Say gogo ollama and ask for code help."
- If the task requires complex reasoning, say "Let me hand this to the reasoning team."
- For simple questions, quick lookups, and casual chat - handle it yourself.

{knowledge_context}

The user's name is TAO."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_text}
        ]
        
        response = self.router.chat(model, messages)
        
        if response:
            self.state.conversation_history = messages + [{"role": "assistant", "content": response}]
            self.state.last_plan = response
            self.speaker.speak(response)
            
            command = {"type": "conversation", "model": model, "plan": response, "timestamp": time.time()}
            with open(CONFIG["command_file"], "w") as f:
                json.dump(command, f, indent=2)
            
            print(f"[{model_short}] {response}")
            return True
        else:
            self.speaker.speak("Sorry, I couldn't get a response.")
            self.play_sound("error")
            return False
    
    def _load_team_knowledge(self):
        """Load team and project knowledge cards for system prompt context."""
        try:
            cards_dir = NEXUS_CARDS_DIR
            if not cards_dir.exists():
                return ""
            
            sections = []
            for card_file in sorted(cards_dir.glob("*.json")):
                try:
                    with open(card_file) as f:
                        card = json.load(f)
                    
                    if card.get("status") != "active":
                        continue
                    if card.get("active") == False:
                        continue
                    
                    card_type = card.get("cardType", "team")
                    # Voice daemon sees team and project cards (not identity or personal)
                    if card_type in ("team", "project"):
                        sections.append(f"[{card.get('title', 'Untitled')}]: {card.get('content', '')}")
                except:
                    continue
            
            if sections:
                return "TEAM KNOWLEDGE:\n" + "\n".join(sections)
            return ""
        except:
            return ""
    
    def handle_continue(self, text):
        if not self.state.in_conversation:
            return self.handle_wake(text)
        
        model = self.state.current_model
        self.state.conversation_history.append({"role": "user", "content": text})
        
        response = self.router.chat(model, self.state.conversation_history)
        
        if response:
            self.state.conversation_history.append({"role": "assistant", "content": response})
            self.state.last_plan = response
            self.speaker.speak(response)
            
            command = {"type": "conversation", "model": model, "plan": response, "timestamp": time.time()}
            with open(CONFIG["command_file"], "w") as f:
                json.dump(command, f, indent=2)
            return True
        return False
    
    def handle_execute(self, text):
        self.state.activate_auto_approve()
        
        command = {
            "type": "execute",
            "auto_approve": True,
            "auto_approve_until": self.state.auto_approve_until,
            "plan": self.state.last_plan,
            "timestamp": time.time(),
        }
        with open(CONFIG["command_file"], "w") as f:
            json.dump(command, f, indent=2)
        
        self.play_sound("success")
        self.speaker.speak("Running it now. Auto-approve is on for 90 seconds.")
        return True
    
    def handle_cancel(self):
        self.state.end_conversation()
        self.play_sound("cancel")
        self.speaker.speak("Cancelled.")
        if os.path.exists(CONFIG["command_file"]):
            os.remove(CONFIG["command_file"])
        return True
    
    def handle_standby(self):
        self.state.enter_standby()
        self.play_sound("standby")
        self.speaker.speak("Standing by. Say gogo ollama when you're ready.")
        return True
    
    def handle_dictation(self, text):
        """Type the transcribed text into the focused window via xdotool."""
        if not text:
            return False
        
        print(f"[Dictation] Typing: {text}")
        
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", text],
                check=True
            )
            self.play_sound("dictate_end")
            return True
        except Exception as e:
            print(f"[Dictation] Error: {e}")
            self.play_sound("error")
            return False


# ============================================
# Dictation Trigger Watcher
# ============================================

class DictationWatcher:
    """Watches for mouse button trigger to start dictation mode."""
    
    def __init__(self, daemon):
        self.daemon = daemon
        self.trigger_file = CONFIG["dictate_trigger_file"]
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        print(f"[Dictation] Watching {self.trigger_file} for mouse button trigger")
    
    def stop(self):
        self.running = False
    
    def _watch_loop(self):
        while self.running:
            try:
                if os.path.exists(self.trigger_file):
                    os.remove(self.trigger_file)
                    print("[Dictation] Trigger detected!")
                    self.daemon._dictation_pending = True
                    self.daemon.recorder.interrupt()
            except Exception as e:
                print(f"[Dictation] Watch error: {e}")
            
            time.sleep(0.1)


# ============================================
# Main Daemon
# ============================================

class VoiceDaemon:
    def __init__(self):
        print("=" * 50)
        print("Nexus Voice Daemon v2.0")
        print("=" * 50)
        
        self.state = DaemonState()
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.speaker = Speaker()
        self.router = ModelRouter()
        self.handler = CommandHandler(self.state, self.router, self.speaker)
        self.recording_lock = threading.Lock()
        
        # Start dictation watcher
        self.dictation_watcher = DictationWatcher(self)
        self.dictation_watcher.start()
        
        print("\nTrigger phrases:")
        print("  • 'gogo ollama' → Wake Hermes (gatekeeper)")
        print("  • 'write this down' → Save as knowledge card")
        print("  • 'run that' → Execute the plan (auto-approve 90s)")
        print("  • 'stand by' / 'hold on' → Pause listening")
        print("  • 'nevermind' → Cancel")
        print("\nMouse side button → Dictation mode (type speech)")
        print("=" * 50)
    
    def trigger_dictation(self):
        if not self.recording_lock.acquire(timeout=3.0):
            print("[Dictation] Recording already in progress, skipping")
            return
        
        try:
            self.handler.play_sound("dictate_start")
            print("[Dictation] Recording for dictation...")
            
            audio = self.recorder.record_until_silence(verbose=False)
            if audio is None or len(audio) < 1000:
                print("[Dictation] No audio captured")
                return
            
            text = self.transcriber.transcribe(audio)
            if text:
                self.handler.handle_dictation(text)
        finally:
            self.recording_lock.release()
    
    def process_audio(self):
        if not self.recording_lock.acquire(blocking=False):
            time.sleep(0.1)
            return None
        
        try:
            if self.state.in_conversation and CONFIG["chirp_on_active_listen"]:
                self.handler.play_sound("listening")
            elif not self.state.in_conversation and CONFIG["chirp_on_passive_listen"]:
                self.handler.play_sound("listening")
            
            audio = self.recorder.record_until_silence()
            if audio is None or len(audio) < 1000:
                return None
            text = self.transcriber.transcribe(audio)
            if not text:
                return None
            print(f"\n[Heard] \"{text}\"")
            return text
        finally:
            self.recording_lock.release()
    
    def run(self):
        print("\n[Daemon] Listening... (Ctrl+C to stop)")
        
        # Ensure cards directory exists on startup
        NEXUS_CARDS_DIR.mkdir(parents=True, exist_ok=True)
        
        while True:
            try:
                if getattr(self, "_dictation_pending", False):
                    self._dictation_pending = False
                    self.trigger_dictation()
                    continue
                    
                # Standby mode: only respond to wake phrase
                if self.state.paused:
                    audio = self.recorder.record_until_silence(verbose=False)
                    if audio is None or len(audio) < 1000:
                        continue
                    text = self.transcriber.transcribe(audio)
                    if not text:
                        continue
                    
                    text_lower = text.lower()
                    wake_phrases = [
                        "gogo ollama", "go go ollama", "go go all llama",
                        "coco ollama", "cocoa ollama", "go go llama",
                        "hey ollama", "okay ollama",
                        "go go gadget llama", "gadget llama",
                    ]
                    if any(wake in text_lower for wake in wake_phrases):
                        print(f"\n[Heard] \"{text}\"")
                        # Extract text after the wake phrase
                        remaining = text
                        for wake in wake_phrases:
                            if wake in text_lower:
                                idx = text_lower.find(wake)
                                remaining = text[idx + len(wake):].strip()
                                break
                        self.handler.handle_wake(remaining)
                    continue
                
                text = self.process_audio()
                if not text:
                    continue
                
                action, content = self.handler.detect_trigger(text)
                
                if action == "knowledge":
                    if not content:
                        self.speaker.speak("What should I write down?")
                        content = self.process_audio()
                    self.handler.handle_knowledge(content)
                    
                elif action == "wake":
                    result = self.handler.handle_wake(content)
                    if result == "await_input":
                        content = self.process_audio()
                        if content:
                            self.handler.handle_wake(content)
                    
                elif action == "continue":
                    self.handler.handle_continue(content)
                    
                elif action == "execute":
                    self.handler.handle_execute(content)
                    
                elif action == "cancel":
                    self.handler.handle_cancel()
                
                elif action == "standby":
                    self.handler.handle_standby()
                    
                elif self.state.in_conversation:
                    self.handler.handle_continue(text)
                
            except KeyboardInterrupt:
                print("\n[Daemon] Shutting down...")
                self.dictation_watcher.stop()
                break
            except Exception as e:
                print(f"[Error] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nexus Voice Daemon v2.0")
    parser.add_argument("--test-tts", type=str, help="Test TTS with a phrase")
    parser.add_argument("--test-ollama", action="store_true", help="Test Ollama connection")
    parser.add_argument("--test-dictation", action="store_true", help="Test dictation mode")
    parser.add_argument("--test-knowledge", type=str, help="Test creating a knowledge card")
    args = parser.parse_args()
    
    if args.test_tts:
        speaker = Speaker()
        speaker.speak(args.test_tts)
    elif args.test_ollama:
        router = ModelRouter()
        if router.is_available():
            print("✓ Ollama running")
            r = requests.get(f"{CONFIG['ollama_endpoint']}/api/tags")
            print(f"✓ Models: {[m['name'] for m in r.json().get('models', [])]}")
        else:
            print("✗ Ollama not running")
    elif args.test_dictation:
        print("Testing dictation mode...")
        print("Speak something, and it will be typed into the focused window.")
        daemon = VoiceDaemon()
        daemon.trigger_dictation()
    elif args.test_knowledge:
        NEXUS_CARDS_DIR.mkdir(parents=True, exist_ok=True)
        card_id = f"voice_{int(time.time() * 1000)}_{os.urandom(3).hex()}"
        card = {
            "id": card_id,
            "title": args.test_knowledge[:50],
            "content": args.test_knowledge,
            "tags": ["voice-note", "test"],
            "active": True,
            "cardType": "team",
            "assignedModels": [],
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
            "createdBy": "voice-daemon",
            "status": "active",
            "priority": 2,
        }
        card_path = NEXUS_CARDS_DIR / f"{card_id}.json"
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        print(f"✓ Created card: {card_path}")
    else:
        VoiceDaemon().run()
