# Nexus Voice Daemon

A standalone, system-wide, always-listening voice command and dictation engine designed for Linux. It utilizes extremely fast `faster-whisper` STT and CTranslate2 for ultra-low latency transcription.

## Features
- **Always Listening:** Listens passively for wake words via lightweight WebRTC VAD.
- **Microphone Lock Interrupts:** Seamlessly transitions between background listening and active dictation modes without lock collisions.
- **Hardware Triggered Dictation:** Bind dictation directly to any physical mouse button using `xbindkeys`.

## Requirements
```bash
pip install -r requirements.txt --break-system-packages
```
You will also need `xdotool` and `xbindkeys` if using hardware triggers:
```bash
sudo apt install xbindkeys xdotool
```

## Mouse Integration (xbindkeys)
Because Linux's Primary Selection overlaps with clipboard managers like Parcellite, the most robust way to trigger Voice Dictation vs Text-to-Speech is to bind them to physical inputs with explicit modifiers. 

Here is an example `~/.xbindkeysrc` layout to map these precisely to your Mouse Forward Button:

```bash
# Start Dictation (Forward Button)
"touch /tmp/nexus_dictate_trigger"
  b:9

# Speak Highlighted Text (Shift + Forward Button)
"bash /home/user/scripts/speakit/speakit.sh"
  shift + b:9

# Stop Speaking / Cancel Dictation (Back Button)
"touch /tmp/nexus_cancel_trigger ; killall -9 piper"
  b:8
```

## Systemd Installation
You can configure the daemon to load entirely seamlessly in the background on startup:
```bash
mkdir -p ~/.local/bin
cp nexus-voice-daemon.py ~/.local/bin/
chmod +x ~/.local/bin/nexus-voice-daemon.py

mkdir -p ~/.config/systemd/user/
cp nexus-voice-daemon.service ~/.config/systemd/user/
systemctl --user enable nexus-voice-daemon
systemctl --user start nexus-voice-daemon
```
