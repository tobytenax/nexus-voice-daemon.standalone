#!/bin/bash
# smart_voice.sh — Context-aware voice.
# Priority order:
#   1. PRIMARY has ≥ 3 words → TTS (regardless of window)
#   2. Cursor is in a text-input window → push-to-talk dictation
#   3. Else → "nothing to do"

PIPER=/home/forge-of-change/.local/bin/piper
ONNX=/home/forge-of-change/piper_voices/en_GB-semaine-medium.onnx
TMPWAV=/tmp/smart_voice_speak.wav
TMPRAW=/tmp/smart_voice_ptt.wav
PIDFILE=/tmp/smart_voice_ptt.pid
SOUND_START=/usr/share/sounds/sound-icons/pisk-up.wav
SOUND_STOP=/usr/share/sounds/sound-icons/pisk-down.wav

# ── 1. Check for intentional text selection (≥3 words) ───────────────────────
SELECTED_TEXT=$(xclip -o -selection primary 2>/dev/null)
WORD_COUNT=$(echo "$SELECTED_TEXT" | wc -w)

if [ "$WORD_COUNT" -ge 3 ]; then
    # ── FRESH SELECTION → TTS with Piper/Obadiah ─────────────────────────────
    # Cancel any stuck recording first
    if [ -f "$PIDFILE" ]; then
        kill "$(cat $PIDFILE)" 2>/dev/null
        rm -f "$PIDFILE"
    fi

    notify-send "Voice Mode" "Reading text aloud..." -t 2000

    aplay -q -d 2 -f S16_LE -r 44100 /dev/zero &>/dev/null &
    sleep 0.8

    CLEAN=$(echo "$SELECTED_TEXT" | \
        sed -E 's|https?://[^ \t\n]+||g; s/[*_~#`]//g' | \
        tr -cd '\11\12\15\40-\176')

    echo "$CLEAN" | "$PIPER" \
        --model "$ONNX" \
        --speaker 2 \
        --length_scale 0.5 \
        --noise_scale 0.33 \
        --output_file "$TMPWAV"

    aplay -q "$TMPWAV"

    # Clear PRIMARY so next button press doesn't re-read stale text
    echo -n | xclip -selection primary 2>/dev/null
    exit 0
fi

# ── 2. No selection → check if we're in a text-input window ──────────────────
WIN_CLASS=$(xdotool getwindowfocus 2>/dev/null | \
    xargs -I{} xprop -id {} WM_CLASS 2>/dev/null | \
    tr '[:upper:]' '[:lower:]')

TEXT_INPUT_PATTERN="antigravity|terminal|alacritty|xterm|konsole|tilix|gnome-term|code|codium|sublime|atom|gedit|mousepad|kate|kwrite|pluma|xed|leafpad|emacs|vim|idea|pycharm|webstorm|jetbrains|libreoffice|writer"

if echo "$WIN_CLASS" | grep -qE "$TEXT_INPUT_PATTERN"; then
    # ── PUSH-TO-TALK DICTATION ────────────────────────────────────────────────
    if [ -f "$PIDFILE" ]; then
        # Second press: STOP
        ARECORD_PID=$(cat "$PIDFILE")
        rm -f "$PIDFILE"
        kill "$ARECORD_PID" 2>/dev/null
        wait "$ARECORD_PID" 2>/dev/null
        sleep 0.3   # let arecord fully flush WAV header to disk

        aplay -q "$SOUND_STOP"
        notify-send "Voice Mode" "Transcribing..." -t 5000

        # Ensure worker is running
        if [ ! -f /tmp/whisper_worker_ready ]; then
            notify-send "Voice Mode" "Starting Whisper worker..." -t 3000
            python3 /home/forge-of-change/.local/bin/whisper_worker.py &
            # Wait up to 90s for it to be ready
            for i in $(seq 1 90); do
                [ -f /tmp/whisper_worker_ready ] && break
                sleep 1
            done
        fi

        # Send transcription request
        rm -f /tmp/whisper_result
        echo "$TMPRAW" > /tmp/whisper_request

        # Wait for result (up to 30s)
        for i in $(seq 1 150); do
            [ -f /tmp/whisper_result ] && break
            sleep 0.2
        done

        DICTATED_TEXT=$(cat /tmp/whisper_result 2>/dev/null)
        rm -f /tmp/whisper_result

        if [ -n "$DICTATED_TEXT" ]; then
            # Restore focus to the window that was active at recording-start
            WINFILE=/tmp/smart_voice_target_win
            if [ -f "$WINFILE" ]; then
                TARGET_WIN=$(cat "$WINFILE")
                rm -f "$WINFILE"
                xdotool windowfocus --sync "$TARGET_WIN" 2>/dev/null
                sleep 0.1
            fi
            notify-send "Voice Mode" "✓ $DICTATED_TEXT" -t 4000
            xdotool type --clearmodifiers --delay 2 "$DICTATED_TEXT"
        else
            notify-send "Voice Mode" "Nothing heard." -t 2000
        fi

    else
        # First press: START — record target window ID so typing lands in the right place
        WINFILE=/tmp/smart_voice_target_win
        xdotool getwindowfocus > "$WINFILE" 2>/dev/null
        arecord -q -f S16_LE -r 16000 -c 1 "$TMPRAW" &
        echo $! > "$PIDFILE"
        sleep 0.15
        aplay -q "$SOUND_START"
        notify-send "Voice Mode" "🎙 Recording... press again to stop." -t 60000
    fi

else
    notify-send "Voice Mode" "Highlight text to read, or click in a text box to dictate." -t 3000
fi
