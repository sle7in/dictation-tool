# Dictation Tool

Local push-to-transcribe dictation for Windows. All speech recognition runs on-device using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no audio leaves your machine.

## How it works

Double-tap left ctrl to start recording. Double-tap again to stop — the audio is transcribed locally via Whisper and the text is typed at your cursor. A beep confirms each action.

Key-repeat and accidental holds are filtered out so only intentional double-taps register.

## Requirements

- Windows
- NVIDIA GPU with CUDA support
- Python 3.10+

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python dictation.py
```

- **Double-tap left ctrl** — toggle recording on/off
- **Ctrl+Q** — quit

## Privacy

This is a fully closed-loop system. Audio is captured, transcribed, and discarded entirely on your local machine. Nothing is sent to any external service.
