import os
import sys
import warnings
warnings.filterwarnings("ignore")

site_packages = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
nvidia_dirs = []
for lib_dir in ['nvidia/cublas/bin', 'nvidia/cudnn/bin', 'nvidia/cuda_nvrtc/bin']:
    path = os.path.normpath(os.path.join(site_packages, lib_dir))
    if os.path.isdir(path):
        nvidia_dirs.append(path)
        os.add_dll_directory(path)
os.environ['PATH'] = ';'.join(nvidia_dirs) + ';' + os.environ.get('PATH', '')

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import time
import threading
import winsound
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DOUBLE_TAP_WINDOW = 0.5

model = WhisperModel("turbo", device="cuda", compute_type="float16")

recording = False
audio_chunks = []
stream = None
last_ctrl_press = 0
ctrl_is_down = False
cooldown_until = 0
first_entry = True


def beep_start():
    winsound.Beep(1000, 150)

def beep_stop():
    winsound.Beep(600, 150)

def beep_done():
    winsound.Beep(1200, 100)
    time.sleep(0.05)
    winsound.Beep(1200, 100)


def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())


def start_recording():
    global recording, audio_chunks, stream
    audio_chunks = []
    recording = True
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
    beep_start()


def stop_and_transcribe():
    global recording, stream, first_entry
    recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None

    beep_stop()

    if not audio_chunks:
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    audio_i16 = (audio * 32767).astype(np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32767.0

    segments, _ = model.transcribe(
        audio_f32,
        beam_size=3,
        language="en",
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=200),
    )
    segments = list(segments)

    if not segments:
        return

    text = " ".join(seg.text.strip() for seg in segments)
    if not first_entry:
        text = " " + text
    first_entry = False

    old_clip = pyperclip.paste()
    pyperclip.copy(text)
    time.sleep(0.1)
    keyboard.send("ctrl+v")
    time.sleep(0.5)
    pyperclip.copy(old_clip)

    with open(os.path.join(os.path.dirname(__file__), "dictation.log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] transcribed: {text.strip()}\n")

    beep_done()


def on_ctrl_event(event):
    global last_ctrl_press, ctrl_is_down, cooldown_until, recording

    if event.scan_code != 29:
        return

    if event.event_type == "up":
        ctrl_is_down = False
        return

    if event.event_type != "down":
        return

    if ctrl_is_down:
        return
    ctrl_is_down = True

    now = time.time()

    if now < cooldown_until:
        last_ctrl_press = now
        return

    gap = now - last_ctrl_press
    last_ctrl_press = now

    if gap < DOUBLE_TAP_WINDOW:
        cooldown_until = now + 1.0
        if not recording:
            start_recording()
        else:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()


keyboard.hook(on_ctrl_event)
keyboard.wait("ctrl+q")
if recording and stream:
    stream.stop()
    stream.close()
