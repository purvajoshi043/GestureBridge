"""Text-to-speech worker process (pyttsx3 -> Windows SAPI5 / macOS / eSpeak).

Started by ``Speaker`` as ``python -m gesturebridge.speech.tts_worker``. It reads
one JSON request per line on stdin and speaks it, and reports "started"/"done"
events on stdout. Running the engine in its own process keeps it away from the
UI thread (pyttsx3 is not thread-safe and blocks while speaking).
"""

from __future__ import annotations

import json
import sys
import time


def _emit(event: str, **data) -> None:
    print(json.dumps({"event": event, "t": time.time(), **data}), flush=True)


def _save_wav(engine, request: dict) -> None:
    """On Windows, SAPI writes the file directly (pyttsx3's save_to_file can hang there)."""
    text, path = request.get("text", ""), request["file"]
    if sys.platform == "win32":
        try:
            import win32com.client

            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(path, 3)                                  # SSFMCreateForWrite
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.AudioOutputStream = stream
            voice.Rate = max(-10, min(10, round((int(request.get("rate", 165)) - 165) / 15)))
            voice.Volume = int(float(request.get("volume", 1.0)) * 100)
            voice.Speak(text)
            stream.Close()
            return
        except Exception:  # fall back to pyttsx3 below
            pass
    engine.save_to_file(text, path)
    engine.runAndWait()


def main() -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.connect("started-utterance", lambda name: _emit("started", id=name))
    _emit("ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if request.get("cmd") == "quit":
            break
        engine.setProperty("rate", int(request.get("rate", 165)))
        engine.setProperty("volume", float(request.get("volume", 1.0)))
        if request.get("file"):          # write a WAV file (played in the browser) instead of the speakers
            _save_wav(engine, request)
        else:
            engine.say(request.get("text", ""), request.get("id"))
            engine.runAndWait()
        _emit("done", id=request.get("id"))


if __name__ == "__main__":
    main()
