"""Text-to-speech output.

English: offline, through a persistent pyttsx3 worker process (fast, private).
         If pyttsx3 is unavailable, Windows' built-in System.Speech is used.
Hindi:   gTTS (Google Text-to-Speech, needs internet), played locally.
TTS is a conversion tool, not AI reasoning - it is the agent's actuator.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from ..config import CACHE_DIR, PROJECT_ROOT

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class SpeechResult:
    ok: bool
    backend: str
    message: str = ""
    request_id: str | None = None
    requested_at: float = 0.0     # wall-clock time the request was sent


class Speaker:
    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True):
        self.rate = rate
        self.volume = volume
        self.enabled = enabled
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._events: dict[str, dict] = {}
        self._cond = threading.Condition()
        self._ids = itertools.count(1)
        self._worker_failed = False

    # -- worker process ------------------------------------------------------
    def _start_worker(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True
        if self._worker_failed:
            return False
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "gesturebridge.speech.tts_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", cwd=str(PROJECT_ROOT), creationflags=_NO_WINDOW,
            )
        except OSError:
            self._worker_failed = True
            return False
        threading.Thread(target=self._read_events, args=(self._proc,), daemon=True).start()
        return True

    def _read_events(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with self._cond:
                if event.get("event") in ("started", "done") and event.get("id"):
                    self._events.setdefault(event["id"], {})[event["event"]] = event["t"]
                elif event.get("event") == "ready":
                    self._events["__ready__"] = {"ready": event["t"]}
                self._cond.notify_all()
        if proc.poll() not in (None, 0):
            self._worker_failed = True  # e.g. pyttsx3 missing -> use the fallback

    def warm_up(self) -> None:
        """Start the worker and prime the speech engine with a silent utterance."""
        with self._lock:
            if self._start_worker():
                try:
                    self._proc.stdin.write(json.dumps({"id": "warmup", "text": "ready", "volume": 0.0}) + "\n")
                    self._proc.stdin.flush()
                except (OSError, ValueError):
                    self._proc = None

    def wait_for(self, request_id: str, event: str = "started", timeout: float = 5.0) -> float | None:
        """Wall-clock time at which the worker started/finished speaking `request_id`."""
        deadline = time.time() + timeout
        with self._cond:
            while event not in self._events.get(request_id, {}):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._events[request_id][event]

    # -- public API ------------------------------------------------------------
    def speak(self, text: str, lang: str = "en") -> SpeechResult:
        if not self.enabled:
            return SpeechResult(False, "disabled", "speech output is switched off")
        text = (text or "").strip()
        if not text:
            return SpeechResult(False, "none", "nothing to say")
        if lang != "en":
            return self._speak_gtts(text, lang)

        with self._lock:
            if self._start_worker():
                request_id = f"u{next(self._ids)}"
                try:
                    self._proc.stdin.write(json.dumps({"id": request_id, "text": text, "rate": self.rate,
                                                       "volume": self.volume}) + "\n")
                    self._proc.stdin.flush()
                    return SpeechResult(True, "pyttsx3", request_id=request_id, requested_at=time.time())
                except (OSError, ValueError):
                    self._proc = None
        return self._speak_system(text)

    def synthesize(self, text: str, lang: str = "en") -> tuple[bytes | None, str]:
        """Speech as audio bytes for the browser to play: (data, mime type) or (None, error)."""
        text = (text or "").strip()
        if not text:
            return None, "nothing to say"
        if lang != "en":
            try:
                from gtts import gTTS
                buffer = io.BytesIO()
                gTTS(text=text, lang=lang).write_to_fp(buffer)
                return buffer.getvalue(), "audio/mpeg"
            except ImportError:
                return None, "gTTS is not installed (pip install gTTS)"
            except Exception as exc:  # gTTS raises its own error types for network problems
                return None, f"Hindi speech needs internet: {type(exc).__name__}"
        audio_dir = CACHE_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = audio_dir / f"speech_{uuid.uuid4().hex[:8]}.wav"
        request_id = f"f{next(self._ids)}"
        with self._lock:
            if not self._start_worker():
                return None, "speech engine (pyttsx3) is not available"
            try:
                self._proc.stdin.write(json.dumps({"id": request_id, "text": text, "rate": self.rate,
                                                   "volume": self.volume, "file": str(path)}) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):
                self._proc = None
                return None, "speech engine stopped - try again"
        self.wait_for(request_id, "done", timeout=15)
        try:
            data = path.read_bytes()
            path.unlink()
        except OSError:
            return None, "speech engine did not produce audio"
        return (data, "audio/wav") if len(data) > 1000 else (None, "speech engine produced no audio")

    def stop(self) -> None:
        """Interrupt current speech (the worker restarts on the next request)."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=2)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self._proc.kill()
            self._proc = None

    # -- fallbacks and other languages -------------------------------------------
    def _speak_system(self, text: str) -> SpeechResult:
        if sys.platform != "win32":
            for player in ("say", "espeak"):
                if shutil.which(player):
                    subprocess.Popen([player, text])
                    return SpeechResult(True, player, requested_at=time.time())
            return SpeechResult(False, "none", "no text-to-speech engine found (pip install pyttsx3)")
        sapi_rate = max(-10, min(10, round((self.rate - 165) / 15)))
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Rate = {sapi_rate}; $s.Volume = {int(self.volume * 100)}; "
                  f"$s.Speak('{text.replace(chr(39), chr(39) * 2)}')")
        subprocess.Popen(["powershell", "-NoProfile", "-Command", script], creationflags=_NO_WINDOW)
        return SpeechResult(True, "windows-sapi", requested_at=time.time())

    def _speak_gtts(self, text: str, lang: str) -> SpeechResult:
        try:
            from gtts import gTTS
        except ImportError:
            return SpeechResult(False, "gtts", "gTTS is not installed (pip install gTTS)")
        try:
            buffer = io.BytesIO()
            gTTS(text=text, lang=lang).write_to_fp(buffer)
        except Exception as exc:  # gTTS raises its own error types for network problems
            return SpeechResult(False, "gtts", f"Hindi speech needs internet: {type(exc).__name__}")
        audio_dir = CACHE_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = str(audio_dir / f"gesturebridge_{uuid.uuid4().hex[:8]}.mp3")
        with open(path, "wb") as f:
            f.write(buffer.getvalue())
        threading.Thread(target=play_mp3, args=(path,), daemon=True).start()
        return SpeechResult(True, "gtts", requested_at=time.time())


def play_mp3(path: str) -> None:
    try:
        if sys.platform == "win32":
            _play_mp3_windows(path)
        elif shutil.which("afplay"):
            subprocess.run(["afplay", path], check=False)
        elif shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=False)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _play_mp3_windows(path: str) -> None:
    """Play an MP3 with the Windows Media Control Interface (no extra packages)."""
    import ctypes

    winmm = ctypes.windll.winmm
    alias = f"gb{uuid.uuid4().hex[:8]}"

    def mci(command: str) -> None:
        error = winmm.mciSendStringW(command, None, 0, None)
        if error:
            raise RuntimeError(f"MCI error {error}: {command}")

    mci(f'open "{path}" type mpegvideo alias {alias}')
    try:
        mci(f"play {alias} wait")
    finally:
        mci(f"close {alias}")
