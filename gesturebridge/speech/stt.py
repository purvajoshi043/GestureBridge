"""Reverse mode: the hearing person speaks, the signer reads the text.

Uses the SpeechRecognition package with Google's free web speech API, so the
recorded audio is sent to Google (internet needed) - the UI tells the user.
"""

from __future__ import annotations

import io


def transcribe(wav_bytes: bytes, language: str = "en-IN") -> tuple[str | None, str | None]:
    """Returns (text, error)."""
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "SpeechRecognition is not installed (pip install SpeechRecognition)"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)
    except (ValueError, EOFError) as exc:
        return None, f"Could not read the recording: {exc}"
    try:
        return recognizer.recognize_google(audio, language=language), None
    except sr.UnknownValueError:
        return None, "Could not understand the speech - please try again."
    except sr.RequestError as exc:
        return None, f"Speech service unreachable (internet needed): {exc}"
