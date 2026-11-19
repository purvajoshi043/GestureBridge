"""GestureBridge: AI-powered sign language to speech translation.

Pipeline:
    camera -> hand landmarks (MediaPipe) -> features -> classifier -> stabiliser
    -> word buffer -> beam-search decoder -> rule engine (forward chaining)
    -> LLM refinement (local, guarded) -> text-to-speech
"""

import os
from pathlib import Path

__version__ = "1.0.0"

# Keep library caches and temporary files inside the project folder (D: drive)
# instead of the user profile on C:.
_CACHE = Path(__file__).resolve().parent.parent / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_CACHE / "joblib"))
