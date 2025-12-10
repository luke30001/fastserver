import base64
import os
import tempfile
from typing import Any, Dict, Optional

import requests
import runpod
from faster_whisper import WhisperModel


# Defaults tuned for GPU RunPod serverless with preinstalled Turbo model
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "turbo")
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEFAULT_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
DEFAULT_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "")  # empty -> auto-detect

# Set cache dirs so model weights persist across warm starts when a volume is attached.
os.environ.setdefault("HF_HOME", "/cache/hf")
os.environ.setdefault("XDG_CACHE_HOME", "/cache")

_model: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            DEFAULT_MODEL_SIZE,
            device=DEFAULT_DEVICE,
            compute_type=DEFAULT_COMPUTE_TYPE,
        )
    return _model


def _write_temp_audio_from_base64(content_b64: str, suffix: str = ".audio") -> str:
    data = base64.b64decode(content_b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _write_temp_audio_from_url(url: str, suffix: str = ".audio") -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def _extract_audio_to_file(event: Dict[str, Any]) -> str:
    """
    Accepts:
    - event["files"]["file"]["content"]: base64 payload (RunPod UI multipart)
    - event["input"]["file"]: same as above
    - event["input"]["file_url"]: http(s) URL to fetch
    """
    files = event.get("files") or {}
    input_payload = event.get("input") or {}

    file_entry = files.get("file") if isinstance(files, dict) else None
    file_b64 = None
    if isinstance(file_entry, dict):
        file_b64 = file_entry.get("content")
    if not file_b64:
        file_b64 = input_payload.get("file")

    if file_b64:
        return _write_temp_audio_from_base64(file_b64)

    file_url = input_payload.get("file_url")
    if file_url:
        return _write_temp_audio_from_url(file_url)

    raise ValueError("No audio provided. Send 'file' (base64) or 'file_url'.")


def run(event: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = _extract_audio_to_file(event)
    model = load_model()

    # Allow per-request overrides
    input_payload = event.get("input") or {}
    language = input_payload.get("language", DEFAULT_LANGUAGE) or None
    beam_size = int(input_payload.get("beam_size", DEFAULT_BEAM_SIZE))

    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language,
    )

    result_segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in segments
    ]

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": result_segments,
    }


runpod.serverless.start({"handler": run})
