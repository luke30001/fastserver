import base64
import os
import tempfile
from typing import Any, Dict

import requests
import runpod
from faster_whisper import WhisperModel

# Defaults tuned for GPU RunPod serverless with preinstalled Turbo model
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "turbo")
DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEFAULT_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
DEFAULT_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "")  # empty -> auto-detect

MODEL_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/app/models")


def load_model() -> WhisperModel:
    """Load a single global WhisperModel instance.

    This keeps cold-start cost low by:
    - Reusing the same model across all invocations within a worker.
    - Pointing download_root at /app/models, which we also pre-populate in the Dockerfile.
    """
    global _MODEL
    try:
        return _MODEL  # type: ignore[name-defined]
    except NameError:
        pass

    print(
        f"Loading Whisper model: size={DEFAULT_MODEL_SIZE}, "
        f"device={DEFAULT_DEVICE}, compute_type={DEFAULT_COMPUTE_TYPE}, "
        f"download_root={MODEL_DOWNLOAD_ROOT}"
    )

    model = WhisperModel(
        DEFAULT_MODEL_SIZE,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE_TYPE,
        download_root=MODEL_DOWNLOAD_ROOT,
    )

    globals()["_MODEL"] = model
    return model


def _write_temp_audio_from_base64(b64: str) -> str:
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _write_temp_audio_from_url(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def _get_audio_file_path(event: Dict[str, Any]) -> str:
    """Support multiple input formats:

    1. RunPod UI upload:
       event.files.file.content  (base64)
    2. JSON input:
       event.input.file          (base64)
       event.input.file_url      (HTTP/HTTPS URL)
       event.input.audio_url     (alias)
    3. Legacy top-level:
       event.file                (base64)
    """
    # 1) RunPod "files" field (UI upload)
    files = event.get("files") or {}
    file_field = files.get("file") if isinstance(files, dict) else None
    if isinstance(file_field, dict) and "content" in file_field:
        return _write_temp_audio_from_base64(file_field["content"])

    # 2) event.input.*
    input_payload = event.get("input") or {}
    if "file" in input_payload:
        return _write_temp_audio_from_base64(input_payload["file"])

    file_url = (
        input_payload.get("file_url")
        or input_payload.get("audio_url")
        or event.get("file_url")
        or event.get("audio_url")
    )
    if file_url:
        return _write_temp_audio_from_url(file_url)

    # 3) legacy top-level base64
    if "file" in event:
        return _write_temp_audio_from_base64(event["file"])

    raise ValueError("No audio file provided (file/file_url/audio_url missing)")


def run(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler.

    Expects an `event` like:
    {
      "input": {
        "file_url": "https://...",
        "language": "en",
        "beam_size": 5,
        "vad_filter": true
      }
    }
    """
    model = load_model()

    input_payload = event.get("input") or {}

    language = input_payload.get("language", DEFAULT_LANGUAGE) or None
    beam_size = int(input_payload.get("beam_size", DEFAULT_BEAM_SIZE))
    vad_filter = bool(input_payload.get("vad_filter", True))

    audio_path = _get_audio_file_path(event)
    print(f"Processing audio file: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
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
