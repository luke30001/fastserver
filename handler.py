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

# Set cache dirs - use /app/models which is baked into the image
# (RunPod may mount volumes at /cache, shadowing baked-in files)
os.environ.setdefault("HF_HOME", "/app/models")
os.environ.setdefault("XDG_CACHE_HOME", "/app/models")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/models")

_model: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    global _model
    if _model is None:
        download_root = os.environ.get("HF_HOME", "/app/models")
        require_cached = os.environ.get("WHISPER_REQUIRE_CACHED", "1") == "1"
        try:
            _model = WhisperModel(
                DEFAULT_MODEL_SIZE,
                device=DEFAULT_DEVICE,
                compute_type=DEFAULT_COMPUTE_TYPE,
                download_root=download_root,
                local_files_only=require_cached,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            # Don't mask CUDA/GPU errors as cache errors
            if "cuda" in exc_str or "gpu" in exc_str or "device" in exc_str:
                raise
            if require_cached and "local" in exc_str:
                raise RuntimeError(
                    f"Model must be cached at {download_root}; "
                    f"set WHISPER_REQUIRE_CACHED=0 to allow downloading."
                ) from exc
            raise
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
    Accepts multiple input formats:
    - event["files"]["file"]["content"]: base64 payload (RunPod UI multipart)
    - event["input"]["file"]: base64 string
    - event["input"]["file_url"] or event["input"]["audio_url"]: http(s) URL to fetch
    - event["file"]: base64 string (direct)
    - event["file_url"] or event["audio_url"]: http(s) URL (direct)
    """
    # Try RunPod UI multipart format first
    files = event.get("files") or {}
    if isinstance(files, dict):
        file_entry = files.get("file")
        if isinstance(file_entry, dict):
            file_b64 = file_entry.get("content")
            if file_b64:
                return _write_temp_audio_from_base64(file_b64)

    # Try input payload
    input_payload = event.get("input") or {}
    if isinstance(input_payload, dict):
        file_b64 = input_payload.get("file")
        if file_b64:
            return _write_temp_audio_from_base64(file_b64)
        
        # Accept both "file_url" and "audio_url" for flexibility
        file_url = input_payload.get("file_url") or input_payload.get("audio_url")
        if file_url:
            return _write_temp_audio_from_url(file_url)

    # Try direct event fields
    if "file" in event:
        file_b64 = event.get("file")
        if file_b64:
            return _write_temp_audio_from_base64(file_b64)
    
    # Accept both "file_url" and "audio_url" for flexibility
    file_url = event.get("file_url") or event.get("audio_url")
    if file_url:
        return _write_temp_audio_from_url(file_url)

    # Provide helpful error message with what was actually received
    available_keys = list(event.keys())
    raise ValueError(
        f"No audio provided. Send 'file' (base64) or 'file_url'. "
        f"Received event keys: {available_keys}. "
        f"Event structure: {str(event)[:500]}"
    )


def run(event: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = _extract_audio_to_file(event)
    model = load_model()

    # Allow per-request overrides
    input_payload = event.get("input") or {}
    language = input_payload.get("language", DEFAULT_LANGUAGE) or None
    beam_size = int(input_payload.get("beam_size", DEFAULT_BEAM_SIZE))
    condition_on_previous_text = input_payload.get("condition_on_previous_text", False)

    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language,
        condition_on_previous_text=condition_on_previous_text,
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
