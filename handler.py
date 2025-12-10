import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import runpod
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("runpod-fasterwhisper")

MODEL_SIZE = os.getenv("MODEL_SIZE", "medium")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
VAD_FILTER = os.getenv("VAD_FILTER", "true").lower() == "true"


def load_model() -> WhisperModel:
    """
    Load the Whisper model once at cold start.
    """

    device = "cuda"
    compute_type = COMPUTE_TYPE

    logger.info("Loading Whisper model %s (device=%s, compute_type=%s)", MODEL_SIZE, device, compute_type)

    try:
        model = WhisperModel(
            MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )
    except ValueError as exc:
        # Fallback if the requested compute type is not supported by the backend.
        if "float16" in str(exc).lower():
            fallback_type = "int8_float16"
            logger.warning(
                "Requested compute_type=%s not supported on device=%s; retrying with %s",
                compute_type,
                device,
                fallback_type,
            )
            model = WhisperModel(
                MODEL_SIZE,
                device=device,
                compute_type=fallback_type,
            )
        else:
            raise

    return model


model = load_model()


def _download_to_temp(url: str, suffix: str = ".mp3") -> str:
    """
    Download a remote audio file to a temporary path.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return tmp.name


def _decode_base64_audio(b64_audio: str, suffix: str = ".wav") -> str:
    """
    Decode base64 audio into a temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(base64.b64decode(b64_audio))
        return tmp.name


def _resolve_audio_source(event_input: Dict[str, Any]) -> str:
    """
    Accept either audio_url or audio_base64 and return a temp file path.
    """
    if "audio_url" in event_input:
        return _download_to_temp(event_input["audio_url"])

    if "audio_base64" in event_input:
        return _decode_base64_audio(event_input["audio_base64"])

    raise ValueError("Provide either `audio_url` or `audio_base64` in input.")


def _transcribe(audio_path: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run transcription and return structured results.
    """
    task = "translate" if opts.get("translate", False) else "transcribe"
    language = opts.get("language")
    beam_size = opts.get("beam_size", 5)
    vad_filter = opts.get("vad_filter", VAD_FILTER)
    chunk_length = opts.get("chunk_length_s", 30)

    segments_out = []

    logger.info(
        "Starting %s (language=%s, beam_size=%s, vad_filter=%s)",
        task,
        language,
        beam_size,
        vad_filter,
    )

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        task=task,
        vad_filter=vad_filter,
        chunk_length=chunk_length,
        log_prob_threshold=opts.get("log_prob_threshold"),
        no_speech_threshold=opts.get("no_speech_threshold", 0.6),
    )

    for seg in segments:
        segments_out.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "temperature": seg.temperature,
            }
        )

    return {
        "task": task,
        "language": info.language,
        "duration": info.duration,
        "transcription": segments_out,
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless entrypoint.
    """
    audio_path: Optional[str] = None
    try:
        event_input = event.get("input", {})
        audio_path = _resolve_audio_source(event_input)

        result = _transcribe(audio_path, event_input)
        return {"status": "ok", "result": result}
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Transcription failed")
        return {"status": "error", "message": str(exc)}
    finally:
        # Clean up temp file, if any
        if audio_path and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
