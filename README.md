# RunPod serverless Whisper (turbo)

## Request / Response
- Endpoint handler: `handler.run` (module `serverless/handler.py`).
- Input (preferred): multipart upload in RunPod UI — select `file` field. RunPod passes this as `event.files.file.content` (base64).
- Alternate input: `event.input.file` (base64 string) or `event.input.file_url` (HTTP/HTTPS URL to fetch).
- Optional overrides: `event.input.language` (e.g. `"en"`), `event.input.beam_size` (int).
- Response JSON:
  - `language`: detected or forced language code
  - `language_probability`: float
  - `segments`: array of `{ start, end, text }`

### Sample rp_tester payload (base64 file)
```json
{
  "files": { "file": { "content": "<base64-audio>" } },
  "input": { "beam_size": 5 }
}
```

### Sample payload using a URL
```json
{
  "input": {
    "file_url": "https://example.com/audio.m4a",
    "language": "en"
  }
}
```

## Build & run locally
```bash
docker build -t whisper-runpod .
docker run -p 8080:8080 whisper-runpod
```
Then test with `rp_tester`:
```bash
python -m runpod.serverless.utils.rp_tester --handler handler --event ./sample-event.json
```

## Environment knobs
- `WHISPER_MODEL_SIZE` (default `turbo`)
- `WHISPER_COMPUTE_TYPE` (default `float16`)
- `WHISPER_DEVICE` (default `cuda`)
- `WHISPER_BEAM_SIZE` (default `5`)
- `WHISPER_LANGUAGE` (default empty -> auto-detect)
- `WHISPER_REQUIRE_CACHED` (default `1` - set to `0` to allow model download)
- Cache directories: `HF_HOME=/app/models`, `XDG_CACHE_HOME=/app/models` (model is pre-baked in Docker image).

## RunPod template notes
- Entrypoint: `python -u handler.py`
- GPU required (CUDA); the turbo model is pre-downloaded in the Docker image for fast cold starts.
- Use `WHISPER_MODEL_SIZE=medium` or `small` if you need smaller models.
