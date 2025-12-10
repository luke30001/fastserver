# RunPod serverless Whisper (turbo, GPU)

GPU-accelerated Faster-Whisper endpoint for RunPod Serverless, using the `turbo`
model by default and caching it under `/app/models`.

## Request / Response

- Endpoint handler: `handler.run` (module `serverless/handler.py`).
- Input (preferred): multipart upload in RunPod UI — select `file` as the field.
  RunPod passes this as `event.files.file.content` (base64).
- Alternate input:
  - `event.input.file` (base64 string)
  - `event.input.file_url` (HTTP/HTTPS URL to fetch)
  - `event.input.audio_url` (alias of `file_url`)
- Optional overrides:
  - `event.input.language` (e.g. `"en"`)
  - `event.input.beam_size` (int)
  - `event.input.vad_filter` (bool, default `true`)
- Response JSON:
  - `language`: detected or forced language code
  - `language_probability`: float
  - `segments`: array of `{ start, end, text }`

### Sample rp_tester payload (base64 file)

```json
{
  "files": { "file": { "content": "<base64-audio>" } },
  "input": {
    "language": "en",
    "beam_size": 5,
    "vad_filter": true
  }
}
```

### Sample rp_tester payload (file_url)

```json
{
  "input": {
    "file_url": "https://example.com/audio.mp3",
    "language": "en",
    "beam_size": 5,
    "vad_filter": true
  }
}
```

## Local testing with rp_tester

```bash
pip install runpod
python -m runpod.serverless.utils.rp_tester --handler handler --event ./sample-event.json
```

## Environment knobs

- `WHISPER_MODEL_SIZE`   (default `turbo`)
- `WHISPER_COMPUTE_TYPE` (default `float16` for GPU, e.g. `int8` for CPU fallback)
- `WHISPER_DEVICE`       (default `cuda`, set to `cpu` to force CPU)
- `WHISPER_BEAM_SIZE`    (default `5`)
- `WHISPER_LANGUAGE`     (default empty -> auto-detect)
- `WHISPER_DOWNLOAD_ROOT` (default `/app/models`)

Cache directories (build + runtime):

- `HF_HOME=/app/models`
- `XDG_CACHE_HOME=/app/models`
- `HUGGINGFACE_HUB_CACHE=/app/models`

## RunPod template notes

- Entrypoint: `python -u serverless/handler.py`
- The Dockerfile pre-downloads the `turbo` model into `/app/models` at build time
  using CPU so that cold starts on RunPod are faster.
- At runtime the handler loads the same model on CUDA (`device=cuda`,
  `compute_type=float16`) if a GPU is available.
- If you want CPU-only, set:
  - `WHISPER_DEVICE=cpu`
  - `WHISPER_COMPUTE_TYPE=int8`
  and optionally lower `WHISPER_MODEL_SIZE` to `medium` or `small`.
