"""RunPod Serverless SeedVR2 3B Q4 video upscaling preview.
Only signed Cloudflare R2 object URLs are accepted. No R2 API keys in workers.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

import requests

REVISION = "seedvr2-3b-q4-preview-v1"
CLI = "/opt/seedvr2/inference_cli.py"
MODEL_DIR = "/opt/seedvr2/models/SEEDVR2"
MAX_INPUT_BYTES = 120 * 1024 * 1024
MAX_OUTPUT_BYTES = 300 * 1024 * 1024
MAX_DURATION = 20.0
MAX_SOURCE_EDGE = 1920


def _r2_url(value):
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("Expected a signed Cloudflare R2 URL")
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https"
        or not host.endswith(".r2.cloudflarestorage.com")
        or host.count(".") != 3
        or parts.port not in (None, 443)
        or parts.username or parts.password or parts.fragment
        or not parts.path.startswith("/")
    ):
        raise ValueError("Only signed Cloudflare R2 URLs are accepted")
    params = parse_qs(parts.query)
    if "X-Amz-Signature" not in params or "X-Amz-Expires" not in params:
        raise ValueError("R2 URL must be presigned")
    return value


def _download(url, target):
    with requests.get(_r2_url(url), stream=True, timeout=(15, 90), allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Unexpected R2 download status")
        length = int(response.headers.get("content-length", "0"))
        if length > MAX_INPUT_BYTES:
            raise ValueError("Video exceeds 120 MiB")
        size = 0
        with open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_INPUT_BYTES:
                    raise ValueError("Video exceeds 120 MiB")
                out.write(chunk)
    if size < 512:
        raise ValueError("Video is empty or too small")
    return size


def _probe(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    info = json.loads(p.stdout)
    video = next((stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("Input contains no video")
    w, h = int(video["width"]), int(video["height"])
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    if w < 128 or h < 128 or max(w, h) > MAX_SOURCE_EDGE:
        raise ValueError("Unsupported source resolution")
    if duration <= 0 or duration > MAX_DURATION:
        raise ValueError("Preview supports videos up to 20 seconds")
    return {"width": w, "height": h, "duration": round(duration, 2),
            "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", []))}


def _command(source, destination, short_edge, report):
    return [
        "python", CLI, str(source),
        "--dit_model", "seedvr2_ema_3b-Q4_K_M.gguf",
        "--model_dir", MODEL_DIR,
        "--resolution", str(short_edge),
        "--batch_size", "5",
        "--chunk_size", "33",
        "--temporal_overlap", "3",
        "--memory_policy", "staged",
        "--attention_mode", "sdpa",
        "--video_backend", "ffmpeg",
        "--output", str(destination),
        "--memory_report", str(report),
    ]


def _remux(source_video, generated_video, target, has_audio):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(generated_video)]
    if has_audio:
        cmd += ["-i", str(source_video), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0", "-c:v", "copy", "-an"]
    cmd += ["-movflags", "+faststart", str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)


def _upload(url, path):
    size = Path(path).stat().st_size
    if size <= 512 or size > MAX_OUTPUT_BYTES:
        raise ValueError("Upscaled MP4 size is invalid")
    with open(path, "rb") as body:
        response = requests.put(
            _r2_url(url), data=body,
            headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
            timeout=(15, 300), allow_redirects=False,
        )
        response.raise_for_status()
    if response.status_code not in (200, 201, 204):
        raise RuntimeError("R2 upload returned unexpected status")
    return size


def handle(event):
    data = event.get("input", {}) if isinstance(event, dict) else {}
    if not isinstance(data, dict):
        raise ValueError("Invalid input")
    action = data.get("action", "health")
    if action == "health":
        return {"status": "ready", "revision": REVISION, "model": "SeedVR2 3B Q4_K_M",
                "model_ready": all((Path(MODEL_DIR) / name).is_file() for name in
                    ("seedvr2_ema_3b-Q4_K_M.gguf", "ema_vae_fp16.safetensors"))}
    if action != "upscale":
        raise ValueError("Unsupported action")
    input_url = _r2_url(data.get("input_video_url"))
    upload_url = _r2_url(data.get("output_upload_url"))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="seedvr2_") as tmp:
        root = Path(tmp)
        source, raw, output = (root / name for name in ("source.mp4", "enhanced.mp4", "result.mp4"))
        _download(input_url, source)
        meta = _probe(source)
        short_edge = min(meta["width"], meta["height"]) * 2
        if short_edge > 2160:
            raise ValueError("Requested output exceeds test limit")
        print(f"[SEEDVR2] start resolution={meta['width']}x{meta['height']} "
              f"duration={meta['duration']} input_audio={meta['has_audio']}", flush=True)
        cmd = _command(source, raw, short_edge, root / "memory.jsonl")
        subprocess.run(cmd, cwd="/opt/seedvr2", check=True, timeout=1500)
        if not raw.is_file():
            raise RuntimeError("SeedVR2 completed without MP4 output")
        _remux(source, raw, output, meta["has_audio"])
        enhanced = _probe(output)
        if meta["has_audio"] and not enhanced["has_audio"]:
            raise RuntimeError("Audio was not preserved")
        size = _upload(upload_url, output)
    elapsed = round(time.monotonic() - start, 2)
    print(f"[SEEDVR2] completed elapsed_s={elapsed} bytes={size}", flush=True)
    return {"status": "success", "revision": REVISION, "source": meta,
            "output": enhanced, "output_bytes": size, "execution_seconds": elapsed,
            "uploaded_to_r2": True}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handle})
