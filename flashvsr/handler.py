"""FlashVSR+ Turbo / RunPod Serverless isolated preview.
Only presigned Cloudflare R2 GET/PUT URLs. No storage credentials enter the worker.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

import requests

REVISION = "flashvsr-plus-turbo-preview-v1"
BASE = Path("/opt/flashvsr")
MAX_INPUT_BYTES = 120 * 1024 * 1024
MAX_OUTPUT_BYTES = 300 * 1024 * 1024
MAX_DURATION_SECONDS = 12.0
MAX_INPUT_EDGE = 1024
MAX_FRAMES = 300
MAX_FPS = 30
INFERENCE_TIMEOUT_SECONDS = 300
MODEL_FILES = (
    BASE / "ckpts/turbo/dit.safetensors",
    BASE / "ckpts/turbo/lq_proj_in.safetensors",
    BASE / "ckpts/TCDecoder.ckpt",
    BASE / "prompt_tensor/posi_prompt.pth",
)


def _signed_r2_url(value):
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("Expected a signed R2 URL")
    parts = urlsplit(value)
    hostname = (parts.hostname or "").lower()
    # Fixed account-style R2 host: <account_id>.r2.cloudflarestorage.com
    if (parts.scheme != "https" or not hostname.endswith(".r2.cloudflarestorage.com")
            or hostname.count(".") != 3 or parts.port not in (None, 443)
            or parts.username or parts.password or parts.fragment
            or not parts.path.startswith("/")):
        raise ValueError("Only signed Cloudflare R2 object URLs are permitted")
    params = parse_qs(parts.query)
    if not all(params.get(key) for key in ("X-Amz-Signature", "X-Amz-Expires", "X-Amz-Date")):
        raise ValueError("R2 URL must be signed and expiring")
    if not 1 <= int(params["X-Amz-Expires"][0]) <= 3600:
        raise ValueError("R2 URL expiration is invalid")
    return value


def _download(url, dest):
    with requests.get(_signed_r2_url(url), timeout=(15, 90),
                      stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Unexpected R2 GET status")
        content_length = int(response.headers.get("content-length", 0))
        if content_length > MAX_INPUT_BYTES:
            raise ValueError("Source video exceeds 120 MiB")
        size = 0
        with open(dest, "wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    size += len(chunk)
                    if size > MAX_INPUT_BYTES:
                        raise ValueError("Source video exceeds 120 MiB")
                    target.write(chunk)
    if size < 512:
        raise ValueError("Input MP4 is empty")
    return size


def _probe(path):
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    info = json.loads(output.stdout)
    streams = info.get("streams", [])
    stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not stream:
        raise ValueError("MP4 has no video stream")
    fps = float(Fraction(stream.get("avg_frame_rate") or "0/1"))
    duration = float(info.get("format", {}).get("duration") or 0)
    frames = int(stream.get("nb_frames") or round(fps * duration))
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": round(fps, 3), "frames": frames, "duration": round(duration, 3),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _geometry(meta):
    width, height = meta["width"], meta["height"]
    if (min(width, height) < 128 or max(width, height) > MAX_INPUT_EDGE
            or meta["duration"] <= 0 or meta["duration"] > MAX_DURATION_SECONDS
            or meta["fps"] <= 0 or meta["fps"] > MAX_FPS
            or meta["frames"] <= 0 or meta["frames"] > MAX_FRAMES):
        raise ValueError("Preview accepts <=12s, <=300 frames, <=30fps and input sides 128–1024 px")
    target_w, target_h = 2 * width, 2 * height
    canvas_w = math.ceil(target_w / 128) * 128
    canvas_h = math.ceil(target_h / 128) * 128
    return {
        "target_width": target_w, "target_height": target_h,
        "canvas_width": canvas_w, "canvas_height": canvas_h,
        "padding_required": (canvas_w != target_w or canvas_h != target_h),
    }


def _prepare_source(original, prepared, meta, geo):
    if not geo["padding_required"]:
        return original
    padded_w, padded_h = geo["canvas_width"] // 2, geo["canvas_height"] // 2
    x = (padded_w - meta["width"]) // 2
    y = (padded_h - meta["height"]) // 2
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(original),
        "-vf", f"pad={padded_w}:{padded_h}:{x}:{y}:black",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", str(prepared),
    ], check=True, timeout=60)
    return prepared


def _command(source, output, geo):
    return [
        "python", "-m", "inference.run_stream",
        "--input", str(source), "--output", str(output),
        "--mode", "turbo", "--height", str(geo["canvas_height"]),
        "--width", str(geo["canvas_width"]),
        "--attn-kernel", "triton_block_sparse",
        "--temporal-compression", "4",
        "--tiny-decoder-mode", "serial",
        "--decode-every", "28",
        "--video-preset", "veryfast",
    ]


def _finalize(original, raw, output, meta, geo):
    generated = _probe(raw)
    if (generated["width"], generated["height"]) != (
        geo["canvas_width"], geo["canvas_height"]
    ):
        raise RuntimeError("FlashVSR+ output canvas dimensions do not match request")
    # Copy when exact 2x already matches the 128-pixel kernel granularity.
    if not geo["padding_required"] and (not meta["has_audio"] or generated["has_audio"]):
        shutil.copyfile(raw, output)
        return
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(raw)]
    if meta["has_audio"] and not generated["has_audio"]:
        cmd += ["-i", str(original)]
    if geo["padding_required"]:
        left = (geo["canvas_width"] - geo["target_width"]) // 2
        top = (geo["canvas_height"] - geo["target_height"]) // 2
        cmd += ["-vf", f"crop={geo['target_width']}:{geo['target_height']}:{left}:{top}"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-map", "0:v:0"]
    if generated["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    elif meta["has_audio"]:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True, timeout=90)


def _upload(url, path):
    size = Path(path).stat().st_size
    if size < 512 or size > MAX_OUTPUT_BYTES:
        raise ValueError("Final MP4 size outside permitted bounds")
    with open(path, "rb") as body:
        resp = requests.put(
            _signed_r2_url(url), data=body, allow_redirects=False,
            headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
            timeout=(15, 180),
        )
    resp.raise_for_status()
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError("R2 upload returned unexpected status")
    return size


def handle(event):
    payload = event.get("input", {}) if isinstance(event, dict) else {}
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    action = payload.get("action", "health")
    if action == "health":
        import torch
        import triton
        import flashvsr_attn
        return {"status": "ready", "revision": REVISION,
                "model": "FlashVSR+ Turbo", "model_ready": all(p.is_file() for p in MODEL_FILES),
                "torch": torch.__version__, "triton": triton.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "attn_installed": True}
    if action != "upscale":
        raise ValueError("Unsupported action")
    input_url = _signed_r2_url(payload.get("input_video_url"))
    output_url = _signed_r2_url(payload.get("output_upload_url"))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="flashvsr_") as tmp:
        root = Path(tmp)
        original = root / "input.mp4"
        prepared = root / "padded.mp4"
        raw = root / "flashvsr.mp4"
        final = root / "result.mp4"
        source_bytes = _download(input_url, original)
        info = _probe(original)
        geometry = _geometry(info)
        used = _prepare_source(original, prepared, info, geometry)
        print(f"[FLASHVSR] start source={info['width']}x{info['height']} "
              f"frames={info['frames']} fps={info['fps']} bytes={source_bytes} "
              f"canvas={geometry['canvas_width']}x{geometry['canvas_height']}", flush=True)
        begin = time.monotonic()
        try:
            subprocess.run(_command(used, raw, geometry),
                           cwd=str(BASE), check=True,
                           timeout=INFERENCE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            print("[FLASHVSR] inference_aborted reason=300s_budget_exceeded", flush=True)
            raise RuntimeError("FlashVSR+ inference exceeded the 300-second test budget") from error
        inference_s = round(time.monotonic() - begin, 2)
        if not raw.is_file():
            raise RuntimeError("FlashVSR+ did not produce the requested MP4")
        _finalize(original, raw, final, info, geometry)
        out = _probe(final)
        if (out["width"], out["height"]) != (
            geometry["target_width"], geometry["target_height"]
        ):
            raise RuntimeError("Output was not cropped back to exactly 2x")
        if info["has_audio"] and not out["has_audio"]:
            raise RuntimeError("Source audio was not preserved")
        if abs(out["duration"] - info["duration"]) > 1.0:
            raise RuntimeError("Output duration differs from source")
        output_bytes = _upload(output_url, final)
    duration = round(time.monotonic() - start, 2)
    print(f"[FLASHVSR] completed inference_s={inference_s} total_s={duration} "
          f"output_bytes={output_bytes}", flush=True)
    return {"status": "success", "revision": REVISION, "mode": "turbo",
            "attn": "triton_block_sparse", "source": info, "output": out,
            "inference_seconds": inference_s, "execution_seconds": duration,
            "output_bytes": output_bytes, "uploaded_to_r2": True}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handle})
