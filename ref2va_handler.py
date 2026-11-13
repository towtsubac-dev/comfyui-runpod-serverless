"""MiniMax H3 native image-reference video generation (Preview only).

This module deliberately does not alter FL2VA I2V/T2V/Extend code paths.
The ComfyUI API requires flat dotted autogrow inputs, indexed from zero:
ref_images.ref_image_0, ref_images.ref_image_1, ...
"""
import base64
import io
import ipaddress
import os
from urllib.parse import urlparse
import uuid

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

REF2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MIN_REFS = 2
MAX_REFS = 3  # Initial 24 GB GPU validation; native node supports up to 9.
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


def _download_reference(url):
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname or hostname in ("localhost",) or hostname.endswith((".local", ".internal", ".localhost")) or parsed.username or parsed.password:
        raise ValueError("Reference image URLs must be public HTTPS URLs.")
    try:
        ipaddress.ip_address(hostname)
        raise ValueError("Reference image URLs must use a public DNS hostname, not an IP address.")
    except ValueError as exc:
        if "not an IP address" in str(exc):
            raise
    with requests.get(url, timeout=(10, 60), stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Reference image URL redirected or returned an error.")
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("Reference image exceeds 12 MiB.")
            chunks.append(chunk)
    return b"".join(chunks)


def _decode_image(item):
    if isinstance(item, str):
        value = item
        kind = "image_url" if item.startswith("https://") else "image"
    elif isinstance(item, dict):
        if "image_url" in item:
            kind, value = "image_url", item["image_url"]
        elif "image" in item:
            kind, value = "image", item["image"]
        else:
            raise ValueError("Each reference must provide image_url or image.")
    else:
        raise ValueError("Reference image entries must be URLs, base64 strings or objects.")

    if not isinstance(value, str) or not value:
        raise ValueError("Empty reference image.")
    if kind == "image_url":
        raw = _download_reference(value)
    else:
        value = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 reference image.") from exc
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Reference image exceeds 12 MiB.")
    if not raw:
        raise ValueError("Reference image is empty.")

    # Decode and re-encode before uploading, so ComfyUI receives a real image.
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Reference image resolution exceeds 12 megapixels.")
            if min(source.size) < 128:
                raise ValueError("Reference image dimensions must be at least 128 pixels.")
            cleaned = ImageOps.exif_transpose(source).convert("RGB")
            out = io.BytesIO()
            cleaned.save(out, format="PNG")
            return out.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid reference image format or dimensions.") from exc


def build_ref2va_workflow(image_filenames, prompt, width=864, height=480,
                          length=124, steps=20, seed=None, ref_image_size="match", optional_loras=None):
    if not (MIN_REFS <= len(image_filenames) <= MAX_REFS):
        raise ValueError("Preview requires 2 or 3 reference images.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing input.prompt.")
    if width % 32 or height % 32 or not (384 <= width <= 1536) or not (384 <= height <= 1536):
        raise ValueError("Width and height must be multiples of 32 between 384 and 1536.")
    if not (5 <= length <= 362 and (length - 5) % 17 == 0):
        raise ValueError("H3 length must be 17n+5 (e.g. 124 frames).")
    if not (4 <= steps <= 40):
        raise ValueError("Steps must be 4–40.")
    if ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max.")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % 10_000_000_000

    ref_inputs = {
        "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "prompt": prompt, "width": width, "height": height,
        "length": length, "ref_image_size": ref_image_size,
    }
    workflow = {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": REF2VA_MODEL, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": ref_inputs},
        "8": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {"model": ["8", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["8", 0], "conditioning": ["6", 0]}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["13", 0], "guider": ["12", 0],
              "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["6", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": 24, "bit_depth": 8}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0],
              "filename_prefix": "video/MiniMax_H3_Ref2VA_" + uuid.uuid4().hex, "format": "mp4", "codec": "h264"}},
    }
    for index, filename in enumerate(image_filenames):
        node_id = str(20 + index)
        workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
        ref_inputs["ref_images.ref_image_" + str(index)] = [node_id, 0]
    # Only optional registered Ref2VA-compatible model LoRAs. Never load the
    # FL2VA Turbo LoRA in the standalone native Ref2VA workflow.
    selected = optional_loras or []
    if len(selected) > 5:
        raise ValueError("At most 5 optional LoRAs are supported.")
    model_ref = ["2", 0]
    for index, item in enumerate(selected):
        strength = float(item["strength"])
        if not 0.0 <= strength <= 1.5:
            raise ValueError("MiniMax Ref2VA LoRA strength must be 0–1.5.")
        node_id = str(30 + index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": item["lora_name"],
                "strength_model": strength,
            },
        }
        model_ref = [node_id, 0]
    workflow["8"]["inputs"]["model"] = model_ref
    return workflow


def ref2va_check(comfy_get, comfy_dir):
    expected = {
        "fl2va_diffusion_model": os.path.join(comfy_dir, "models", "diffusion_models", REF2VA_MODEL),
        "text_encoder": os.path.join(comfy_dir, "models", "text_encoders", TEXT_ENCODER),
        "video_vae": os.path.join(comfy_dir, "models", "vae", VIDEO_VAE),
        "audio_vae": os.path.join(comfy_dir, "models", "vae", AUDIO_VAE),
    }
    models = {key: {"exists": os.path.isfile(path),
                    "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0}
              for key, path in expected.items()}
    node_response = comfy_get("/object_info/MiniMaxH3ReferenceToVideo").json()
    nodes_ok = "MiniMaxH3ReferenceToVideo" in node_response
    ready = all(info["exists"] for info in models.values()) and nodes_ok
    return {"status": "ok" if ready else "error",
            "ref2va_ready": ready, "nodes_ok": nodes_ok, "models": models,
            "reference_images": {"min": MIN_REFS, "max": MAX_REFS},
            "checkpoint_family": "FL2VA", "reference_conditioning_proven": False}


def generate_ref2va(job_input, upload_image, queue_prompt, wait_for_result,
                    extract_outputs, download_output, comfy_dir, resolve_loras=None):
    refs = job_input.get("reference_images")
    if not isinstance(refs, list) or not MIN_REFS <= len(refs) <= MAX_REFS:
        raise ValueError("input.reference_images must contain 2 or 3 images.")
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing input.prompt.")
    width = int(job_input.get("width", 864))
    height = int(job_input.get("height", 480))
    length = int(job_input.get("length", 124))
    steps = int(job_input.get("steps", 20))
    seed = job_input.get("seed")
    seed = int(seed) if seed is not None else None
    ref_image_size = job_input.get("ref_image_size", "match")
    optional_loras = []
    if job_input.get("loras") or job_input.get("lora"):
        if resolve_loras is None:
            raise ValueError("LoRA resolver is not available.")
        prompt, optional_loras, _ = resolve_loras(job_input, prompt)
        if len(optional_loras) > 5:
            raise ValueError("At most 5 optional LoRAs are supported.")

    # Validate all input metadata BEFORE uploading anything.
    if width % 32 or height % 32 or not (384 <= width <= 1536) or not (384 <= height <= 1536):
        raise ValueError("Invalid H3 dimensions.")
    if not (5 <= length <= 362 and (length - 5) % 17 == 0):
        raise ValueError("Invalid H3 length.")
    if not (4 <= steps <= 40) or ref_image_size not in ("match", "max"):
        raise ValueError("Invalid Ref2VA generation settings.")

    uploaded = []
    try:
        for index, item in enumerate(refs):
            encoded = _decode_image(item)
            filename = "h3_ref2va_" + uuid.uuid4().hex + "_" + str(index) + ".png"
            actual = upload_image(encoded, filename)
            uploaded.append(actual)
        workflow = build_ref2va_workflow(uploaded, prompt, width, height, length, steps, seed, ref_image_size, optional_loras=optional_loras)
        prompt_id = queue_prompt(workflow)
        history = wait_for_result(prompt_id, timeout=1800)
        outputs = []
        for output in extract_outputs(history):
            if not output["filename"].lower().endswith(".mp4"):
                continue
            content = download_output(output["filename"], output.get("subfolder", ""),
                                      output.get("file_type", "output"))
            outputs.append({"type": output["type"], "filename": output["filename"],
                            "data": base64.b64encode(content).decode("utf-8")})
        if not outputs:
            raise RuntimeError("Ref2VA finished but no MP4 was saved.")
        return {"status": "ok", "mode": "ref2va", "prompt_id": prompt_id,
                "reference_image_count": len(uploaded), "lora_count": len(optional_loras), "outputs": outputs}
    finally:
        input_root = os.path.join(comfy_dir, "input")
        for filename in uploaded:
            # Only delete images created for this request, never user-supplied paths.
            if filename.startswith("h3_ref2va_") and filename.endswith(".png") and os.path.basename(filename) == filename:
                try:
                    os.remove(os.path.join(input_root, filename))
                except FileNotFoundError:
                    pass
