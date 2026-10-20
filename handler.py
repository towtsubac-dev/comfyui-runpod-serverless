import base64
import os
import time
import uuid
import json
import subprocess
import hashlib
import re
from urllib.parse import urlparse

import requests
import runpod


# ============================================================
# ComfyUI
# ============================================================

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_DIR = "/workspace/ComfyUI"


# ============================================================
# MiniMax H3 models
# ============================================================

H3_MODEL_FILES = {
    "diffusion_model":
        "/workspace/ComfyUI/models/diffusion_models/"
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",

    "text_encoder":
        "/workspace/ComfyUI/models/text_encoders/"
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",

    "video_vae":
        "/workspace/ComfyUI/models/vae/"
        "minimax_h3_video_vae_int8_convrot.safetensors",

    "audio_vae":
        "/workspace/ComfyUI/models/vae/"
        "minimax_h3_audio_vae_fp32.safetensors",

    "turbo_lora":
        "/workspace/ComfyUI/models/loras/"
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",

    "hmnsfw_lora":
        "/workspace/ComfyUI/models/loras/"
        "HMNSFW-AIO-V2.5.safetensors",

    "free_pussy_lora":
        "/workspace/ComfyUI/models/loras/"
        "MM-H3 - Pussy Lora.safetensors",

    "hmpenis_lora":
        "/workspace/ComfyUI/models/loras/"
        "HMPenis_v2_e35.safetensors",
}

H3_MODEL_NAMES = {
    "diffusion_model":
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder":
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae":
        "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae":
        "minimax_h3_audio_vae_fp32.safetensors",
    "turbo_lora":
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",

    "hmnsfw_lora":
        "HMNSFW-AIO-V2.5.safetensors",

    "free_pussy_lora":
        "MM-H3 - Pussy Lora.safetensors",

    "hmpenis_lora":
        "HMPenis_v2_e35.safetensors",
}


OPTIONAL_LORA_CONFIG = {
    "hmnsfw": {
        "model_key": "hmnsfw_lora",
        "trigger": "hmmotion",
        "strength": 0.7,
    },
    "free_pussy": {
        "model_key": "free_pussy_lora",
        "trigger": "pussy",
        "strength": 1.0,
    },
    "hmpenis": {
        "model_key": "hmpenis_lora",
        "trigger": "HMPenis",
        "strength": 1.0,
    },
}


def download_registered_lora(item):
    registered = item.get("registered")
    if not isinstance(registered, dict):
        raise ValueError("Missing registered LoRA metadata.")
    lora_id = registered.get("id")
    digest = registered.get("sha256")
    size = registered.get("size")
    url = registered.get("url")
    if not isinstance(lora_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", lora_id):
        raise ValueError("Invalid registered LoRA ID.")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid registered LoRA SHA256.")
    if type(size) is not int or not 16 <= size <= 2 * 1024 ** 3:
        raise ValueError("Invalid registered LoRA size.")
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".r2.cloudflarestorage.com"):
        raise ValueError("Invalid registered LoRA download host.")

    directory = os.path.join(COMFYUI_DIR, "models", "loras")
    os.makedirs(directory, exist_ok=True)
    filename = "library_" + lora_id + "_" + digest[:16] + ".safetensors"
    target = os.path.join(directory, filename)
    if os.path.isfile(target) and os.path.getsize(target) == size:
        with open(target, "rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == digest:
                return filename
    temporary = target + "." + uuid.uuid4().hex + ".tmp"
    try:
        hasher = hashlib.sha256()
        total = 0
        with requests.get(url, stream=True, timeout=(15, 180), allow_redirects=False) as response:
            response.raise_for_status()
            with open(temporary, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > size:
                        raise ValueError("Registered LoRA exceeds declared size.")
                    hasher.update(chunk)
                    output.write(chunk)
        if total != size or hasher.hexdigest() != digest:
            raise ValueError("Registered LoRA checksum or size mismatch.")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return filename


def resolve_optional_loras(job_input, prompt):
    requested = job_input.get("loras")
    if requested is None:
        # Backward compatibility with requests using the former single-LoRA field.
        selection = str(job_input.get("lora", "none")).strip().lower()
        requested = [] if selection in ("", "none") else [
            {
                "name": selection,
                "strength": OPTIONAL_LORA_CONFIG.get(selection, {}).get("strength"),
            }
        ]

    if not isinstance(requested, list) or len(requested) > 3:
        raise ValueError("input.loras must be a list of at most 3 LoRAs.")

    selected = []
    seen = set()
    added_triggers = []

    for item in requested:
        if not isinstance(item, dict):
            raise ValueError("Each input.loras item must be an object.")

        selection = str(item.get("name", "")).strip().lower()
        config = OPTIONAL_LORA_CONFIG.get(selection)
        is_library = selection.startswith("library:") and re.fullmatch(r"library:[a-z0-9][a-z0-9_-]{0,63}", selection)
        if config is None and is_library:
            raw_strength = item.get("strength", 1.0)
            try:
                strength = float(raw_strength)
            except (TypeError, ValueError):
                raise ValueError("Invalid registered LoRA strength.")
            if not 0.0 <= strength <= 1.5:
                raise ValueError("Registered LoRA strength must be between 0.0 and 1.5.")
            if selection in seen:
                raise ValueError("Duplicate optional LoRA selection.")
            seen.add(selection)
            registered = item.get("registered")
            if not isinstance(registered, dict) or registered.get("id") != selection[8:]:
                raise ValueError("Registered LoRA ID mismatch.")
            filename = download_registered_lora(item)
            selected.append({"key": selection, "lora_name": filename, "strength": strength, "trigger": ""})
            continue
        if config is None:
            raise ValueError(f"Unsupported optional LoRA selection: {selection}")

        if selection in seen:
            raise ValueError(f"Duplicate optional LoRA selection: {selection}")
        seen.add(selection)

        raw_strength = item.get("strength", config["strength"])
        try:
            strength = float(raw_strength)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid strength for {selection}.")

        if not 0.0 <= strength <= 1.5:
            raise ValueError(f"Strength for {selection} must be between 0.0 and 1.5.")

        trigger = config["trigger"]
        trigger_added = trigger.lower() not in prompt.lower()
        if trigger_added:
            prompt = f"{trigger}, {prompt}"

        selected.append({
            "key": selection,
            "lora_name": H3_MODEL_NAMES[config["model_key"]],
            "strength": strength,
            "trigger": trigger,
        })
        added_triggers.append({
            "name": selection,
            "trigger": trigger,
            "added": trigger_added,
        })

    return prompt, selected, added_triggers


def log_lora_selection(mode, optional_loras, added_triggers):
    print(f"[H3] mode={mode}", flush=True)
    print(f"[LoRA] turbo=1.0", flush=True)

    if not optional_loras:
        print("[LoRA] optional=none", flush=True)
        return

    for item in optional_loras:
        print(
            f"[LoRA] optional={item['key']} strength={item['strength']:.2f}",
            flush=True,
        )

    for item in added_triggers:
        print(
            f"[LoRA] trigger={item['trigger']} added={str(item['added']).lower()}",
            flush=True,
        )


# ============================================================
# First test settings
# ============================================================

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_LENGTH = 124
DEFAULT_STEPS = 8
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0
DEFAULT_LORA_STRENGTH = 1.0
DEFAULT_TIMEOUT = 900


# ============================================================
# Start / wait for ComfyUI
# ============================================================

_comfy_process = None


def start_comfyui():
    global _comfy_process

    print("Starting ComfyUI...", flush=True)

    _comfy_process = subprocess.Popen(
        [
            "python",
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--use-sage-attention",
        ],
        cwd=COMFYUI_DIR,
        stdout=None,
        stderr=None,
    )

    return _comfy_process


def wait_for_comfyui(timeout=180):
    print("Waiting for ComfyUI...", flush=True)

    start = time.time()

    while time.time() - start < timeout:
        if _comfy_process is not None and _comfy_process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited during startup with code "
                f"{_comfy_process.returncode}"
            )

        try:
            response = requests.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=5,
            )

            if response.status_code == 200:
                print("ComfyUI is ready.", flush=True)
                return True

        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError("ComfyUI did not become ready in time.")


def ensure_comfyui():
    global _comfy_process

    if _comfy_process is None:
        start_comfyui()

    try:
        response = requests.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=5,
        )
        if response.status_code == 200:
            return True
    except Exception:
        pass

    return wait_for_comfyui()


# ============================================================
# HTTP helpers
# ============================================================

def comfy_get(path, params=None):
    response = requests.get(
        f"{COMFYUI_URL}{path}",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response


def comfy_post(path, payload, timeout=60):
    response = requests.post(
        f"{COMFYUI_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


# ============================================================
# Upload image
# ============================================================

def upload_image(image_bytes, filename="input.png"):
    response = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={
            "image": (
                filename,
                image_bytes,
                "application/octet-stream",
            )
        },
        data={
            "overwrite": "true",
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()
    print("Uploaded image:", data, flush=True)

    return data["name"]


# ============================================================
# Queue / result helpers
# ============================================================

def queue_prompt(workflow):
    client_id = str(uuid.uuid4())

    response = comfy_post(
        "/prompt",
        {
            "prompt": workflow,
            "client_id": client_id,
        },
        timeout=60,
    )

    data = response.json()

    if data.get("node_errors"):
        raise RuntimeError(
            f"ComfyUI node validation failed: {data['node_errors']}"
        )

    prompt_id = data.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(
            f"No prompt_id returned: {data}"
        )

    print(
        f"Queued ComfyUI prompt: {prompt_id}",
        flush=True,
    )

    return prompt_id


def wait_for_result(prompt_id, timeout=DEFAULT_TIMEOUT):
    print(
        f"Waiting for ComfyUI prompt: {prompt_id}",
        flush=True,
    )

    start = time.time()

    while time.time() - start < timeout:
        response = comfy_get(
            f"/history/{prompt_id}"
        )

        history = response.json()

        if prompt_id in history:
            record = history[prompt_id]

            status = record.get("status", {})
            status_str = status.get(
                "status_str",
                "",
            )

            print(
                "ComfyUI status:",
                status_str,
                flush=True,
            )

            if status_str == "success":
                return record

            if status_str in (
                "error",
                "failed",
                "canceled",
            ):
                raise RuntimeError(
                    f"ComfyUI job failed: {record}"
                )

        time.sleep(2)

    raise TimeoutError(
        f"ComfyUI job timed out after {timeout} seconds."
    )


# ============================================================
# Download output
# ============================================================

def download_output(
    filename,
    subfolder="",
    file_type="output",
):
    response = comfy_get(
        "/view",
        params={
            "filename": filename,
            "subfolder": subfolder,
            "type": file_type,
        },
    )

    return response.content


def extract_outputs(record):
    outputs = record.get("outputs", {})
    result = []

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for key, items in node_output.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                result.append({
                    "type": key,
                    "filename": filename,
                    "subfolder": item.get("subfolder", ""),
                    "file_type": item.get("type", "output"),
                    "node_id": node_id,
                })

    return result


# ============================================================
# Build H3 I2V workflow
# ============================================================

def build_h3_i2v_workflow(
    image_filename,
    prompt,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    length=DEFAULT_LENGTH,
    steps=DEFAULT_STEPS,
    seed=None,
    optional_loras=None,
):
    if seed is None:
        seed = (
            int.from_bytes(
                os.urandom(8),
                "big",
            )
            % 10_000_000_000
        )

    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename,
            },
        },

        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name":
                    H3_MODEL_NAMES[
                        "diffusion_model"
                    ],
                "weight_dtype": "default",
            },
        },

        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name":
                    H3_MODEL_NAMES[
                        "text_encoder"
                    ],
                "type": "minimax",
                "device": "default",
            },
        },

        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name":
                    H3_MODEL_NAMES[
                        "video_vae"
                    ],
            },
        },

        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name":
                    H3_MODEL_NAMES[
                        "audio_vae"
                    ],
            },
        },

        "6": {
            "class_type":
                "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "first_frame": ["1", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },

        "7": {
            "class_type":
                "EmptyMiniMaxH3LatentAV",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
            },
        },

        "8": {
            "class_type":
                "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["2", 0],
                "shift_video":
                    DEFAULT_SHIFT_VIDEO,
                "shift_audio":
                    DEFAULT_SHIFT_AUDIO,
            },
        },

        "9": {
            "class_type":
                "LoraLoaderModelOnly",
            "inputs": {
                "model": ["8", 0],
                "lora_name":
                    H3_MODEL_NAMES[
                        "turbo_lora"
                    ],
                "strength_model":
                    DEFAULT_LORA_STRENGTH,
            },
        },

        "10": {
            "class_type":
                "KSamplerSelect",
            "inputs": {
                "sampler_name":
                    "res_multistep",
            },
        },

        "11": {
            "class_type":
                "BasicScheduler",
            "inputs": {
                "model": ["9", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },

        "12": {
            "class_type":
                "BasicGuider",
            "inputs": {
                "model": ["9", 0],
                "conditioning": ["6", 0],
            },
        },

        "13": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },

        "14": {
            "class_type":
                "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": ["12", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 0],
            },
        },

        "15": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["4", 0],
            },
        },

        "16": {
            "class_type":
                "VAEDecodeAudio",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["5", 0],
            },
        },

        "17": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["15", 0],
                "audio": ["16", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },

        "18": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["17", 0],
                "filename_prefix":
                    "video/MiniMax_H3",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }

    if optional_loras:
        previous_model_node = "9"
        for index, lora in enumerate(optional_loras, start=19):
            node_id = str(index)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [previous_model_node, 0],
                    "lora_name": lora["lora_name"],
                    "strength_model": lora["strength"],
                },
            }
            previous_model_node = node_id

        workflow["11"]["inputs"]["model"] = [previous_model_node, 0]
        workflow["12"]["inputs"]["model"] = [previous_model_node, 0]

    return workflow


# ============================================================
# Build H3 T2V workflow
# ============================================================

def build_h3_t2v_workflow(prompt, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, length=DEFAULT_LENGTH, steps=DEFAULT_STEPS, seed=None, optional_loras=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % 10_000_000_000

    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_MODEL_NAMES["diffusion_model"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_MODEL_NAMES["text_encoder"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_MODEL_NAMES["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_MODEL_NAMES["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width, "height": height, "length": length}},
        "6": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": width, "height": height, "length": length}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": DEFAULT_SHIFT_VIDEO, "shift_audio": DEFAULT_SHIFT_AUDIO}},
        "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["7", 0], "lora_name": H3_MODEL_NAMES["turbo_lora"], "strength_model": DEFAULT_LORA_STRENGTH}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["8", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["8", 0], "conditioning": ["5", 0]}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24, "bit_depth": 8}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": "video/MiniMax_H3_T2V", "format": "mp4", "codec": "h264"}},
    }

    if optional_loras:
        previous_model_node = "8"
        for index, lora in enumerate(optional_loras, start=18):
            node_id = str(index)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [previous_model_node, 0],
                    "lora_name": lora["lora_name"],
                    "strength_model": lora["strength"],
                },
            }
            previous_model_node = node_id

        workflow["10"]["inputs"]["model"] = [previous_model_node, 0]
        workflow["11"]["inputs"]["model"] = [previous_model_node, 0]

    return workflow


# ============================================================
# Health
# ============================================================

def health_check():
    response = comfy_get("/system_stats")
    stats = response.json()

    return {
        "status": "ok",
        "message": "ComfyUI is running",
        "system_stats": stats,
    }


# ============================================================
# H3 check
# ============================================================

def h3_check():
    models = {}
    all_models_ok = True

    for name, path in H3_MODEL_FILES.items():
        exists = os.path.isfile(path)

        size_bytes = (
            os.path.getsize(path)
            if exists
            else 0
        )

        models[name] = {
            "exists": exists,
            "path": path,
            "size_bytes": size_bytes,
            "size_gb": round(
                size_bytes / (1024 ** 3),
                3,
            ),
        }

        if not exists:
            all_models_ok = False

    object_info_error = None
    
    h3_nodes = {}

    try:
        response = comfy_get("/object_info")
        object_info = response.json()

        for node_name, node_data in object_info.items():
            lower = node_name.lower()

            if (
                "minimaxh3" in lower
                or "minimax" in lower
                or lower ==
                "emptymanimaxh3latentav"
            ):
                h3_nodes[node_name] = {
                    "display_name":
                        node_data.get(
                            "display_name"
                        ),
                    "category":
                        node_data.get(
                            "category"
                        ),
                }

    except Exception as exc:
        object_info_error = str(exc)

    native_i2v_exists = (
        "MiniMaxH3ImageToVideo"
        in h3_nodes
    )

    return {
        "status": (
            "ok"
            if (
                all_models_ok
                and native_i2v_exists
            )
            else "error"
        ),
        "h3_ready":
            all_models_ok
            and native_i2v_exists,
        "models_ok": all_models_ok,
        "nodes_ok": native_i2v_exists,
        "models": models,
        "h3_nodes": h3_nodes,
        "object_info_error": object_info_error,
    }

# ============================================================
# H3 native I2V node schema
# ============================================================


# ============================================================
# H3 text-to-video generation
# ============================================================

def h3_t2v_generate(job_input):
    ensure_comfyui()
    prompt = job_input.get("prompt")
    if not prompt:
        raise ValueError("Missing input.prompt.")
    prompt, optional_loras, added_triggers = resolve_optional_loras(job_input, prompt)
    log_lora_selection("T2V", optional_loras, added_triggers)
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    length = int(job_input.get("length", DEFAULT_LENGTH))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    seed = job_input.get("seed")
    if seed is not None:
        seed = int(seed)
    workflow = build_h3_t2v_workflow(prompt=prompt, width=width, height=height, length=length, steps=steps, seed=seed, optional_loras=optional_loras)
    prompt_id = queue_prompt(workflow)
    history = wait_for_result(prompt_id)
    results = []
    for output in extract_outputs(history):
        content = download_output(output["filename"], output.get("subfolder", ""), output.get("file_type", "output"))
        results.append({"type": output["type"], "filename": output["filename"], "data": base64.b64encode(content).decode("utf-8")})
    return {"prompt_id": prompt_id, "outputs": results}


# ============================================================
# H3 image-to-video generation
# ============================================================

def h3_generate(job_input):
    ensure_comfyui()

    image_b64 = job_input.get("image")
    image_url = job_input.get("image_url")

    if not image_b64 and not image_url:
        raise ValueError(
            "Missing input.image or input.image_url."
        )

    prompt = job_input.get(
        "prompt",
        (
            "Create a natural cinematic video "
            "from the input image. "
            "Preserve the subject identity and "
            "overall composition. "
            "Use smooth realistic motion and "
            "subtle camera movement. "
            "Audio: natural ambient sound "
            "appropriate to the scene."
        ),
    )

    prompt, optional_loras, added_triggers = resolve_optional_loras(job_input, prompt)
    log_lora_selection("I2V", optional_loras, added_triggers)

    width = int(
        job_input.get(
            "width",
            DEFAULT_WIDTH,
        )
    )

    height = int(
        job_input.get(
            "height",
            DEFAULT_HEIGHT,
        )
    )

    length = int(
        job_input.get(
            "length",
            DEFAULT_LENGTH,
        )
    )

    steps = int(
        job_input.get(
            "steps",
            DEFAULT_STEPS,
        )
    )

    seed = job_input.get("seed")

    if seed is not None:
        seed = int(seed)

    if image_url:
        if not image_url.startswith(("https://", "http://")):
            raise ValueError(
                "input.image_url must start with http:// or https://"
            )

        try:
            response = requests.get(
                image_url,
                timeout=60,
                allow_redirects=True,
            )
            response.raise_for_status()
            image_bytes = response.content
        except Exception as exc:
            raise ValueError(
                f"Failed to download input.image_url: {exc}"
            )

        if not image_bytes:
            raise ValueError(
                "Downloaded image is empty."
            )

    else:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(
                ",",
                1,
            )[1]

        try:
            image_bytes = base64.b64decode(
                image_b64
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid base64 image: {exc}"
            )

    filename = (
        f"runpod_h3_{uuid.uuid4().hex}.png"
    )

    uploaded_filename = upload_image(
        image_bytes,
        filename,
    )

    workflow = build_h3_i2v_workflow(
        image_filename=uploaded_filename,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=seed,
        optional_loras=optional_loras,
    )

    prompt_id = queue_prompt(workflow)

    history = wait_for_result(
        prompt_id
    )

    output_files = extract_outputs(
        history
    )

    results = []

    for output in output_files:
        content = download_output(
            output["filename"],
            output.get("subfolder", ""),
            output.get(
                "file_type",
                "output",
            ),
        )

        results.append(
            {
                "type": output["type"],
                "filename":
                    output["filename"],
                "data":
                    base64.b64encode(
                        content
                    ).decode("utf-8"),
            }
        )

    return {
        "prompt_id": prompt_id,
        "outputs": results,
    }



# ============================================================
# H3 video extension
# ============================================================

def _probe_video(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,codec_type",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError("The uploaded file does not contain a video stream.")
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    frames = int(video_stream.get("nb_read_frames") or round(duration * 24))
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "frames": frames,
        "duration": duration,
        "has_audio": audio_stream is not None,
    }


def _normalize_video_for_concat(source, destination, width, height):
    info = _probe_video(source)
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "fps=24,setsar=1,format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-i", source]
    if info["has_audio"]:
        cmd += [
            "-map", "0:v:0", "-map", "0:a:0",
            "-vf", video_filter,
            "-af", "aresample=32000:async=1:first_pts=0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", destination,
        ]
    else:
        cmd += [
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", destination,
        ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)


def _concat_extended_video(original_path, segment_path, output_path):
    info = _probe_video(original_path)
    work_dir = os.path.dirname(output_path)
    normalized_original = os.path.join(work_dir, "normalized_original.mp4")
    normalized_segment = os.path.join(work_dir, "normalized_segment.mp4")
    _normalize_video_for_concat(
        original_path, normalized_original, info["width"], info["height"]
    )
    _normalize_video_for_concat(
        segment_path, normalized_segment, info["width"], info["height"]
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", normalized_original,
            "-i", normalized_segment,
            "-filter_complex",
            "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-movflags", "+faststart", output_path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    for path in (normalized_original, normalized_segment):
        try:
            os.remove(path)
        except OSError:
            pass


def build_h3_extend_workflow(video_filename, prompt, length, skip_first_frames,
                             frame_load_cap, steps=DEFAULT_STEPS, seed=None,
                             optional_loras=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % 10_000_000_000

    workflow = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_filename,
                "force_rate": 24,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_load_cap,
                "skip_first_frames": skip_first_frames,
                "select_every_nth": 1,
                "format": "AnimateDiff",
            },
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": H3_MODEL_NAMES["diffusion_model"], "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": H3_MODEL_NAMES["text_encoder"], "type": "minimax", "device": "default"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_MODEL_NAMES["video_vae"]},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_MODEL_NAMES["audio_vae"]},
        },
        "6": {
            "class_type": "MiniMaxH3EncodeAVPatched",
            "inputs": {
                "vae": ["4", 0],
                "images": ["1", 0],
                "audio_vae": ["5", 0],
                "audio": ["1", 2],
            },
        },
        "7": {
            "class_type": "MiniMaxH3VideoExtendPatched",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "context_latent": ["6", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt,
                "length": length,
                "context_frames": 2,
                "pin_last_frame": True,
            },
        },
        "8": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["2", 0],
                "shift_video": DEFAULT_SHIFT_VIDEO,
                "shift_audio": DEFAULT_SHIFT_AUDIO,
            },
        },
        "9": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["8", 0],
                "lora_name": H3_MODEL_NAMES["turbo_lora"],
                "strength_model": DEFAULT_LORA_STRENGTH,
            },
        },
        "10": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["9", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["9", 0], "conditioning": ["7", 0]},
        },
        "13": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": ["12", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 1],
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["4", 0]},
        },
        "16": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["14", 0], "vae": ["5", 0]},
        },
        "17": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": 24, "bit_depth": 8},
        },
        "18": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["17", 0],
                "filename_prefix": f"video/H3_Extend_{uuid.uuid4().hex}",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }

    if optional_loras:
        previous_model_node = "9"
        next_node_id = 19
        for lora in optional_loras:
            node_id = str(next_node_id)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [previous_model_node, 0],
                    "lora_name": lora["lora_name"],
                    "strength_model": lora["strength"],
                },
            }
            previous_model_node = node_id
            next_node_id += 1
        workflow["11"]["inputs"]["model"] = [previous_model_node, 0]
        workflow["12"]["inputs"]["model"] = [previous_model_node, 0]

    return workflow


def h3_extend_generate(job_input):
    ensure_comfyui()
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Missing input.prompt.")

    try:
        extend_seconds = int(job_input.get("extend_seconds", 5))
    except (TypeError, ValueError):
        raise ValueError("input.extend_seconds must be 3, 5, 8, or 10.")
    if extend_seconds not in (3, 5, 8, 10):
        raise ValueError("input.extend_seconds must be 3, 5, 8, or 10.")

    video_b64 = job_input.get("video")
    video_url = job_input.get("video_url")
    if not video_b64 and not video_url:
        raise ValueError("Missing input.video (base64 MP4) or input.video_url.")

    if video_url:
        if not str(video_url).startswith(("https://", "http://")):
            raise ValueError("input.video_url must start with http:// or https://")
        response = requests.get(video_url, timeout=90, allow_redirects=True)
        response.raise_for_status()
        video_bytes = response.content
    else:
        if video_b64.startswith("data:"):
            video_b64 = video_b64.split(",", 1)[1]
        try:
            video_bytes = base64.b64decode(video_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 video: {exc}")

    if not video_bytes:
        raise ValueError("Uploaded video is empty.")

    job_id = uuid.uuid4().hex
    input_name = f"runpod_h3_extend_{job_id}.mp4"
    input_path = os.path.join(COMFYUI_DIR, "input", input_name)
    temp_dir = os.path.join("/workspace", f"h3_extend_{job_id}")
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with open(input_path, "wb") as video_file:
            video_file.write(video_bytes)

        # Normalize only the extension input: H3 requires dimensions divisible by
        # 32. Preserve aspect ratio and pad to the nearest compatible dimensions.
        raw_info = _probe_video(input_path)
        if raw_info["frames"] < 5:
            raise ValueError("The input video is too short for MiniMax H3 extension.")

        target_width = max(32, round(raw_info["width"] / 32) * 32)
        target_height = max(32, round(raw_info["height"] / 32) * 32)
        normalized_path = os.path.join(temp_dir, "h3_context_normalized.mp4")
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,fps=24,format=yuv420p"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
                "-shortest", "-movflags", "+faststart", normalized_path,
            ],
            check=True, capture_output=True, text=True, timeout=300,
        )
        os.replace(normalized_path, input_path)
        info = _probe_video(input_path)

        # H3 frame counts must be 17n+5. Select the largest compatible trailing
        # context up to 124 frames instead of rejecting otherwise usable videos.
        frame_load_cap = min(124, info["frames"])
        frame_load_cap -= (frame_load_cap - 5) % 17
        if frame_load_cap < 5:
            raise ValueError("The input video is too short for MiniMax H3 extension.")
        skip_first_frames = max(0, info["frames"] - frame_load_cap)

        prompt, optional_loras, added_triggers = resolve_optional_loras(job_input, prompt)
        log_lora_selection("EXTEND", optional_loras, added_triggers)
        steps = int(job_input.get("steps", DEFAULT_STEPS))
        seed = job_input.get("seed")
        if seed is not None:
            seed = int(seed)

        workflow = build_h3_extend_workflow(
            video_filename=input_name,
            prompt=prompt,
            length=max(5, round(extend_seconds * 24)) +
                   (5 - (max(5, round(extend_seconds * 24)) % 17)) % 17,
            skip_first_frames=skip_first_frames,
            frame_load_cap=frame_load_cap,
            steps=steps,
            seed=seed,
            optional_loras=optional_loras,
        )
        prompt_id = queue_prompt(workflow)
        history = wait_for_result(prompt_id, timeout=1800)
        output_files = extract_outputs(history)
        print(
            "[H3 EXTEND] SaveVideo node output:",
            json.dumps(history.get("outputs", {}).get("18", {}), ensure_ascii=False, default=str)[:8000],
            flush=True,
        )
        segment = next(
            (
                item
                for item in output_files
                if item.get("filename", "").lower().endswith(".mp4")
            ),
            None,
        )
        if segment is None:
            raise RuntimeError(
                f"H3 extension completed without producing an MP4 video. Outputs: {output_files}"
            )
        segment_path = os.path.join(
            COMFYUI_DIR, "output", segment.get("subfolder", ""), segment["filename"]
        )
        if not os.path.isfile(segment_path):
            raise RuntimeError(f"Generated extension segment was not found: {segment_path}")

        combined_path = os.path.join(temp_dir, "extended_video.mp4")
        _concat_extended_video(input_path, segment_path, combined_path)
        with open(combined_path, "rb") as result_file:
            combined_bytes = result_file.read()

        return {
            "status": "ok",
            "prompt_id": prompt_id,
            "extend_seconds_requested": extend_seconds,
            "extension_frames": workflow["7"]["inputs"]["length"],
            "original_duration_seconds": round(info["duration"], 3),
            "outputs": [{
                "type": "videos",
                "filename": f"MiniMax_H3_Extended_{job_id}.mp4",
                "data": base64.b64encode(combined_bytes).decode("utf-8"),
            }],
        }
    finally:
        for path in (input_path,):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================
# Generic workflow
# ============================================================

def run_workflow(
    workflow,
    image_bytes=None,
    image_filename="input.png",
    image_node_id=None,
    image_input_name=None,
    prompt_node_id=None,
    prompt_input_name=None,
    prompt_text=None,
    overrides=None,
):
    ensure_comfyui()

    workflow = json.loads(
        json.dumps(workflow)
    )

    if image_bytes is not None:
        uploaded_name = upload_image(
            image_bytes,
            image_filename,
        )

        if (
            image_node_id is not None
            and image_input_name is not None
        ):
            workflow[
                str(image_node_id)
            ][
                "inputs"
            ][
                image_input_name
            ] = uploaded_name

    if (
        prompt_text is not None
        and prompt_node_id is not None
        and prompt_input_name is not None
    ):
        workflow[
            str(prompt_node_id)
        ][
            "inputs"
        ][
            prompt_input_name
        ] = prompt_text

    if overrides:
        for node_id, values in overrides.items():
            node_id = str(node_id)

            if node_id not in workflow:
                raise KeyError(
                    f"Workflow node not found: "
                    f"{node_id}"
                )

            for key, value in values.items():
                workflow[
                    node_id
                ][
                    "inputs"
                ][key] = value

    prompt_id = queue_prompt(
        workflow
    )

    history = wait_for_result(
        prompt_id
    )

    output_files = extract_outputs(
        history
    )

    results = []

    for output in output_files:
        content = download_output(
            output["filename"],
            output.get("subfolder", ""),
            output.get(
                "file_type",
                "output",
            ),
        )

        results.append(
            {
                "type": output["type"],
                "filename":
                    output["filename"],
                "data":
                    base64.b64encode(
                        content
                    ).decode("utf-8"),
            }
        )

    return {
        "prompt_id": prompt_id,
        "outputs": results,
    }


# ============================================================
# Main handler
# ============================================================

def handler(event):
    try:
        input_data = event.get(
            "input",
            {},
        )

        action = input_data.get(
            "action",
            "generate",
        )

        if action == "health":
            return health_check()

        if action == "h3_check":
            return h3_check()

        if action == "h3_generate":
            return h3_generate(
                input_data
            )

        if action == "h3_t2v_generate":
            return h3_t2v_generate(
                input_data
            )

        if action == "h3_extend_generate":
            return h3_extend_generate(input_data)

        if action == "generate":
            workflow = input_data.get(
                "workflow"
            )

            if workflow is None:
                raise ValueError(
                    "workflow is required"
                )

            image_bytes = None

            image_base64 = input_data.get(
                "image"
            )

            if image_base64:
                if image_base64.startswith(
                    "data:"
                ):
                    image_base64 = (
                        image_base64.split(
                            ",",
                            1,
                        )[1]
                    )

                image_bytes = (
                    base64.b64decode(
                        image_base64
                    )
                )

            return run_workflow(
                workflow=workflow,
                image_bytes=image_bytes,
                image_filename=input_data.get(
                    "image_filename",
                    "input.png",
                ),
                image_node_id=input_data.get(
                    "image_node_id"
                ),
                image_input_name=input_data.get(
                    "image_input_name"
                ),
                prompt_node_id=input_data.get(
                    "prompt_node_id"
                ),
                prompt_input_name=input_data.get(
                    "prompt_input_name"
                ),
                prompt_text=input_data.get(
                    "prompt"
                ),
                overrides=input_data.get(
                    "overrides"
                ),
            )

        raise ValueError(
            f"Unknown action: {action}"
        )

    except Exception as e:
        print(
            f"Handler error: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ============================================================
# RunPod Serverless
# ============================================================

if __name__ == "__main__":
    print(
        "Starting ComfyUI...",
        flush=True,
    )

    start_comfyui()

    wait_for_comfyui()

    print(
        "Starting RunPod Serverless handler...",
        flush=True,
    )

    runpod.serverless.start(
        {
            "handler": handler
        }
    )
