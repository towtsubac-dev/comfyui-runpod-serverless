import base64
import os
import time
import uuid
import json
import subprocess

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
}


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
        for key in (
            "images",
            "gifs",
            "videos",
        ):
            items = node_output.get(key)

            if not items:
                continue

            for item in items:
                filename = item.get(
                    "filename",
                    "",
                )

                if not filename:
                    continue

                result.append(
                    {
                        "type": key,
                        "filename": filename,
                        "subfolder": item.get(
                            "subfolder",
                            "",
                        ),
                        "file_type": item.get(
                            "type",
                            "output",
                        ),
                        "node_id": node_id,
                    }
                )

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
):
    if seed is None:
        seed = (
            int.from_bytes(
                os.urandom(8),
                "big",
            )
            % 10_000_000_000
        )

    return {
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
        "object_info_error":
            object_info_error,
    }


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
