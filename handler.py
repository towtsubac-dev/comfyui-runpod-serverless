import os
import io
import json
import time
import base64
import threading
import subprocess

import requests
import runpod


# =========================================================
# Configuration
# =========================================================

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

COMFYUI_DIR = "/workspace/ComfyUI"

STARTUP_TIMEOUT = 180
POLL_INTERVAL = 1.0
REQUEST_TIMEOUT = 30


# =========================================================
# MiniMax H3 model files
# =========================================================

H3_MODEL_FILES = {
    "diffusion_model": (
        "/workspace/ComfyUI/models/diffusion_models/"
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    ),

    "text_encoder": (
        "/workspace/ComfyUI/models/text_encoders/"
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    ),

    "video_vae": (
        "/workspace/ComfyUI/models/vae/"
        "minimax_h3_video_vae_int8_convrot.safetensors"
    ),

    "audio_vae": (
        "/workspace/ComfyUI/models/vae/"
        "minimax_h3_audio_vae_fp32.safetensors"
    ),

    "turbo_lora": (
        "/workspace/ComfyUI/models/loras/"
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    ),
}


# =========================================================
# Global state
# =========================================================

_comfy_process = None
_comfy_lock = threading.Lock()


# =========================================================
# Utility
# =========================================================

def file_info(path):
    """
    Return information about a model file.
    """

    exists = os.path.isfile(path)

    if not exists:
        return {
            "exists": False,
            "path": path,
            "size_bytes": None,
            "size_gb": None,
        }

    size = os.path.getsize(path)

    return {
        "exists": True,
        "path": path,
        "size_bytes": size,
        "size_gb": round(size / (1024 ** 3), 3),
    }


# =========================================================
# Start ComfyUI
# =========================================================

def start_comfyui():
    global _comfy_process

    with _comfy_lock:

        # Already running
        if _comfy_process is not None:
            if _comfy_process.poll() is None:
                return

        print("Starting ComfyUI...")

        cmd = [
            "python",
            os.path.join(COMFYUI_DIR, "main.py"),
            "--listen",
            COMFYUI_HOST,
            "--port",
            str(COMFYUI_PORT),
        ]

        _comfy_process = subprocess.Popen(
            cmd,
            cwd=COMFYUI_DIR,
            stdout=None,
            stderr=None,
        )

    # Wait until ComfyUI responds
    start_time = time.time()

    while time.time() - start_time < STARTUP_TIMEOUT:

        try:
            r = requests.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=5,
            )

            if r.status_code == 200:
                print("ComfyUI is ready.")
                return

        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError(
        "ComfyUI did not become ready within timeout."
    )


# =========================================================
# Ensure ComfyUI
# =========================================================

def ensure_comfyui():
    try:
        r = requests.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=5,
        )

        if r.status_code == 200:
            return

    except Exception:
        pass

    start_comfyui()


# =========================================================
# Health check
# =========================================================

def health_check():

    ensure_comfyui()

    r = requests.get(
        f"{COMFYUI_URL}/system_stats",
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    return {
        "status": "ok",
        "message": "ComfyUI is running",
        "system_stats": r.json(),
    }


# =========================================================
# H3 model check
# =========================================================

def h3_check():

    ensure_comfyui()

    # -----------------------------------------------------
    # Check model files
    # -----------------------------------------------------

    models = {}

    for name, path in H3_MODEL_FILES.items():
        models[name] = file_info(path)

    models_ok = all(
        item["exists"]
        for item in models.values()
    )

    # -----------------------------------------------------
    # Check ComfyUI H3 nodes
    # -----------------------------------------------------

    h3_nodes = {}
    object_info_error = None

    try:

        r = requests.get(
            f"{COMFYUI_URL}/object_info",
            timeout=REQUEST_TIMEOUT,
        )

        r.raise_for_status()

        object_info = r.json()

        for node_name, node_data in object_info.items():

            if (
                "MiniMaxH3" in node_name
                or "MiniMax" in node_name
                or "H3" in node_name
            ):
                h3_nodes[node_name] = {
                    "display_name": node_data.get(
                        "display_name"
                    ),
                    "category": node_data.get(
                        "category"
                    ),
                }

    except Exception as e:

        object_info_error = str(e)

    nodes_ok = len(h3_nodes) > 0

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    return {
        "status": "ok",

        "h3_ready": (
            models_ok
            and nodes_ok
        ),

        "models_ok": models_ok,
        "nodes_ok": nodes_ok,

        "models": models,

        "h3_nodes": h3_nodes,

        "object_info_error": object_info_error,
    }


# =========================================================
# Upload image
# =========================================================

def upload_image(image_bytes, filename="input.png"):

    ensure_comfyui()

    files = {
        "image": (
            filename,
            io.BytesIO(image_bytes),
            "image/png",
        )
    }

    data = {
        "overwrite": "true",
    }

    r = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    result = r.json()

    return result["name"]


# =========================================================
# Queue workflow
# =========================================================

def queue_prompt(workflow):

    ensure_comfyui()

    payload = {
        "prompt": workflow,
    }

    r = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    result = r.json()

    if "error" in result:
        raise RuntimeError(
            f"ComfyUI prompt error: {result}"
        )

    return result["prompt_id"]


# =========================================================
# Wait for workflow
# =========================================================

def wait_for_result(prompt_id, timeout=1800):

    start_time = time.time()

    while True:

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Workflow timeout: {prompt_id}"
            )

        try:

            r = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:

                history = r.json()

                if prompt_id in history:

                    result = history[prompt_id]

                    status = result.get(
                        "status",
                        {}
                    )

                    status_str = status.get(
                        "status_str"
                    )

                    if status_str == "error":
                        raise RuntimeError(
                            f"ComfyUI execution error: "
                            f"{result}"
                        )

                    if status.get(
                        "completed",
                        False
                    ):
                        return result

        except requests.RequestException:
            pass

        time.sleep(POLL_INTERVAL)


# =========================================================
# Download ComfyUI output
# =========================================================

def download_output(
    filename,
    subfolder="",
    folder_type="output",
):

    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    }

    r = requests.get(
        f"{COMFYUI_URL}/view",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    return r.content


# =========================================================
# Extract outputs
# =========================================================

def extract_outputs(history):

    outputs = []

    result_outputs = history.get(
        "outputs",
        {}
    )

    for node_id, node_output in result_outputs.items():

        # -------------------------------------------------
        # Videos
        # -------------------------------------------------

        for video in node_output.get(
            "gifs",
            []
        ):

            outputs.append({
                "type": "video",
                "filename": video.get(
                    "filename"
                ),
                "subfolder": video.get(
                    "subfolder",
                    ""
                ),
                "format": video.get(
                    "format"
                ),
            })

        # -------------------------------------------------
        # Images
        # -------------------------------------------------

        for image in node_output.get(
            "images",
            []
        ):

            outputs.append({
                "type": "image",
                "filename": image.get(
                    "filename"
                ),
                "subfolder": image.get(
                    "subfolder",
                    ""
                ),
                "format": image.get(
                    "type"
                ),
            })

    return outputs


# =========================================================
# Run workflow
# =========================================================

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

    # -----------------------------------------------------
    # Copy workflow
    # -----------------------------------------------------

    workflow = json.loads(
        json.dumps(workflow)
    )

    # -----------------------------------------------------
    # Upload image
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Arbitrary overrides
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Queue
    # -----------------------------------------------------

    prompt_id = queue_prompt(
        workflow
    )

    print(
        f"Queued ComfyUI prompt: "
        f"{prompt_id}"
    )

    # -----------------------------------------------------
    # Wait
    # -----------------------------------------------------

    history = wait_for_result(
        prompt_id
    )

    # -----------------------------------------------------
    # Outputs
    # -----------------------------------------------------

    output_files = extract_outputs(
        history
    )

    results = []

    for output in output_files:

        content = download_output(
            output["filename"],
            output.get("subfolder", ""),
            "output",
        )

        results.append({
            "type": output["type"],
            "filename": output["filename"],
            "data": base64.b64encode(
                content
            ).decode("utf-8"),
        })

    return {
        "prompt_id": prompt_id,
        "outputs": results,
    }


# =========================================================
# Main handler
# =========================================================

def handler(event):

    try:

        input_data = event.get(
            "input",
            {}
        )

        action = input_data.get(
            "action",
            "generate"
        )

        # -------------------------------------------------
        # Health
        # -------------------------------------------------

        if action == "health":

            return health_check()

        # -------------------------------------------------
        # H3 check
        # -------------------------------------------------

        if action == "h3_check":

            return h3_check()

        # -------------------------------------------------
        # Generate
        # -------------------------------------------------

        if action == "generate":

            workflow = input_data.get(
                "workflow"
            )

            if workflow is None:

                raise ValueError(
                    "workflow is required"
                )

            # ---------------------------------------------
            # Image
            # ---------------------------------------------

            image_bytes = None

            image_base64 = input_data.get(
                "image"
            )

            if image_base64:

                # Allow data URI
                if image_base64.startswith(
                    "data:"
                ):

                    image_base64 = (
                        image_base64
                        .split(",", 1)[1]
                    )

                image_bytes = base64.b64decode(
                    image_base64
                )

            # ---------------------------------------------
            # Run
            # ---------------------------------------------

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

        # -------------------------------------------------
        # Unknown action
        # -------------------------------------------------

        raise ValueError(
            f"Unknown action: {action}"
        )

    except Exception as e:

        print(
            f"Handler error: {type(e).__name__}: {e}"
        )

        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }


# =========================================================
# RunPod Serverless
# =========================================================

if __name__ == "__main__":

    print(
        "Starting RunPod Serverless handler..."
    )

    runpod.serverless.start({
        "handler": handler
    })
