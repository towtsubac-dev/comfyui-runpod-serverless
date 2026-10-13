import base64
import json
import os
import subprocess
import sys
import time
import uuid

import requests
import runpod


COMFYUI_DIR = "/workspace/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

# ---------------------------------------------------------
# Start ComfyUI
# ---------------------------------------------------------

comfy = subprocess.Popen(
    [
        sys.executable,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
    ],
    cwd=COMFYUI_DIR,
)


# ---------------------------------------------------------
# Wait for ComfyUI
# ---------------------------------------------------------

def wait_for_comfyui(timeout=180):
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=5,
            )

            if r.status_code == 200:
                print("ComfyUI is ready")
                return

        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError("ComfyUI did not start within 180 seconds")


wait_for_comfyui()


# ---------------------------------------------------------
# Upload image to ComfyUI
# ---------------------------------------------------------

def upload_image(image_base64, filename="input.png"):
    image_data = base64.b64decode(image_base64)

    files = {
        "image": (
            filename,
            image_data,
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
        timeout=60,
    )

    r.raise_for_status()

    return r.json()


# ---------------------------------------------------------
# Queue ComfyUI workflow
# ---------------------------------------------------------

def queue_prompt(prompt):
    client_id = str(uuid.uuid4())

    payload = {
        "prompt": prompt,
        "client_id": client_id,
    }

    r = requests.post(
        f"{COMFYUI_URL}/prompt",
        json=payload,
        timeout=60,
    )

    r.raise_for_status()

    return r.json()


# ---------------------------------------------------------
# Wait for generation
# ---------------------------------------------------------

def wait_for_result(prompt_id, timeout=1800):
    start = time.time()

    while time.time() - start < timeout:
        r = requests.get(
            f"{COMFYUI_URL}/history/{prompt_id}",
            timeout=30,
        )

        r.raise_for_status()

        history = r.json()

        if prompt_id in history:
            return history[prompt_id]

        time.sleep(2)

    raise TimeoutError(
        f"ComfyUI generation timed out: {prompt_id}"
    )


# ---------------------------------------------------------
# Read generated file
# ---------------------------------------------------------

def read_output_file(filename, subfolder="", folder_type="output"):
    if folder_type == "output":
        base_dir = os.path.join(COMFYUI_DIR, "output")
    else:
        base_dir = os.path.join(COMFYUI_DIR, folder_type)

    path = os.path.join(base_dir, subfolder, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode("utf-8")


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})

    # -----------------------------------------------------
    # Health check
    # -----------------------------------------------------

    if job_input.get("action") == "health":
        r = requests.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=10,
        )

        return {
            "status": "ok",
            "message": "ComfyUI is running",
            "system_stats": r.json(),
        }

    # -----------------------------------------------------
    # Workflow is supplied by the caller
    # -----------------------------------------------------

    workflow = job_input.get("workflow")

    if not workflow:
        return {
            "status": "error",
            "message": "Missing input.workflow",
        }

    # -----------------------------------------------------
    # Optional image upload
    # -----------------------------------------------------

    image_base64 = job_input.get("image")

    if image_base64:
        filename = job_input.get(
            "filename",
            "input.png",
        )

        upload_result = upload_image(
            image_base64,
            filename,
        )

        print("Uploaded image:", upload_result)

    # -----------------------------------------------------
    # Queue workflow
    # -----------------------------------------------------

    queue_result = queue_prompt(workflow)

    prompt_id = queue_result.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(
            f"ComfyUI did not return prompt_id: {queue_result}"
        )

    print("Prompt ID:", prompt_id)

    # -----------------------------------------------------
    # Wait
    # -----------------------------------------------------

    history = wait_for_result(prompt_id)

    # -----------------------------------------------------
    # Find generated outputs
    # -----------------------------------------------------

    outputs = []

    for node_id, node_output in history.get(
        "outputs",
        {}
    ).items():

        for output_type in (
            "images",
            "gifs",
            "videos",
        ):
            items = node_output.get(
                output_type,
                [],
            )

            for item in items:
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                folder_type = item.get(
                    "type",
                    "output",
                )

                if not filename:
                    continue

                try:
                    data = read_output_file(
                        filename,
                        subfolder,
                        folder_type,
                    )

                    outputs.append({
                        "node_id": node_id,
                        "type": output_type,
                        "filename": filename,
                        "data_base64": data,
                    })

                except Exception as e:
                    print(
                        "Could not read output:",
                        filename,
                        e,
                    )

    return {
        "status": "success",
        "prompt_id": prompt_id,
        "outputs": outputs,
    }


# ---------------------------------------------------------
# Start RunPod Serverless
# ---------------------------------------------------------

runpod.serverless.start({
    "handler": handler
})
