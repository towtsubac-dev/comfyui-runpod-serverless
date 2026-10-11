import runpod
import subprocess
import time
import requests
import os
import uuid
import json


COMFYUI_URL = "http://127.0.0.1:8188"


def start_comfyui():
    print("Starting ComfyUI...")

    process = subprocess.Popen([
        "python3",
        "/workspace/ComfyUI/main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188"
    ])

    for i in range(60):
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)

            if response.status_code == 200:
                print("ComfyUI is ready.")
                return process

        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError("ComfyUI failed to start.")


comfy_process = start_comfyui()


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt")

    if not prompt:
        return {
            "error": "prompt is required"
        }

    client_id = str(uuid.uuid4())

    payload = {
        "prompt": prompt,
        "client_id": client_id
    }

    try:
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        return {
            "status": "submitted",
            "prompt_id": result.get("prompt_id")
        }

    except Exception as e:
        return {
            "error": str(e)
        }


runpod.serverless.start({
    "handler": handler
})
