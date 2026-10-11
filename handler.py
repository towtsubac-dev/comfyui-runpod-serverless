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
    return {
        "status": "ok",
        "message": "ComfyUI Serverless worker is running"
    }


runpod.serverless.start({
    "handler": handler
})
