
import os
import sys
import time
import subprocess
import requests
import runpod

COMFYUI_DIR = "/workspace/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

# Start the ComfyUI API server
comfy_process = subprocess.Popen(
    [
        sys.executable,
        os.path.join(COMFYUI_DIR, "main.py"),
        "--listen", "127.0.0.1",
        "--port", "8188",
    ],
    cwd=COMFYUI_DIR,
)

def wait_for_comfyui():
    for _ in range(180):
        if comfy_process.poll() is not None:
            raise RuntimeError("ComfyUI exited during startup")

        try:
            response = requests.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=3,
            )
            if response.ok:
                return
        except requests.RequestException:
            pass

        time.sleep(2)

    raise RuntimeError("ComfyUI startup timed out")

def handler(job):
    try:
        response = requests.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=10,
        )
        response.raise_for_status()

        return {
            "status": "ok",
            "message": "ComfyUI API is running",
            "comfyui": response.json(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

wait_for_comfyui()

runpod.serverless.start({
    "handler": handler
})
