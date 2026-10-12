import subprocess
import sys
import time
import requests
import runpod

COMFYUI_DIR = "/workspace/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

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

for i in range(90):
    try:
        r = requests.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=2,
        )

        if r.status_code == 200:
            break

    except Exception:
        pass

    time.sleep(2)
else:
    raise RuntimeError("ComfyUI did not start within 180 seconds")


def handler(job):
    r = requests.get(
        f"{COMFYUI_URL}/system_stats",
        timeout=10,
    )

    return {
        "status": "ok",
        "message": "ComfyUI is running",
        "system_stats": r.json(),
    }


runpod.serverless.start({
    "handler": handler
})
