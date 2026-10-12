import os
import sys
import time
import subprocess
import requests
import runpod

COMFYUI_DIR = "/workspace/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

comfy_process = None


def start_comfyui():
    global comfy_process

    comfy_process = subprocess.Popen(
        [
            sys.executable,
            os.path.join(COMFYUI_DIR, "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
        ],
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_comfyui(timeout=180):
    start_time = time.time()

    while time.time() - start_time < timeout:

        if comfy_process.poll() is not None:
            output = comfy_process.stdout.read()
            raise RuntimeError(
                "ComfyUI exited during startup:\n" + output
            )

        try:
            response = requests.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=3,
            )

            if response.status_code == 200:
                print("ComfyUI is ready")
                return True

        except requests.RequestException:
            pass

        time.sleep(2)

    raise RuntimeError("ComfyUI startup timed out")


def handler(job):
    print("Received job:", job)

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

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":
    print("Starting ComfyUI...")

    start_comfyui()
    wait_for_comfyui()

    print("Starting RunPod Serverless handler...")

    runpod.serverless.start({
        "handler": handler
    })
