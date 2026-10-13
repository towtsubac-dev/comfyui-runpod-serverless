
import os
import sys
import time
import uuid
import json
import base64
import logging
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlencode

import requests
import runpod


# =========================================================
# Configuration
# =========================================================

COMFYUI_DIR = os.environ.get(
    "COMFYUI_DIR", "/workspace/ComfyUI"
)

COMFYUI_HOST = os.environ.get(
    "COMFYUI_HOST", "127.0.0.1"
)

COMFYUI_PORT = int(
    os.environ.get("COMFYUI_PORT", "8188")
)

COMFYUI_URL = (
    f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
)

START_COMFYUI = (
    os.environ.get("START_COMFYUI", "true").lower()
    in ("1", "true", "yes")
)

STARTUP_TIMEOUT = int(
    os.environ.get("COMFYUI_STARTUP_TIMEOUT", "600")
)

GENERATION_TIMEOUT = int(
    os.environ.get("GENERATION_TIMEOUT", "1800")
)

POLL_INTERVAL = float(
    os.environ.get("POLL_INTERVAL", "2")
)

OUTPUT_DIR = Path(
    os.environ.get(
        "COMFYUI_OUTPUT_DIR",
        os.path.join(COMFYUI_DIR, "output")
    )
)

INPUT_DIR = Path(
    os.environ.get(
        "COMFYUI_INPUT_DIR",
        os.path.join(COMFYUI_DIR, "input")
    )
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("minimax-h3-handler")

session = requests.Session()

comfy_process = None
comfy_lock = threading.Lock()


# =========================================================
# HTTP helpers
# =========================================================

def comfy_request(
    method,
    path,
    *,
    timeout=60,
    **kwargs
):
    url = f"{COMFYUI_URL}{path}"

    response = session.request(
        method,
        url,
        timeout=timeout,
        **kwargs
    )

    response.raise_for_status()
    return response


def wait_for_comfyui(timeout=STARTUP_TIMEOUT):
    """Wait until the ComfyUI API responds."""

    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        try:
            response = session.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=5
            )

            if response.ok:
                logger.info("ComfyUI API is ready")
                return True

            last_error = (
                f"HTTP {response.status_code}"
            )

        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(2)

    raise TimeoutError(
        "ComfyUI did not become ready within "
        f"{timeout} seconds. Last error: {last_error}"
    )


def start_comfyui():
    """Start ComfyUI when it is not already responding."""

    global comfy_process

    with comfy_lock:
        try:
            response = session.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=3
            )

            if response.ok:
                logger.info("ComfyUI is already running")
                return

        except requests.RequestException:
            pass

        if not START_COMFYUI:
            raise RuntimeError(
                "ComfyUI is unavailable and "
                "START_COMFYUI is disabled"
            )

        main_py = os.path.join(
            COMFYUI_DIR, "main.py"
        )

        if not os.path.isfile(main_py):
            raise FileNotFoundError(
                f"ComfyUI main.py not found: {main_py}"
            )

        command = [
            sys.executable,
            main_py,
            "--listen",
            COMFYUI_HOST,
            "--port",
            str(COMFYUI_PORT),
        ]

        logger.info(
            "Starting ComfyUI: %s",
            " ".join(command)
        )

        comfy_process = subprocess.Popen(
            command,
            cwd=COMFYUI_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy(),
        )

    wait_for_comfyui()


def ensure_comfyui():
    """Ensure ComfyUI is available before handling a job."""

    try:
        response = session.get(
            f"{COMFYUI_URL}/system_stats",
            timeout=5
        )

        if response.ok:
            return

    except requests.RequestException:
        pass

    start_comfyui()


# =========================================================
# Input validation
# =========================================================

def get_job_input(job):
    if not isinstance(job, dict):
        raise ValueError("Job must be a JSON object")

    data = job.get("input", {})

    if not isinstance(data, dict):
        raise ValueError("'input' must be a JSON object")

    return data


def decode_image(image_value):
    """
    Accept either:
      - base64-encoded image data
      - a data:image/...;base64,... string
    """

    if not isinstance(image_value, str) or not image_value:
        raise ValueError(
            "Provide 'image' as a base64 string"
        )

    if image_value.startswith("data:"):
        if "," not in image_value:
            raise ValueError("Invalid image data URL")

        header, image_value = image_value.split(",", 1)

        if ";base64" not in header:
            raise ValueError(
                "Image data URL must use base64 encoding"
            )

    try:
        image_bytes = base64.b64decode(
            image_value,
            validate=True
        )
    except Exception as exc:
        raise ValueError(
            "Image is not valid base64"
        ) from exc

    if not image_bytes:
        raise ValueError("Image is empty")

    return image_bytes


def upload_image(image_bytes, filename=None):
    """Upload the input image to ComfyUI."""

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename:
        filename = os.path.basename(filename)
    else:
        filename = f"input_{uuid.uuid4().hex}.png"

    if not filename:
        filename = f"input_{uuid.uuid4().hex}.png"

    response = comfy_request(
        "POST",
        "/upload/image",
        files={
            "image": (
                filename,
                image_bytes,
                "application/octet-stream"
            )
        },
        data={
            "type": "input",
            "overwrite": "true",
        },
        timeout=120,
    )

    result = response.json()

    uploaded_name = result.get("name")

    if not uploaded_name:
        raise RuntimeError(
            f"ComfyUI image upload failed: {result}"
        )

    subfolder = result.get("subfolder", "")

    logger.info(
        "Uploaded input image: %s",
        uploaded_name
    )

    return {
        "name": uploaded_name,
        "subfolder": subfolder,
        "type": result.get("type", "input"),
    }


# =========================================================
# Workflow helpers
# =========================================================

def load_workflow(data):
    """
    The workflow must be ComfyUI API-format JSON,
    exported using Save (API Format).
    """

    workflow = data.get("workflow")

    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "'workflow' is not valid JSON"
            ) from exc

    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(
            "Missing 'workflow'. Supply a ComfyUI "
            "API-format workflow in the request."
        )

    # API workflows contain node IDs as top-level keys.
    # A normal UI-format workflow is not accepted here.
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            raise ValueError(
                f"Invalid workflow node: {node_id}"
            )

        if "class_type" not in node:
            raise ValueError(
                f"Node {node_id} has no class_type. "
                "Export the workflow using Save (API Format)."
            )

    return workflow


def set_node_input(workflow, node_id, input_name, value):
    """Set one input on a specific workflow node."""

    node_id = str(node_id)

    if node_id not in workflow:
        raise ValueError(
            f"Workflow node {node_id} does not exist"
        )

    node = workflow[node_id]

    if not isinstance(node.get("inputs"), dict):
        raise ValueError(
            f"Node {node_id} has no inputs object"
        )

    if input_name not in node["inputs"]:
        raise ValueError(
            f"Input '{input_name}' not found on "
            f"node {node_id}"
        )

    node["inputs"][input_name] = value


def apply_overrides(workflow, overrides):
    """
    Apply explicit node-input overrides.

    Example:
    {
      "12": {
        "inputs": {
          "text": "A cinematic video"
        }
      }
    }
    """

    if not overrides:
        return

    if not isinstance(overrides, dict):
        raise ValueError("'overrides' must be an object")

    for node_id, specification in overrides.items():
        if not isinstance(specification, dict):
            raise ValueError(
                f"Invalid override for node {node_id}"
            )

        inputs = specification.get("inputs", {})

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Invalid inputs for node {node_id}"
            )

        for input_name, value in inputs.items():
            set_node_input(
                workflow,
                node_id,
                input_name,
                value
            )


def prepare_workflow(data, uploaded_image):
    """
    Inject image and prompt into an API workflow.

    Supply image_node_id and image_input_name to identify
    the image loader node.

    Supply prompt_node_id and prompt_input_name to identify
    the positive-prompt node.
    """

    workflow = load_workflow(data)

    image_node_id = data.get("image_node_id")
    prompt_node_id = data.get("prompt_node_id")

    image_input_name = data.get(
        "image_input_name", "image"
    )

    prompt_input_name = data.get(
        "prompt_input_name", "text"
    )

    if image_node_id is not None:
        set_node_input(
            workflow,
            image_node_id,
            image_input_name,
            uploaded_image["name"]
        )

    if prompt_node_id is not None:
        prompt = data.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "'prompt' must be a non-empty string"
            )

        set_node_input(
            workflow,
            prompt_node_id,
            prompt_input_name,
            prompt.strip()
        )

    apply_overrides(
        workflow,
        data.get("overrides")
    )

    return workflow


# =========================================================
# Generation
# =========================================================

def queue_workflow(workflow):
    """Submit the workflow to the ComfyUI API."""

    client_id = str(uuid.uuid4())

    response = comfy_request(
        "POST",
        "/prompt",
        json={
            "prompt": workflow,
            "client_id": client_id,
        },
        timeout=120,
    )

    result = response.json()

    prompt_id = result.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(
            f"ComfyUI did not return prompt_id: {result}"
        )

    logger.info(
        "Queued ComfyUI workflow: %s",
        prompt_id
    )

    return prompt_id


def wait_for_generation(prompt_id, timeout=None):
    """Poll ComfyUI history until the job completes."""

    if timeout is None:
        timeout = GENERATION_TIMEOUT

    deadline = time.monotonic() + timeout
    history_path = f"/history/{prompt_id}"

    while time.monotonic() < deadline:
        response = comfy_request(
            "GET",
            history_path,
            timeout=30,
        )

        history = response.json()
        record = history.get(prompt_id)

        if record:
            status = record.get("status", {})
            messages = status.get("messages", [])

            if status.get("status_str") == "error":
                raise RuntimeError(
                    "ComfyUI generation failed: "
                    + json.dumps(
                        messages,
                        ensure_ascii=False
                    )
                )

            if status.get("completed", False):
                return record

            # Some ComfyUI versions expose execution
            # errors through status messages.
            for message in messages:
                if (
                    isinstance(message, list)
                    and message
                    and message[0] == "execution_error"
                ):
                    raise RuntimeError(
                        "ComfyUI execution error: "
                        + json.dumps(
                            message,
                            ensure_ascii=False
                        )
                    )

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Generation timed out after {timeout} seconds"
    )


# =========================================================
# Output extraction
# =========================================================

def collect_output_files(history_record):
    """Collect generated files from ComfyUI history."""

    outputs = history_record.get("outputs", {})
    files = []

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue

        for output_type, entries in node_output.items():
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                filename = entry.get("filename")

                if not filename:
                    continue

                files.append({
                    "node_id": str(node_id),
                    "output_type": output_type,
                    "filename": filename,
                    "subfolder": entry.get(
                        "subfolder", ""
                    ),
                    "type": entry.get("type", "output"),
                })

    return files


def download_output_file(file_info):
    """Download one generated file from ComfyUI."""

    params = urlencode({
        "filename": file_info["filename"],
        "subfolder": file_info.get("subfolder", ""),
        "type": file_info.get("type", "output"),
    })

    response = comfy_request(
        "GET",
        f"/view?{params}",
        timeout=180,
    )

    return response.content


def is_video_file(file_info):
    filename = file_info.get("filename", "").lower()

    return filename.endswith((
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
        ".gif",
    ))


# =========================================================
# RunPod handler
# =========================================================

def handler(job):
    """
    Request example:

    {
      "input": {
        "image": "<base64-image>",
        "prompt": "A cinematic shot...",
        "workflow": { "...": "ComfyUI API workflow" },
        "image_node_id": "10",
        "prompt_node_id": "12",
        "timeout": 1800
      }
    }

    Node IDs and input names depend on your workflow.
    """

    try:
        data = get_job_input(job)

        # Health check: no workflow or image required.
        if data.get("action") == "health":
            ensure_comfyui()

            stats = comfy_request(
                "GET",
                "/system_stats",
                timeout=15,
            ).json()

            return {
                "status": "ok",
                "message": "ComfyUI is running",
                "system_stats": stats,
            }

        ensure_comfyui()

        image_bytes = decode_image(
            data.get("image")
        )

        uploaded_image = upload_image(
            image_bytes,
            data.get("image_filename")
        )

        workflow = prepare_workflow(
            data,
            uploaded_image
        )

        prompt_id = queue_workflow(workflow)

        history_record = wait_for_generation(
            prompt_id,
            timeout=int(
                data.get(
                    "timeout",
                    GENERATION_TIMEOUT
                )
            ),
        )

        output_files = collect_output_files(
            history_record
        )

        if not output_files:
            raise RuntimeError(
                "Workflow completed but produced no "
                "downloadable output files"
            )

        videos = [
            item for item in output_files
            if is_video_file(item)
        ]

        # Prefer video outputs. If none exist, return
        # available outputs so workflow problems are visible.
        selected_files = videos if videos else output_files

        return_files = []
        total_bytes = 0

        max_output_bytes = int(
            data.get(
                "max_output_bytes",
                100 * 1024 * 1024
            )
        )

        for item in selected_files:
            content = download_output_file(item)

            if total_bytes + len(content) > max_output_bytes:
                raise RuntimeError(
                    "Generated output exceeds max_output_bytes. "
                    "Reduce output size or increase the limit."
                )

            total_bytes += len(content)

            return_files.append({
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "output_type": item["output_type"],
                "mime_type": (
                    "video/mp4"
                    if item["filename"].lower().endswith(".mp4")
                    else "application/octet-stream"
                ),
                "data": base64.b64encode(
                    content
                ).decode("ascii"),
            })

        return {
            "status": "ok",
            "prompt_id": prompt_id,
            "output_count": len(return_files),
            "output": return_files,
        }

    except Exception as exc:
        logger.exception("Handler execution failed")

        return {
            "status": "error",
            "error": str(exc),
        }


# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    logger.info("Starting RunPod Serverless handler")

    try:
        start_comfyui()
    except Exception:
        logger.exception(
            "Initial ComfyUI startup failed. "
            "The handler will retry when a job arrives."
        )

    runpod.serverless.start({
        "handler": handler
    })
