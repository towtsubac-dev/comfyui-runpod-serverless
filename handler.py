import runpod


def handler(job):
    return {
        "status": "ok",
        "message": "ComfyUI Serverless worker is running"
    }


runpod.serverless.start({
    "handler": handler
})
