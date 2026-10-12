import runpod

def handler(job):
    return {
        "status": "ok",
        "message": "RunPod worker is running"
    }

runpod.serverless.start({
    "handler": handler
})
