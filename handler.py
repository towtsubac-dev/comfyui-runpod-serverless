import runpod

print("=== HANDLER STARTED ===", flush=True)

def handler(job):
    print("=== JOB RECEIVED ===", flush=True)

    return {
        "status": "ok",
        "message": "Handler is running"
    }

runpod.serverless.start({
    "handler": handler
})
