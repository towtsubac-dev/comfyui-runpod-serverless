import runpod

def handler(job):
    print("JOB RECEIVED:", job, flush=True)

    return {
        "status": "ok",
        "message": "Serverless handler is working"
    }

runpod.serverless.start({
    "handler": handler
})
