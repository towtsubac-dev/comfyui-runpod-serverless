# FlashVSR+ Turbo — separate RunPod test (2026-11-19)

**This is not a change to Production MiniMax H3, ErosMax, or SeedVR2.**
GitHub branch: `test/flashvsr-plus-turbo-20261119`
Dockerfile: `Dockerfile.flashvsr-turbo-preview`

- Official implementation: [OpenImagingLab/FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus) at `b515ffc6dba57fdc9ad84c66cadd87d05f27590a`
- Model weights: `ruchangyao/FlashVSR_plus`, fixed revision `46b79520946ac67dec94d1566490690953c7debc`
- Only Turbo / TinyDecoder, no Base, V1 or Wan VAE models
- PyTorch 2.7.1, CUDA 12.8, FlashVSR native Triton
- Model weights included in Docker image, not downloaded per Worker
- Source audio retained; exactly 2× output dimensions using pad -> inference -> crop
- Input and output transferred by signed R2 object URLs (no R2 secrets on Worker)
- Preview accepts up to 12 seconds, 300 frames, 30fps and 1024px source edge
- Inference terminates after **300 seconds** without retry; source download, container startup, MP4 encoding and upload are extra

## Serverless Github-linked build
Create a **new GitHub-built Endpoint** from this repository and branch.
Do not reuse the SeedVR2 endpoint or overwrite the MiniMax H3 endpoint.

Recommended RunPod Endpoint settings:
- GPU: NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb where available
- Serverless GPU pool: AMPERE_24 with allowed compute type MIG 24gb, one GPU
- Active/min Workers 0, max Workers 1, idleTimeout 5s, queueDelay 4s
- Container Disk 50GB, timeout 600000ms (10min), Network Volume none
- Dockerfile Path: `Dockerfile.flashvsr-turbo-preview`

### Health test
```json
{"input":{"action":"health"}}
```
Expected: `status:ready`, `model_ready:true`, `attn_installed:true`, `cuda_available:true`, GPU `...MIG 1g.24gb`.

### Video job (after R2 signed GET + PUT prepared by authenticated Studio API)
```json
{"input":{"action":"upscale","input_video_url":"<R2 signed GET>","output_upload_url":"<R2 signed PUT>"}}
```

First benchmark is **640x384, 10s, 24fps**, exact output 1280x768. Review
`[FLASHVSR] completed inference_s=...` and `execution_seconds`.
Do not claim performance until an actual MIG 24GB job succeeds. No automated
real-video job is submitted on build completion.

### Notes
The official CLI's optimized attention backend is explicitly
`--attn-kernel triton_block_sparse`; it must not silently fall back to SDPA.
No audio is sent back as JSON. The worker directly uploads MP4 to R2.
Production integration and legacy SeedVR2 cleanup are separate later tasks.
