# SeedVR2 3B Q4 — isolated RunPod preview (2026-11-19)

Source branch: `test/seedvr2-3b-q4-upscale-20261119`.
Dockerfile path: `Dockerfile.seedvr2-preview`.

**This preview must never be deployed over MiniMax H3 / ErosMax.**
Its source is the pinned `CunFeng-W/SeedVR2-CLI` commit
`c7086f2de4144789b568ec94605c27b59c5f28e1`.
Model weights are baked into Docker. Model SHA-256 checks run at build time.

## RunPod
- New separate Serverless Endpoint (Queue)
- GPU: one NVIDIA RTX PRO 6000 MIG 1g.24gb
- Active/min workers 0, max workers 1; idle timeout 5s; queue delay 4s
- Container disk 50GB; max job 30 minutes
- Source: this GitHub branch, Dockerfile.seedvr2-preview

**API integration is intentionally NOT yet deployed to Vercel.**
Both input and output must be explicitly signed per object by a trusted backend;
no R2 credentials are sent to the worker.

### Health job
```json
{"input":{"action":"health"}}
```

### Video job
```json
{"input":{"action":"upscale","input_video_url":"<R2 presigned GET>","output_upload_url":"<R2 presigned PUT>"}}
```

Input max 120MiB, 20s, source edges 128–1920px. Initial mode is SeedVR2
3B Q4, batch 5, streaming chunk 33, SDPA, staged offload, target short edge 2x.
The final MP4 remuxes source audio when present, then uploads directly to R2.
Output max 300MiB. User-facing playback/history integration follows E2E tests.

Tests are run during build; GPU inference still needs an actual RunPod job.
