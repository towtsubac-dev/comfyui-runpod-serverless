# FL2VA-only Ref2VA feasibility test (2026-11-13)

This experiment tests whether MiniMax H3's **existing** `minimax_h3_fl2va_pruned_int8_convrot.safetensors` can run the native `MiniMaxH3ReferenceToVideo` conditioning without bundling the separate Ref2VA model.

**Important:** MiniMax's official release treats FL2VA and Ref2VA as different task-specific checkpoint families. A passing node-level or ComfyUI graph-level test is *not* proof of comparable reference identity or character consistency.

## Isolation

- Production endpoint: `mxx0h93h0h14e8` — **never changed**
- Separate RunPod test endpoint: `90zcdp1rlohy03`
- Test endpoint image: **exact same immutable image as Production**, digest suffix `a2146fabb`
- Model weights: FL2VA INT8 ConvRot only, original text encoder, audio/video VAEs and 8-step Turbo LoRA unchanged
- Worker policy: min 0, max 1, idle 5s; only the test endpoint CUDA availability was widened to 13.2/13.1
- Test jobs: `fce7d652-2a31-4a4c-8cc7-fc45251f05b5-u2` (`h3_check`) and `0863386e-2d66-4df4-a855-3073be3f7483-u2` (FL2VA with native Ref2VA graph)

The second job is a short smoke test, not a production-grade quality test: 384x384, 22 frames, 8 steps, seed 33122, two distinct cartoon reference images cropped out of one composite source using ComfyUI `ImageCrop` nodes, with no LoRA. It uses the Production image's existing generic `action=generate` workflow execution, so no custom container build or Production deployment is needed for the smoke test.

## Experimental source

- `handler.py` retains original I2V/T2V/Extend and adds explicit `h3_ref2va_check` / `h3_ref2va_generate`
- `ref2va_handler.py` uses the *existing* FL2VA diffusion checkpoint for `MiniMaxH3ReferenceToVideo`
- `Dockerfile` is unchanged except copying `ref2va_handler.py` into the container
- Offline unit tests must pass before any test build
- **Do not merge or deploy to Production based on offline tests alone**

## Acceptance gates

1. Test endpoint obtains a GPU; `h3_check` succeeds
2. Native two-reference graph validates and outputs MP4 without mismatch/OOM
3. Repeat at standard 124 frames with the same two photographs used for dedicated Ref2VA, seed/settings as close as possible
4. Reverse Picture 1/2; verify the outputs follow those references rather than merely ignoring the second one
5. Test I2V/T2V/Extend regression and optional LoRA before any Production replacement
6. If direct FL2VA fails to preserve references, test a proven compatible lightweight reference patch on a separate branch; do not assume community patch compatibility with pruned INT8 ConvRot

No weights, image routes, Production Endpoint or Vercel Production environment variables are changed by this experiment.
