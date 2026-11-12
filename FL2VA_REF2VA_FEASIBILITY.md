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

## Verified GPU findings — 2026-11-13

All tests ran on the isolated endpoint `90zcdp1rlohy03`, using the same immutable FL2VA-only image as Production (`:a2146fabb`) on **NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb (CUDA 13.2)**. Production was not touched.

- `h3_check`: **COMPLETED** (`fce7d652-2a31-4a4c-8cc7-fc45251f05b5-u2`).
- Native `MiniMaxH3ReferenceToVideo` + existing FL2VA INT8 model: **COMPLETED**, saved MP4 at **384×384, 22 frames, 8 steps**, job `0863386e-2d66-4df4-a855-3073be3f7483-u2`, `executionTime=26333` ms. It used no dedicated Ref2VA model and no Turbo LoRA.
- Decoded frames 0, 10, 20 show two distinct simplified cartoon characters (blue shirt/light-brown hair on left, green shirt/dark hair on right), with blinking.
- Controlled same-seed comparison with a **color-neutral prompt**: two jobs with reference inputs in normal vs swapped order (`f00d3174-2f2b-447d-80e5-fd3171e2acd3-u1` and `fb6d7798-ec55-4592-8209-9fa72ddd489f-u1`). Both **COMPLETED**. Normal order retained blue-left/green-right; swapped order produced green-left/blue-right. Face tones and hair also followed their source sides. This is direct evidence that both reference inputs affect the output, not merely a color instruction in the prompt.
- Full standard-length smoke test: **COMPLETED**, `864×480`, `124 frames`, `20 steps`, seed `77701234`. Job `64519951-b61c-4bd2-930e-def6d19db6eb-u1`, `executionTime=174529` ms, `delayTime=16614` ms; **MP4 generated successfully**, 327,027 bytes. ComfyUI logs: `Prompt executed in 172.36 seconds`. Samples at frames 0, 60, 120 retain the two characters and their blue/green outfits while heads move.
- Offline unit and integration-contract tests **passed**: [GitHub Actions #43327494149](https://github.com/towtsubac-dev/comfyui-runpod-serverless/actions/runs/43327494149).

### Interpretation and rollout boundary

**Node execution, reference sensitivity, and 124-frame stability: verified on simple cartoon images.**
This does **not** establish photorealistic identity fidelity, face consistency, or reference strength versus the dedicated Ref2VA checkpoint. The source references used in this first set are simplistic flat cartoons; photo-realistic A/B and LoRA tests have not been performed. Keep the dedicated Ref2VA endpoint as baseline and keep Production unchanged until those quality/regression tests pass. A successful FL2VA conditioning path is useful but not proof of an equivalent dedicated Ref2VA model.
