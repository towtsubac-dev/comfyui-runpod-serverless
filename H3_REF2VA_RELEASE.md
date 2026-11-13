# H3 FL2VA-only Ref2VA — production promotion runbook

## Tested source and guardrails

- Based on current Production H3 branch `fix/h3-extend-preserve-source-av`, commit `a2146fabb7e071ae779cb745ca71076474ed4ca3`.
- Only modifies `handler.py` to add **two actions**, copies `ref2va_handler.py`, and adds the new module to the Dockerfile. Existing `h3_generate`, `h3_t2v_generate`, and `h3_extend_generate` stay unchanged.
- Only one diffusion checkpoint is bundled: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`. The separate `minimax_h3_ref2va_pruned_w6a8.safetensors` is **not** bundled.
- New actions: `h3_ref2va_check` and `h3_ref2va_generate`.
- Native Ref2VA graph loads registered LoRAs via the same **per-job** SHA256-verified signed R2 URL mechanism as I2V/T2V. This is important: dynamic files are not guaranteed to survive worker restarts when min workers is zero.
- Offline static unit tests and graph tests: GitHub Actions run `43349664541` **PASS**.
- Live FL2VA-only native Ref2VA + one Realism Slider LoRA (0.5) + 2 real photos (reduced reference resolution) at 864×480, 124 frames, 20 steps on 24GB GPU **PASS**, job `24ba05c2-7a8c-4166-9c5a-9dfddc8d3cb8-u2` (164,615 ms). 50 patches attached, valid MP4; visual 0/60/120-frame spot check acceptable.

## Staging build (do NOT modify Production yet)

From Runpod Serverless, choose **Clone Endpoint** on existing H3 Production endpoint `mxx0h93h0h14e8` (it has an existing GitHub-build source). In Repository Configuration, choose:
- GitHub repository: `towtsubac-dev/comfyui-runpod-serverless`
- Branch: `release/h3-fl2va-ref2va-prod-v1`
- Dockerfile path: `Dockerfile`
- Name: `MiniMax-H3-FL2VA-Ref2VA-Staging` or similar
- workers min 0, max 1, idle 5 seconds, 24GB GPU, network volume none, CUDA >=13.1 as previously tested (CUDA 13.2 is preferable when stock is limited)

Build only this staging endpoint. The H3 Production endpoint and current Studio Production env are unchanged.

### Staging acceptance

1. Build completes. `h3_check` and `h3_ref2va_check` report healthy and the single FL2VA diffusion checkpoint exists.
2. `h3_ref2va_generate` using **two references and the registered Realism Slider at strength 0.5** succeeds in a single job with min workers 0, showing model patches attached.
3. Smoke-test existing I2V / T2V / Extend (one minimal sample each) before Production. Check audio remains intact on Extend.
4. Compare runtime and memory against baseline, confirm no second 20GB model, and preserve the staging image URL/release ID for rollback.

### Production promotion — only after staging acceptance

1. Build/promote the **same release branch** to H3 Production endpoint `mxx0h93h0h14e8`, retaining its old image for rollback:
   `registry.runpod.net/towtsubac-dev-comfyui-runpod-serverless-fix-h3-extend-preserve-source-av-dockerfile_1414786088_032e3b1a722c4bb9c23f:a2146fabb`.
2. Test H3 Production direct `h3_check`, `h3_ref2va_check`, short `h3_ref2va_generate`; ensure existing modes remain functional.
3. Change Vercel Production `H3_REF2VA_ENDPOINT_ID` **to** `mxx0h93h0h14e8`, leave `H3_ENDPOINT_ID` unchanged. Trigger a production deployment with `target=production`. The existing `/video/ref2va` UI and API already use the env variable and do not need redesign.
4. Check Studio Ref2VA with user login, R2 result/archive, LoRA selection, and basic I2V/T2V/Extend functionality.
5. Keep the dedicated Ref2VA endpoint `swr6qgm13rknsl` until the unified Production is verified and rollback is safe; do **not** delete it or remove the rollback image prematurely.

Never silently deploy a branch to Production before a staging build verifies that it starts and serves all modes.


## Actual staging endpoint / build tracking (2026-11-13)

- **Staging Endpoint ID**: `rvl3p6ee60wafg`, name `MiniMax H3 FL2VA Ref2VA Staging`.
- Endpoint created with the **existing Production image** as a safe placeholder:
  `registry.runpod.net/towtsubac-dev-comfyui-runpod-serverless-fix-h3-extend-preserve-source-av-dockerfile_1414786088_032e3b1a722c4bb9c23f:a2146fabb`.
- Staging GPU: 24GB pools `AMPERE_24, ADA_24`, CUDA `13.2/13.1`; workers min=0/max=1/idle=5s; no network volume; Production `mxx0h93h0h14e8` unchanged.
- Isolated GitHub Actions workflow `.github/workflows/h3-fl2va-ref2va-staging-build.yml` created. **Run ID** `43351781480`, trigger SHA `d9ab3ce7549c805fd7ad21a76c15ede7187af13f`. The resulting tag (only if the build succeeds):
  `ghcr.io/towtsubac-dev/comfyui-runpod-serverless:h3-fl2va-ref2va-staging-d9ab3ce75`.
- The workflow is never permitted to overwrite `latest`, `v1`, or Production registry tags.
- The staging endpoint **must not** be switched to the new tag before the workflow reports success and the image is pullable. After that, change **only staging** via `update_endpoint({id:"rvl3p6ee60wafg",body:{image:"ghcr.io/towtsubac-dev/comfyui-runpod-serverless:h3-fl2va-ref2va-staging-d9ab3ce75"}})`.
- After the image is applied: `h3_ref2va_check` and `h3_ref2va_generate` with LoRA in the *same job*, then I2V/T2V/Extend regressions. The current 10-endpoint quota is full (9 old endpoints after removing compatibility test `90zcdp1rlohy03` + this staging endpoint).
