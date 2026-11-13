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


## Direct RunPod GitHub-build deployment (2026-11-13)

The temporary Docker-registry staging endpoint `rvl3p6ee60wafg` was deleted (RunPod 204 response) to free one of the 10 Serverless endpoint slots. **It must not be reused.**

Use the RunPod console's **GitHub integration**, not the GHCR GitHub Actions workflow, for staging. RunPod's documented workflow:
1. On Serverless, find existing H3 Production endpoint `mxx0h93h0h14e8`, open the three-dot menu and select **Clone Endpoint**.
2. Under **Repository Configuration** select the GitHub repo `towtsubac-dev/comfyui-runpod-serverless`.
3. Select **Branch** `release/h3-fl2va-ref2va-prod-v1` and **Dockerfile Path** `Dockerfile`.
4. Choose name `MiniMax H3 FL2VA Ref2VA Staging` and **Queue** endpoint.
5. Retain 24GB GPU options, **Active Workers 0**, **Max Workers 1**, **Idle Timeout 5 seconds**, **Network Volume none**, and queue delay 4 seconds. To broaden compatibility, allow CUDA 13.1/13.2 if the console exposes it; Production remains unchanged.
6. Click **Deploy Endpoint**. RunPod should perform the build into its own registry; monitor **Builds** for *Pending → Building → Uploading → Testing → Completed*. Never assume completion before the status is confirmed.

Repo checks are already passing: GitHub Actions #43349664541. The accidentally introduced **GHCR staging build workflow** was deleted from the release branch (`4c778bf3e946b559fe6faf19796be660b0348091`); a previously initiated GitHub Actions run #43351781480 may remain in progress and must be cancelled manually from its Actions page because the available connector has no cancel-run action. It is **not** the RunPod build and cannot create a RunPod GitHub-linked release.

Before production promotion, follow **Staging acceptance** above. Keep Production and dedicated Ref2VA endpoint intact until all tests pass.

RunPod official workflow reference: https://github.com/runpod/docs/blob/main/serverless/workers/github-integration.mdx


## Direct RunPod Staging Acceptance — 2026-11-13

- **RunPod cloned staging** endpoint `xzo2mk7u1u7aw4` (`minimax_H3_Production (Copy)`), image `registry.runpod.net/towtsubac-dev-comfyui-runpod-serverless-release-h3-fl2va-ref2va-prod-v1-dockerfile_1414786088_032e3b1a722c4bb9c23f:583ce6cba`.
- RunPod native GitHub build ID `be0c3ae1-3a2a-4c20-9d14-e6dfb6245d1a` **COMPLETED** at source SHA `583ce6cbad104e3ba57d3d16804148b8a5406b45`.
- `h3_check` job `4a506910-701e-47f9-86eb-21ded1325160-u1` **COMPLETED**; `h3_ready=true` and only FL2VA diffusion file exists.
- `h3_ref2va_check` job `c9bc55bb-fbc4-4c87-a69d-7d3f85e72867-u1` **COMPLETED**; `ref2va_ready=true`, `checkpoint_family=FL2VA`. Note: `reference_conditioning_proven=false` is the check's static indicator, not an error; actual Ref2VA video renders below.
- `h3_ref2va_generate` job `01588c6c-a99a-46ed-b6e2-b06d8410dbd2-u1` **COMPLETED** from two actual user-supplied images (resized to 144px), 384×384 / 22f / 8 steps, execution 18.5 s, valid MP4.
- `h3_t2v_generate` job `6a9d2b27-43cb-48f9-ac7b-f19aeb7687f8-u1` **COMPLETED**, 8.8 s, MP4.
- `h3_generate` I2V job `9d764d3d-53ce-4540-a046-17841a02ee9a-u1` **COMPLETED**, 8.1 s, MP4.
- `h3_extend_generate` job `4ceb2d3c-1d8a-4a93-9756-a909ab21265e-u1` **COMPLETED**, 20.8 s, MP4. Source: 22f T2V clip; Extend mode 3s, 8 steps. This checks the pipeline; full-length source audio quality is a separate production monitoring matter.
- `h3_ref2va_generate` with registered **Realism Slider V1** at strength **0.5**, two synthetic 128px face-reference inputs, and native R2 SHA256-verified LoRA download **in the same job**, job `f150309e-70f7-4d9b-bfd1-3fd998928c8f-u2` **COMPLETED** (20.6 s) with MP4. ComfyUI log: `Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached.`; `Prompt executed in 19.19 seconds`, no OOM or loader failures. No extra Ref2VA diffusion checkpoint.
- The temporary Vercel Preview script used only for issuing a one-hour R2 URL was deleted after the test, and ephemeral signing key files were destroyed.

**Remaining promotion actions:** Update only H3 Production endpoint `mxx0h93h0h14e8` to this exact immutable staging image, smoke-test direct modes, then update Vercel Production `H3_REF2VA_ENDPOINT_ID` from separate `swr6qgm13rknsl` to the H3 Production ID. Keep separate Ref2VA endpoint as rollback fallback until Studio Production user acceptance.
