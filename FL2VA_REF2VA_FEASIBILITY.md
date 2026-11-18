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


## Real-person reference validation — 2026-11-13

The same two previously uploaded portraits were reused without distributing the photographs in this repository:
- Picture A: black blouse, indoor striped backdrop (\`108362.jpg\`)
- Picture B: blue sports jersey, green event backdrop (\`108363.jpg\`)

For both native FL2VA and dedicated Ref2VA, the photos were resized into 384×384 reference crops solely for GPU tests. All tests used the same English, single-woman identity-preservation prompt and seed \`22446688\`, with no additional library LoRA.

### Short 22-frame A/B tests (384×384, 8 steps)

| Model | Reference order | GPU execution | RunPod job |
|---|---|---:|---|
| Existing FL2VA INT8 | black → blue | 42.562 seconds | \`750b77a8-7bdc-43ab-904a-3c4798258aa5-u2\` |
| Existing FL2VA INT8 | blue → black | 28.436 seconds | \`9fd2e5a1-f95f-46ae-b97b-9f6ebf7688a9-u1\` |
| Dedicated Ref2VA W6A8 | black → blue | 40.656 seconds | \`d25b7a2d-42e8-4eb1-acb8-6509bfe68a28-u2\` |
| Dedicated Ref2VA W6A8 | blue → black | 24.545 seconds | \`e37b8e53-cf5f-42b3-97a9-9a87c57e247d-u2\` |

All four jobs **COMPLETED with MP4s**. Frames 0, 10, 20 were inspected. Both checkpoints preserve the subject's visible face/hair/outfit well over this short span. Both favor Picture 1's outfit/background, so an A/B reversal alone cannot prove how much Picture 2 improves facial identity.

### Full 124-frame validation (864×480, 20 steps)

| Model | GPU execution | RunPod job |
|---|---:|---|
| Existing FL2VA INT8 | 184.564 seconds | \`f52bd749-3738-404c-9479-515bd2564875-u2\` |
| Dedicated Ref2VA W6A8 | 172.811 seconds | \`9fbc490d-334a-4b8b-9c84-8d884348d4a6-u2\` |

Both **COMPLETED with MP4** without an observed OOM on 24GB MIG. Frames 0, 60, 120 were extracted/inspected.

Visual observations: the FL2VA output maintained the person's recognizable face and black outfit but introduced greater hand/arm motion and some background reframing than requested. The dedicated Ref2VA output remained closer to the original close-up, with a restrained head tilt and consistent face, wardrobe and scene. No hard numerical identity ranking can be made from one seed or this source pair.

### LoRA and second-reference control

- Bundled official FL2VA Turbo 8-step LoRA with the native Ref2VA graph, strength 1.0, 384×384 / 22 frames / 8 steps, same seed: **COMPLETED**, job \`ca5f355d-ca5e-4055-9b07-79e1de2f5b42-u2\`, GPU 44.706 seconds. Frames 0/10/20 retained recognizable appearance. This proves the bundled Turbo LoRA can be injected, **not** that any user-registered library LoRA is compatible.
- Second reference isolation: kept Picture 1 black, replaced Picture 2 blue with an identical copy of Picture 1, holding prompt, seed and all parameters fixed. **COMPLETED**, job \`7bcc1c88-fbaf-46aa-98a7-4b0ba435632f-u1\`, 22.167 seconds. Mid-frame changed (notably blinking) relative to original black+blue. This supports that the second reference is consumed, but does not quantify identity enhancement.

### Deployment decision

**FL2VA is technically viable for native Ref2VA on the same 24GB Production-class GPU, with zero extra diffusion checkpoints.** It remains an **experimental code branch** and is **not installed on the MiniMax H3 Production Endpoint**. The tested jobs used ComfyUI's generic \`action=generate\` on an isolated clone of the existing Production container; the new \`h3_ref2va_generate\` handler in this branch still needs an isolated container smoke test before Production.

Acceptance before rollout:
1. Build this branch as a uniquely tagged Preview container (preserve the existing Production image for rollback).
2. Smoke-test \`h3_ref2va_check\` / \`h3_ref2va_generate\` on a test Endpoint.
3. Repeat I2V, T2V and Extend regression tests and validate registered R2 library LoRAs, not just bundled Turbo.
4. Keep the dedicated Ref2VA Endpoint as a quality baseline, especially for subtle facial identity and restrained motion.

**Production settings and Vercel production routing were not modified during these real-person tests.**


## MMH3 Realism Slider one-LoRA smoke test — 2026-11-13

**PASS**: LoRA `MMH3-RealismSlider-V1.safetensors` (Civitai version `3421124`), SHA256 `96a7bed24f7aa1351c6066666495ee64046faf130f53224f4e4a44b3b49b038d`, 2,700,080 bytes. The registered Cloudflare R2 artifact matched the original upload exactly.

On the **isolated** test endpoint `90zcdp1rlohy03` (production configuration unchanged), the base checkpoint was `minimax_h3_fl2va_pruned_int8_convrot.safetensors`. The LoRA strength was **0.5**.

1. `h3_t2v_generate` was used only to fetch and verify the registered LoRA via the existing SHA256-checked R2 downloader. Job `0b19af44-8ebe-40e9-bc19-cca1cb269dfc-u1` **COMPLETED**, 51,816 ms. Log: optional LoRA at strength 0.50; normal video output.
2. On the **same worker** (`m3tbcc5dyx2nsu`), a native `MiniMaxH3ReferenceToVideo` workflow loaded the existing FL2VA INT8 model and applied the registered file using `LoraLoaderModelOnly` (strength 0.5; no FL2VA Turbo LoRA). Two synthetic references (`EmptyImage`) were used: this is an **application/compatibility smoke test**, not a person-identity or quality comparison.
3. Combined Ref2VA + FL2VA + RealismSlider job `67484264-ecf9-4e2c-b492-cc543139c28c-u2` **COMPLETED**, `executionTime=28373` ms and `delayTime=138` ms, 384×384, 22 frames, 8 steps. It output MP4 `fl2va_ref2va_lora_realism05_00001_.mp4`.
4. ComfyUI Worker logs for the combined job show **`Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached.`**, plus `Prompt executed in 26.17 seconds`. No LoRA loader exception or VRAM OOM. This confirms the LoRA was **actually attached to FL2VA inside the native Ref2VA graph** rather than merely downloaded.

Test-only endpoint idleTimeout was restored to **5 seconds** and workers.min stays **0**. Vercel's ephemeral Preview signing script was removed after use. Production endpoint `mxx0h93h0h14e8`, Studio Production, and dedicated Ref2VA Production were not modified.

**Conclusion:** For the requested narrow acceptance criterion ("Can one Realism Slider LoRA be applied successfully to FL2VA-only native Ref2VA?"), the answer is **yes**. Quality comparisons, multiple LoRAs and full-length LoRA rendering remain outside this test.


## Real-image + Realism Slider 0.5 124-frame full test — 2026-11-13

**PASS** on independent endpoint `90zcdp1rlohy03`: Existing FL2VA INT8 ConvRot checkpoint, native `MiniMaxH3ReferenceToVideo`, registered Realism Slider V1 LoRA strength **0.5** loaded via `LoraLoaderModelOnly`, two previously supplied real-person references (resized and JPEG-compressed for transfer), `864×480`, `124 frames`, `20 steps`, seed `22446688`, no FL2VA Turbo LoRA. RunPod job `24ba05c2-7a8c-4166-9c5a-9dfddc8d3cb8-u2` **COMPLETED** (`executionTime=164615` ms, `delayTime=18` ms); video `fl2va_ref2va_realism05_full_124_00001_.mp4`. Worker log: **`19995MB Staged. 50 patches attached`**. No OOM or LoRA/model-compatibility errors.

The output was decoded independently in follow-up job `f54a9f96-3a73-4617-ae89-30ec2385e177-u2`, which completed successfully. Frames 0/60/120 showed one person, consistent hair, face composition and blue sports shirt, with a natural head tilt near the end. This confirms basic functionality and stability; reference images were downsampled to 144×144 before upload, so precise identity fidelity is not certified.

**Operational issue discovered:** With `minWorkers=0`, RunPod may restart the container between separate jobs even when queued successively, clearing dynamically downloaded R2 LoRA files. A standalone raw workflow that references the downloaded filename can then fail `Value not in list: lora_name ...`. For a production-ready implementation, the registered LoRA must be downloaded/verified in the **same job** as Ref2VA execution, as implemented in `h3_ref2va_generate` on this branch. The final test used a temporary `minWorkers=1` on the isolated endpoint only; after success, **minWorkers=0, maxWorkers=1, idleTimeout=5 seconds** was restored.

The earlier JPEG transfer attempt failed with `PIL.UnidentifiedImageError`; after re-encoding and checking base64 chunk lengths, the full test succeeded. Neither of those errors was a model/LoRA incompatibility.
