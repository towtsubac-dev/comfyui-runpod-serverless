# MiniMax H3 Ref2VA Preview (multi-image identity)

**Isolation:** Branch `feature/h3-ref2va-multiref-preview` is based on the
existing v11 `fix/h3-extend-preserve-source-av` code. Do **not** merge it into
`main` or switch the Production endpoint to this image until tested.

## Build on RunPod (new Preview endpoint only)

- GitHub repository: `towtsubac-dev/comfyui-runpod-serverless`
- Git branch: `feature/h3-ref2va-multiref-preview`
- Dockerfile: `Dockerfile.ref2va` (**not** `Dockerfile`)
- Network Volume: None
- GPU: 24 GB supported class, initially PRO 6000 Blackwell MIG 1g.24GB or RTX 4090
- Active/min workers: 0
- Idle timeout: 5 seconds; max workers 1
- Queue delay: 4 seconds; Pause idle workers when supported
- Container disk: confirm capacity for the model, image and caches
- Request timeout: allow enough for native 20-step Ref2VA (start at 1800 seconds if available)

The image bundles only the **Ref2VA** diffusion model
(`minimax_h3_ref2va_pruned_w6a8.safetensors`, ~16GB), shared NVFP4 text encoder,
video VAE, and audio VAE. It intentionally omits the FL2VA diffusion model
and FL2VA Turbo LoRA. Therefore **I2V/T2V/Extend will not run inside this
preview-only image**. Production continues handling these actions unchanged.

## Verification requests

1. Submit `{"input":{"action":"h3_ref2va_check"}}` to the **Preview** endpoint.
   `ref2va_ready` must be `true` before GPU generation.
2. Replace the two placeholder signed R2 URLs in
   `tests/ref2va_two_reference_request.json`. Use two views of the same
   consenting adult person, e.g., front view and three-quarter view.
3. Submit that JSON to the Preview endpoint. Check the MP4 output in
   `output.outputs[0].data`, base64 encoded as in existing MiniMax H3 responses.
4. Confirm the ComfyUI logs show `MiniMaxH3ReferenceToVideo`, the Ref2VA
   model, and nonempty references. Save execution time and VRAM observations.
5. Repeat with three references before considering Vercel Preview integration.

For human-identity stability, the prompt should explicitly identify
`<Picture 1>`, `<Picture 2>`, and `<Picture 3>` when present.
These are **references, not guaranteed first frames**. Native conditioning may
still exhibit identity drift.

## API

Action: `h3_ref2va_generate`

- `reference_images`: exactly 2–3 entries initially, ordered, each either
  `{"image_url": "<signed HTTPS URL>"}` or `{"image": "<base64 or data URI>"}`
- `prompt`: required string
- `width`/`height`: default 864 × 480, multiples of 32
- `length`: default 124 frames; 17n+5
- `steps`: default 20 (native base schedule, no Turbo)
- `ref_image_size`: `match` default; `max` is more expensive in memory/time
- `seed`: optional integer
- Additional LoRAs: rejected during initial testing

The node's API inputs **must be flat and zero-based**:
`ref_images.ref_image_0`, `ref_images.ref_image_1`,
`ref_images.ref_image_2`; using a nested `ref_images` object silently
loses conditioning in some ComfyUI versions.

Unit checks run in GitHub Actions without a GPU:
`.github/workflows/h3-ref2va-preview-checks.yml`.

## References

- Official ComfyUI Ref2VA built-in node:
  https://github.com/Comfy-Org/docs/blob/main/built-in-nodes/MiniMaxH3ReferenceToVideo.mdx
- Official H3 workflows:
  https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
- Dotted-key API format confirmation:
  https://github.com/Comfy-Org/ComfyUI/issues/15667
