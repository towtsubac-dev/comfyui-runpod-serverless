# ---------------------------------------------------------
# Download H3 I2V diffusion model
# ---------------------------------------------------------

from huggingface_hub import hf_hub_download

H3_REPO = "Comfy-Org/MiniMax-H3"
H3_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"

diffusion_dir = os.path.join(
    COMFYUI_DIR,
    "models",
    "diffusion_models",
)

os.makedirs(diffusion_dir, exist_ok=True)

diffusion_model = os.path.join(
    diffusion_dir,
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
)

if not os.path.exists(diffusion_model):
    print("Downloading H3 diffusion model...")
    hf_hub_download(
        repo_id=H3_REPO,
        filename="diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        revision=H3_REVISION,
        local_dir=COMFYUI_DIR,
    )
    print("H3 diffusion model downloaded.")
else:
    print("H3 diffusion model already exists.")
