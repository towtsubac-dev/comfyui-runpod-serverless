"""Static checks only. Never use this as a substitute for 24GB GPU validation."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class H3AllModesContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile=(ROOT/"Dockerfile").read_text()
        cls.handler=(ROOT/"handler.py").read_text()
        cls.mod=(ROOT/"ref2va_handler.py").read_text()

    def test_both_diffusion_models_bundled(self):
        for model in ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                      "minimax_h3_ref2va_pruned_w6a8.safetensors"):
            self.assertIn(model,self.dockerfile)
        for shared in ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                       "minimax_h3_video_vae_int8_convrot.safetensors",
                       "minimax_h3_audio_vae_fp32.safetensors"):
            self.assertEqual(self.dockerfile.count("filename='"+shared+"'"),1)
        self.assertIn("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",self.dockerfile)

    def test_actions_remain_distinct(self):
        for action in ("h3_generate","h3_t2v_generate","h3_extend_generate",
                       "h3_ref2va_check","h3_ref2va_generate"):
            with self.subTest(action=action):
                self.assertIn('"' + action + '"',self.handler)
        self.assertIn("from ref2va_handler import generate_ref2va",self.handler)
        self.assertIn("resolve_loras=resolve_optional_loras",self.handler)

    def test_no_turbo_lora_in_native_ref2va(self):
        self.assertIn("MiniMaxH3ReferenceToVideo",self.mod)
        self.assertNotIn("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",self.mod)
        self.assertIn("LoraLoaderModelOnly",self.mod)

    def test_dockerfile_copies_both_modules(self):
        self.assertIn("COPY handler.py /workspace/handler.py",self.dockerfile)
        self.assertIn("COPY ref2va_handler.py /workspace/ref2va_handler.py",self.dockerfile)

if __name__=="__main__":
    unittest.main()
