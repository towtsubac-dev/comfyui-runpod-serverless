"""FL2VA-only experimental Ref2VA: static validation, not visual proof."""
import ast
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

class FL2VARef2VAContract(unittest.TestCase):
    def test_single_weight_and_legacy_modes(self):
        docker=(ROOT/"Dockerfile").read_text()
        handler=(ROOT/"handler.py").read_text()
        probe=(ROOT/"ref2va_handler.py").read_text()
        self.assertIn("minimax_h3_fl2va_pruned_int8_convrot.safetensors",docker)
        self.assertNotIn("minimax_h3_ref2va_pruned_w6a8.safetensors",docker)
        self.assertNotIn("minimax_h3_ref2va_pruned_int8_convrot.safetensors",docker)
        self.assertIn('REF2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"',probe)
        self.assertIn("COPY ref2va_handler.py /workspace/ref2va_handler.py",docker)
        self.assertIn('"reference_conditioning_proven": False',probe)
        for action in ('h3_generate','h3_t2v_generate','h3_extend_generate',
                       'h3_ref2va_check','h3_ref2va_generate'):
            self.assertIn('"' + action + '"',handler)
        self.assertIn("resolve_loras=resolve_optional_loras",handler)
    def test_models_are_shared(self):
        d=(ROOT/"Dockerfile").read_text()
        for filename in ("minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                         "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                         "minimax_h3_video_vae_int8_convrot.safetensors",
                         "minimax_h3_audio_vae_fp32.safetensors"):
            self.assertEqual(d.count("filename='"+filename+"'"),1)
    def test_python_parses(self):
        for p in ["handler.py","ref2va_handler.py"]:
            ast.parse((ROOT/p).read_text())
if __name__=="__main__":
    unittest.main()
