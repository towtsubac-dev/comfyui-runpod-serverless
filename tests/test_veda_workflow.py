"""Fast, GPU-free regression tests for the Veda workflow patch."""
import ast
import copy
from pathlib import Path
import unittest

SOURCE = Path("/workspace/handler.py")
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("apply_h3_attention", "validate_veda_sparsity", "validate_h3_t2v_resolution")]
namespace = {"VEDA_PREDICTOR": "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
patch = namespace["apply_h3_attention"]


class VedaWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = {
            "1": {"class_type": "UNETLoader", "inputs": {}},
            "10": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0]}},
            "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0]}},
        }

    def test_sage_is_unchanged(self):
        before = copy.deepcopy(self.workflow)
        self.assertEqual(patch(self.workflow, "sage"), before)

    def test_veda_has_every_required_input(self):
        result = patch(self.workflow, "veda")
        veda = result["12"]
        self.assertEqual(veda["class_type"], "VedaSparseAttention")
        self.assertEqual(result["10"]["inputs"]["model"], ["12", 0])
        self.assertEqual(result["11"]["inputs"]["model"], ["12", 0])
        self.assertEqual(veda["inputs"], {
            "model": ["1", 0],
            "predictor": namespace["VEDA_PREDICTOR"],
            "generated_sparsity": "trained",
            "reference_sparsity": "trained",
            "full_attention_layers": "",
            "full_attention_steps": "",
            "verbose": True,
        })

    def test_custom_sparsity(self):
        result = patch(self.workflow, "veda", "90%", "24")
        self.assertEqual(result["12"]["inputs"]["generated_sparsity"], "90%")
        self.assertEqual(result["12"]["inputs"]["reference_sparsity"], "24")

    def test_invalid_sparsity_fails(self):
        for value in ("0", "257", "100%", "-1", "invalid", 32):
            with self.subTest(value=value), self.assertRaises(ValueError):
                patch(copy.deepcopy(self.workflow), "veda", value, "trained")

    def test_veda_aligned_resolution_allowed(self):
        namespace["validate_h3_t2v_resolution"](1024, 512, "veda")

    def test_sage_aligned_resolution_rejected(self):
        with self.assertRaises(ValueError):
            namespace["validate_h3_t2v_resolution"](1024, 512, "sage")

    def test_veda_other_large_resolutions_rejected(self):
        for width, height in ((1152, 448), (1024, 544), (1344, 768)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                namespace["validate_h3_t2v_resolution"](width, height, "veda")

    def test_mismatched_model_paths_fail(self):
        self.workflow["11"]["inputs"]["model"] = ["9", 0]
        with self.assertRaises(RuntimeError):
            patch(self.workflow, "veda")

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            patch(self.workflow, "unknown")


if __name__ == "__main__":
    unittest.main()
