import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ref2va_handler import (
    REF2VA_MODEL, _decode_image, _download_reference,
    build_ref2va_workflow, generate_ref2va,
)


class Ref2vaPreviewTests(unittest.TestCase):
    def test_two_references_use_flat_comfy_autogrow_keys(self):
        wf = build_ref2va_workflow(
            ["front.png", "side.png"],
            "Keep the identity from <Picture 1> and <Picture 2>.",
            seed=42,
        )
        ref = wf["6"]["inputs"]
        self.assertEqual(wf["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(wf["2"]["inputs"]["unet_name"], REF2VA_MODEL)
        self.assertEqual(ref["ref_images.ref_image_0"], ["20", 0])
        self.assertEqual(ref["ref_images.ref_image_1"], ["21", 0])
        self.assertNotIn("ref_images", ref)
        self.assertEqual(wf["14"]["inputs"]["latent_image"], ["6", 1])
        self.assertEqual(wf["12"]["inputs"]["conditioning"], ["6", 0])
        self.assertEqual(wf["11"]["inputs"]["steps"], 20)
        self.assertEqual(wf["11"]["inputs"]["model"], ["8", 0])
        self.assertNotIn("9", wf)  # FL2VA Turbo must not be applied.
        self.assertNotIn("7", wf)  # Latent comes from the Ref2VA node.

    def test_three_references(self):
        wf = build_ref2va_workflow(["a.png", "b.png", "c.png"], "Scene", seed=0)
        self.assertEqual(wf["6"]["inputs"]["ref_images.ref_image_2"], ["22", 0])
        self.assertEqual(wf["22"]["inputs"]["image"], "c.png")

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            build_ref2va_workflow(["one.png"], "Scene")
        with self.assertRaises(ValueError):
            build_ref2va_workflow(["a.png", "b.png"], "Scene", width=800, height=500)
        with self.assertRaises(ValueError):
            build_ref2va_workflow(["a.png", "b.png"], "Scene", length=120)
        with self.assertRaises(ValueError):
            build_ref2va_workflow(["a.png", "b.png"], "Scene", ref_image_size="oversized")

    def test_base64_reference_normalizes_to_png(self):
        image = Image.new("RGB", (256, 256), "red")
        source = io.BytesIO()
        image.save(source, "PNG")
        encoded = base64.b64encode(source.getvalue()).decode("ascii")
        actual = _decode_image({"image": "data:image/png;base64," + encoded})
        with Image.open(io.BytesIO(actual)) as im:
            self.assertEqual(im.size, (256, 256))
            self.assertEqual(im.format, "PNG")

    def test_rejects_local_network_url_without_request(self):
        for url in ("http://example.org/a.png", "https://localhost/x.png",
                    "https://127.0.0.1/x.png"):
            with self.assertRaises(ValueError):
                _download_reference(url)

    def test_generate_passes_workflow_and_returns_mp4(self):
        image = Image.new("RGB", (256, 256), "green")
        buf = io.BytesIO()
        image.save(buf, "PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        queued = []
        def queue(workflow):
            queued.append(workflow)
            return "test-prompt"
        with tempfile.TemporaryDirectory() as temp:
            result = generate_ref2va(
                {"reference_images": [{"image": encoded}, {"image": encoded}],
                 "prompt": "Keep the same person in <Picture 1> and <Picture 2>.",
                 "seed": 123},
                upload_image=lambda data, name: name,
                queue_prompt=queue,
                wait_for_result=lambda prompt_id, timeout: {"outputs": {}},
                extract_outputs=lambda history: [{"type": "videos", "filename": "test.mp4"}],
                download_output=lambda filename, subfolder, file_type: b"fake-mp4",
                comfy_dir=temp,
            )
        self.assertEqual(result["mode"], "ref2va")
        self.assertEqual(result["reference_image_count"], 2)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["13"]["inputs"]["noise_seed"], 123)
        self.assertEqual(base64.b64decode(result["outputs"][0]["data"]), b"fake-mp4")

    def test_ordered_registered_loras_are_connected_without_turbo(self):
        selected = [
            {"lora_name": "library_first.safetensors", "strength": 0.8},
            {"lora_name": "library_second.safetensors", "strength": 0.35},
            {"lora_name": "library_third.safetensors", "strength": 1.1},
            {"lora_name": "library_fourth.safetensors", "strength": 0.5},
            {"lora_name": "library_fifth.safetensors", "strength": 0.75},
        ]
        wf = build_ref2va_workflow(["front.png", "side.png"], "Scene", seed=123,
                                    optional_loras=selected)
        self.assertEqual(wf["30"]["inputs"]["model"], ["2", 0])
        self.assertEqual(wf["31"]["inputs"]["model"], ["30", 0])
        self.assertEqual(wf["34"]["inputs"]["model"], ["33", 0])
        self.assertEqual(wf["8"]["inputs"]["model"], ["34", 0])
        self.assertEqual(wf["31"]["inputs"]["strength_model"], 0.35)
        self.assertEqual(wf["34"]["inputs"]["lora_name"], "library_fifth.safetensors")
        self.assertNotIn("9", wf)  # Never silently activate FL2VA Turbo.

    def test_lora_limit_and_invalid_strength(self):
        selected = [{"lora_name": f"library_{i}.safetensors", "strength": 1}
                    for i in range(6)]
        with self.assertRaisesRegex(ValueError, "5"):
            build_ref2va_workflow(["a.png", "b.png"], "Scene", optional_loras=selected)
        with self.assertRaisesRegex(ValueError, "strength"):
            build_ref2va_workflow(["a.png", "b.png"], "Scene",
                                   optional_loras=[{"lora_name": "x", "strength": 1.8}])

    def test_generate_resolves_registered_lora_and_applies_to_model(self):
        im = Image.new("RGB", (256, 256), "pink")
        blob = io.BytesIO()
        im.save(blob, "PNG")
        encoded = base64.b64encode(blob.getvalue()).decode("ascii")
        queued = []
        resolver_called = []
        def resolve(item, prompt):
            resolver_called.append(item["loras"])
            return prompt + " soft smile", [
                {"lora_name": "library_example.safetensors", "strength": 0.8}
            ], []
        with tempfile.TemporaryDirectory() as temp:
            result = generate_ref2va(
                {"reference_images": [{"image": encoded}, {"image": encoded}],
                 "prompt": "Scene", "loras": [{"name": "library:test", "strength": 0.8}],
                 "seed": 123},
                upload_image=lambda data, name: name,
                queue_prompt=lambda wf: queued.append(wf) or "prompt-1",
                wait_for_result=lambda pid, timeout: {"outputs": {}},
                extract_outputs=lambda record: [{"type": "videos", "filename": "result.mp4"}],
                download_output=lambda filename, subfolder, file_type: b"mp4",
                comfy_dir=temp, resolve_loras=resolve,
            )
        self.assertEqual(len(resolver_called), 1)
        self.assertEqual(result["lora_count"], 1)
        self.assertEqual(queued[0]["30"]["inputs"]["lora_name"], "library_example.safetensors")
        self.assertEqual(queued[0]["8"]["inputs"]["model"], ["30", 0])

    def test_lora_requires_trusted_resolver(self):
        with self.assertRaisesRegex(ValueError, "resolver"):
            generate_ref2va({"reference_images": [{}, {}], "prompt": "Test",
                              "loras": [{"name": "foo"}]},
                             None, None, None, None, None, "/tmp")


if __name__ == "__main__":
    unittest.main()
