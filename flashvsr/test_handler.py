import unittest
from unittest.mock import patch
import flashvsr_handler as h


class FlashVSRHandlerTests(unittest.TestCase):
    URL = ("https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/"
           "bucket/video.mp4?X-Amz-Signature=example&X-Amz-Date=20261119T081500Z&X-Amz-Expires=900")

    def test_signed_r2_only(self):
        self.assertEqual(h._signed_r2_url(self.URL), self.URL)
        rejected = (
            "http://127.0.0.1/private",
            "https://localhost/file",
            "https://x.r2.cloudflarestorage.com.evil.org/path?X-Amz-Signature=x",
            "https://x.r2.cloudflarestorage.com/object.mp4",
            "https://x.r2.cloudflarestorage.com/object.mp4?X-Amz-Signature=x&X-Amz-Expires=900",
            "file:///etc/passwd",
            "https://x.r2.cloudflarestorage.com/object.mp4?X-Amz-Signature=x&X-Amz-Expires=7200&X-Amz-Date=20261119T081500Z",
        )
        for value in rejected:
            with self.subTest(url=value), self.assertRaises(ValueError):
                h._signed_r2_url(value)

    def test_geometry_benchmark_640x384(self):
        out = h._geometry({
            "width": 640, "height": 384, "duration": 10.0, "frames": 240, "fps": 24,
        })
        self.assertEqual((out["target_width"], out["target_height"]), (1280, 768))
        self.assertFalse(out["padding_required"])

    def test_geometry_minimax_864x480(self):
        out = h._geometry({
            "width": 864, "height": 480, "duration": 5.0, "frames": 120, "fps": 24,
        })
        self.assertEqual((out["canvas_width"], out["canvas_height"]), (1792, 1024))
        self.assertTrue(out["padding_required"])
        self.assertEqual((out["target_width"], out["target_height"]), (1728, 960))

    def test_budget_bounds(self):
        meta = {"width": 640, "height": 384, "duration": 13.0, "frames": 312, "fps": 24}
        with self.assertRaises(ValueError):
            h._geometry(meta)
        meta.update(duration=10.0, frames=301)
        with self.assertRaises(ValueError):
            h._geometry(meta)

    def test_turbo_forces_triton_and_time_budget(self):
        cmd = h._command("input.mp4", "output.mp4", {
            "canvas_width": 1280, "canvas_height": 768,
        })
        self.assertIn("triton_block_sparse", cmd)
        self.assertIn("turbo", cmd)
        self.assertIn("auto", cmd)
        self.assertEqual(h.INFERENCE_TIMEOUT_SECONDS, 300)

    def test_reject_unknown_action(self):
        with self.assertRaises(ValueError):
            h.handle({"input": {"action": "delete"}})


if __name__ == "__main__":
    unittest.main()
