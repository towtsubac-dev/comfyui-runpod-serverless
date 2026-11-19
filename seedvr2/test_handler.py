import unittest
from unittest.mock import patch, Mock
import seedvr2_handler as worker


class HandlerTests(unittest.TestCase):
    SAMPLE = ("https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/"
              "bucket/file.mp4?X-Amz-Signature=abc&X-Amz-Expires=900")

    def test_allow_presigned_r2_url(self):
        self.assertEqual(worker._r2_url(self.SAMPLE), self.SAMPLE)

    def test_reject_ssrf_and_unsigned(self):
        for url in [
            "http://127.0.0.1/private",
            "https://internal.example/file.mp4",
            "https://example.r2.cloudflarestorage.com.evil.org/file?X-Amz-Signature=x",
            "https://foo.r2.cloudflarestorage.com/file",
            "file:///etc/passwd",
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                worker._r2_url(url)

    def test_health(self):
        with patch.object(worker.Path, "is_file", return_value=True):
            result = worker.handle({"input": {"action": "health"}})
        self.assertTrue(result["model_ready"])
        self.assertEqual(result["status"], "ready")

    def test_reject_unknown_action(self):
        with self.assertRaises(ValueError):
            worker.handle({"input": {"action": "delete"}})

    def test_command_has_q4_staged_batch(self):
        cmd = worker._command("/tmp/in.mp4", "/tmp/out.mp4", 960, "/tmp/ram.jsonl")
        self.assertIn("seedvr2_ema_3b-Q4_K_M.gguf", cmd)
        self.assertIn("staged", cmd)
        self.assertIn("960", cmd)
        self.assertIn("sdpa", cmd)


if __name__ == "__main__":
    unittest.main()
