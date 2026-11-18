import contextlib
import io
import unittest
from unittest.mock import patch

import memory_diagnostics


class MemoryDiagnosticsTests(unittest.TestCase):
    def _snapshot(self, *, current=3 * 1024 ** 3, oom=0, killed=0):
        return {
            "current": current,
            "limit": 29 * 1024 ** 3,
            "highwater": current,
            "anon": current // 2,
            "file": current // 2,
            "oom": oom,
            "oom_kill": killed,
        }

    def test_non_generation_does_not_start_monitor(self):
        calls = []

        @memory_diagnostics.monitor_generation_memory
        def handler(event):
            calls.append(event)
            return {"status": "ok"}

        with patch.object(memory_diagnostics, "_snapshot", side_effect=AssertionError("unexpected read")):
            result = handler({"input": {"action": "health"}})
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(len(calls), 1)

    def test_generation_returns_original_payload_and_logs_end(self):
        sample = self._snapshot()

        @memory_diagnostics.monitor_generation_memory
        def handler(event):
            return {"outputs": [1, 2], "status": "ok"}

        captured = io.StringIO()
        with patch.object(memory_diagnostics, "_snapshot", return_value=sample), contextlib.redirect_stdout(captured):
            result = handler({"input": {"action": "h3_generate"}})
        self.assertEqual(result, {"outputs": [1, 2], "status": "ok"})
        log = captured.getvalue()
        self.assertIn("action=h3_generate phase=start", log)
        self.assertIn("action=h3_generate phase=end", log)
        self.assertIn("limit_gib=29.00", log)
        self.assertIn("oom_kill_delta=0", log)

    def test_handler_exception_is_not_swallowed(self):
        sample = self._snapshot()

        @memory_diagnostics.monitor_generation_memory
        def handler(event):
            raise ValueError("original error")

        with patch.object(memory_diagnostics, "_snapshot", return_value=sample), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "original error"):
                handler({"input": {"action": "h3_t2v_generate"}})

    def test_missing_cgroup_does_not_change_result(self):
        @memory_diagnostics.monitor_generation_memory
        def handler(event):
            return 42

        with patch.object(memory_diagnostics, "_snapshot", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(handler({"input": {"action": "h3_extend_generate"}}), 42)

    def test_oom_deltas_are_job_relative(self):
        first = self._snapshot(oom=7, killed=1)
        last = self._snapshot(oom=9, killed=2)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            memory_diagnostics._log("h3_generate", "end", last, last["current"], 0, first)
        log = captured.getvalue()
        self.assertIn("oom_delta=2", log)
        self.assertIn("oom_kill_delta=1", log)


if __name__ == "__main__":
    unittest.main()
