"""Unit tests for scripts/check_learnable.py's log-parsing and
learnability decision, against synthetic log lines (no real training run
needed) reproducing the exact console-logger line format PRHBench's real
Gate 1 runs produced.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import check_learnable as cl  # noqa: E402


def metric_line(step, hidden, observed):
    # Mirrors the real console-logger format, e.g.:
    # "step:1 - actor/entropy_loss:1.285 - ... - val/cumulative_hidden_reward_mean:-50.000 - ... - val/cumulative_observed_reward_mean:-50.000 - ..."
    return (
        f"[36m(TaskRunner pid=123)[0m step:{step} - actor/entropy_loss:1.285 - "
        f"episode/hidden_reward_mean:{hidden:.3f} - "
        f"val/cumulative_hidden_reward_mean:{hidden:.3f} - val/cumulative_hidden_reward_std:0.000 - "
        f"val/cumulative_observed_reward_mean:{observed:.3f} - val/cumulative_observed_reward_std:0.000\n"
    )


class TestExtractRecords(unittest.TestCase):
    def test_extracts_step_hidden_observed(self):
        lines = [metric_line(0, -50.0, -50.0), metric_line(5, -40.0, -35.0)]
        records = list(cl.extract_records(lines))
        self.assertEqual(records, [(0, -50.0, -50.0), (5, -40.0, -35.0)])

    def test_ignores_non_metric_lines(self):
        lines = ["some unrelated log line\n", "Capturing CUDA graph shapes: 10%\n"]
        records = list(cl.extract_records(lines))
        self.assertEqual(records, [])

    def test_tolerates_missing_observed_reward(self):
        line = "step:2 - val/cumulative_hidden_reward_mean:-30.000\n"
        records = list(cl.extract_records([line]))
        self.assertEqual(records, [(2, -30.0, None)])


class TestLoadRecords(unittest.TestCase):
    def test_combines_two_phase_logs_in_order(self):
        # Phase A: fresh start, steps 0 (initial val) and 5 (first update).
        # Phase B: resumed, steps 10..20 (no overlap with phase A here).
        phase_a = [metric_line(0, -50.0, -50.0), metric_line(5, -45.0, -40.0)]
        phase_b = [metric_line(10, -30.0, -28.0), metric_line(20, -10.0, -8.0)]

        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fa:
            fa.writelines(phase_a)
            path_a = fa.name
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fb:
            fb.writelines(phase_b)
            path_b = fb.name
        try:
            records = cl.load_records([path_a, path_b])
            self.assertEqual(records[0], (-50.0, -50.0))
            self.assertEqual(records[5], (-45.0, -40.0))
            self.assertEqual(records[10], (-30.0, -28.0))
            self.assertEqual(records[20], (-10.0, -8.0))
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_first_occurrence_wins_on_overlapping_step(self):
        import tempfile

        # Simulates a step re-validated right after a resume: keep the
        # earlier phase's value, not the later one.
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fa:
            fa.write(metric_line(10, -30.0, -28.0))
            path_a = fa.name
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fb:
            fb.write(metric_line(10, -99.0, -99.0))
            path_b = fb.name
        try:
            records = cl.load_records([path_a, path_b])
            self.assertEqual(records[10], (-30.0, -28.0))
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestMainDecision(unittest.TestCase):
    def _run(self, lines, min_delta=0.0):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            path = f.name
        try:
            records = cl.load_records([path])
            steps = sorted(records)
            hidden_start, _ = records[steps[0]]
            hidden_end, _ = records[steps[-1]]
            return hidden_end - hidden_start
        finally:
            os.unlink(path)

    def test_learnable_case_has_positive_delta(self):
        lines = [metric_line(0, -50.0, -50.0), metric_line(40, 5.0, 3.0)]
        delta = self._run(lines)
        self.assertGreater(delta, 0.0)

    def test_not_learnable_case_has_nonpositive_delta(self):
        lines = [metric_line(0, -50.0, -50.0), metric_line(10, -52.0, -48.0)]
        delta = self._run(lines)
        self.assertLessEqual(delta, 0.0)


if __name__ == "__main__":
    unittest.main()
