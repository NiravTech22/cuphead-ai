"""Frame capture: the synthetic source and duplicate/drop detection."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.perception.capture import (  # noqa: E402
    FrameIntegrityError,
    IntegrityTracker,
    SyntheticFrameSource,
)


class TestSyntheticFrameSource(unittest.TestCase):
    def test_indices_are_monotonic_by_default(self):
        src = SyntheticFrameSource()
        indices = [src.read().index for _ in range(10)]
        self.assertEqual(indices, list(range(10)))

    def test_content_changes_every_frame_by_default(self):
        src = SyntheticFrameSource()
        checksums = {src.read().checksum for _ in range(20)}
        self.assertEqual(len(checksums), 20)

    def test_closed_source_rejects_further_reads(self):
        src = SyntheticFrameSource()
        src.read()
        src.close()
        with self.assertRaises(RuntimeError):
            src.read()

    def test_injected_duplicate_repeats_the_previous_checksum(self):
        src = SyntheticFrameSource(duplicate_at=frozenset({3}))
        frames = [src.read() for _ in range(5)]
        self.assertEqual(frames[3].checksum, frames[2].checksum)
        self.assertEqual(frames[3].index, 3)  # index still advances

    def test_injected_drop_creates_an_index_gap(self):
        src = SyntheticFrameSource(drop_at=frozenset({2}))
        indices = [src.read().index for _ in range(5)]
        self.assertEqual(indices, [0, 1, 3, 4, 5])


class TestIntegrityTracker(unittest.TestCase):
    def test_clean_stream_passes_through_untouched(self):
        tracker = IntegrityTracker(SyntheticFrameSource())
        frames = list(tracker.stream(50))
        self.assertEqual(len(frames), 50)
        self.assertTrue(tracker.report.clean)
        self.assertEqual(tracker.report.frames_seen, 50)

    def test_strict_mode_raises_on_a_duplicate(self):
        tracker = IntegrityTracker(SyntheticFrameSource(duplicate_at=frozenset({5})), strict=True)
        with self.assertRaises(FrameIntegrityError):
            for _ in range(10):
                tracker.read()

    def test_strict_mode_raises_on_a_drop(self):
        tracker = IntegrityTracker(SyntheticFrameSource(drop_at=frozenset({4})), strict=True)
        with self.assertRaises(FrameIntegrityError):
            for _ in range(10):
                tracker.read()

    def test_non_strict_mode_accumulates_a_report_instead_of_raising(self):
        source = SyntheticFrameSource(duplicate_at=frozenset({3}), drop_at=frozenset({6}))
        tracker = IntegrityTracker(source, strict=False)
        for _ in range(10):
            tracker.read()
        self.assertFalse(tracker.report.clean)
        self.assertEqual(tracker.report.duplicates, 1)
        self.assertEqual(tracker.report.drops, 1)

    def test_ten_thousand_clean_frames_report_zero_violations(self):
        """Stand-in for the 10-minute-capture acceptance criterion at 60 fps."""
        tracker = IntegrityTracker(SyntheticFrameSource(fps=60.0), strict=True)
        for _ in range(10_000):
            tracker.read()
        self.assertTrue(tracker.report.clean)
        self.assertEqual(tracker.report.frames_seen, 10_000)

    def test_close_delegates_to_the_wrapped_source(self):
        source = SyntheticFrameSource()
        tracker = IntegrityTracker(source)
        tracker.close()
        with self.assertRaises(RuntimeError):
            source.read()


if __name__ == "__main__":
    unittest.main()
