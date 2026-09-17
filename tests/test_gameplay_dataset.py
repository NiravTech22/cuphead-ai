"""Data provenance, temporal alignment, coverage and fail-closed diversity checks."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cuphead.evaluation.dataset_gate import (
    inspect_dataset, label_segment, select_samples, variance_metrics, finish_report,
)
from cuphead.events.schema import EventTrace
from cuphead.memory.replay import ReplayWriter, ReplayReader

HAS_PIXELS = importlib.util.find_spec("PIL") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def make_segment(root, name="segment", kind="attempt", n=100, outcome="DEATH", mode="real"):
    from PIL import Image

    writer = ReplayWriter(root, run_id=name, boss="forest_follies", fps=30)
    rows = []
    for i in range(n):
        writer.log_action(i, {"stick_x": 0, "stick_y": 0, "a": False})
        image = f"frames/{i:08d}.png"
        Image.new("RGB", (16, 16), (i % 256, 0, 0)).save(root / name / image)
        rows.append({"frame": i, "capture_t": 100 + i / 30,
                     "action_t": 100 + i / 30 + .001, "checksum": i, "image": image,
                     "raw_input": {"backend": "evdev", "keys": {}, "axes": {}}})
    (root / name / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    writer.finalize(events=EventTrace(boss="forest_follies", events=[], fps=30),
                    outcome=outcome, frame_count=n,
                    extras={"schema": "dataset_segment_v1", "mode": mode,
                            "segment_type": kind, "phase": "whole_attempt", "resolution": [16, 16],
                            "labels_verified": True, "full_attempt": kind == "attempt"})
    return root / name


class SamplingTests(unittest.TestCase):
    def test_sample_spans_types_outcomes_and_time_without_duplicates(self):
        segments = [dict(segment=str(i), segment_type=kind, boss="forest", phase="all",
                         outcome=outcome, frame_count=1000)
                    for i, (kind, outcome) in enumerate([
                        ("attempt", "DEATH"), ("attempt", "KNOCKOUT"),
                        ("menu", "NAVIGATION"), ("idle", "IDLE")])]
        selected = select_samples(segments, 240)
        self.assertEqual(len(selected), 240)
        self.assertEqual(len({(s["segment"], i) for s, i in selected}), 240)
        for segment in segments:
            indices = [i for s, i in selected if s is segment]
            self.assertEqual(len(indices), 60)
            self.assertEqual((min(indices), max(indices)), (0, 999))

    def test_under_200_is_not_configurable(self):
        with self.assertRaises(ValueError):
            select_samples([], 8)

    def test_no_probe_never_allows_training(self):
        result = finish_report({"coverage_passed": True, "blocking_reasons": []}, None)
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["data_collection_done"])

    def test_collapsed_probe_blocks_good_coverage(self):
        result = finish_report({"coverage_passed": True, "blocking_reasons": []}, {"passed": False})
        self.assertFalse(result["training_allowed"])

    def test_good_variance_does_not_override_missing_coverage(self):
        result = finish_report({"coverage_passed": False, "blocking_reasons": ["missing attempts"]}, {"passed": True})
        self.assertFalse(result["training_allowed"])


@unittest.skipUnless(HAS_PIXELS, "Pillow required for actual frame validation")
class DatasetIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def audit(self):
        return inspect_dataset(self.root, width=16, height=16, fps=30, boss="forest_follies")[0]

    def test_counts_real_frames_and_actual_capture_time(self):
        for name, kind, outcome in [("death", "attempt", "DEATH"), ("menu", "menu", "NAVIGATION"), ("idle", "idle", "IDLE")]:
            make_segment(self.root, name, kind, outcome=outcome)
        report = self.audit()
        self.assertEqual(report["total_frame_count"], 300)
        self.assertEqual(report["frames_by_segment_type"], {"attempt": 100, "menu": 100, "idle": 100})
        self.assertAlmostEqual(report["captured_wall_clock_seconds"], 10)
        self.assertFalse(report["coverage_passed"])
        self.assertFalse(report["training_allowed"])

    def test_synthetic_and_legacy_cannot_pad_target(self):
        make_segment(self.root, mode="synthetic")
        report = self.audit()
        self.assertEqual(report["total_frame_count"], 0)
        self.assertEqual(len(report["excluded"]), 1)

    def test_missing_pixels_fail_closed(self):
        directory = make_segment(self.root)
        (directory / "frames/00000005.png").unlink()
        report = self.audit()
        self.assertEqual(report["total_frame_count"], 0)
        self.assertEqual(len(report["invalid_segments"]), 1)

    def test_invalid_timing_and_missing_raw_input_fail_closed(self):
        directory = make_segment(self.root)
        path = directory / "frames.jsonl"
        original = path.read_text()
        for change in (lambda r: r.update(capture_t=99), lambda r: r.update(raw_input=None),
                       lambda r: r.update(action_t=999)):
            rows = [json.loads(line) for line in original.splitlines()]
            change(rows[5])
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(self.audit()["total_frame_count"], 0)

    def test_wrong_training_resolution_fails(self):
        make_segment(self.root)
        report, _ = inspect_dataset(self.root, width=32, height=32, fps=30, boss="forest_follies")
        self.assertEqual(report["total_frame_count"], 0)

    def test_label_requires_real_ending_for_full_attempt(self):
        directory = make_segment(self.root)
        with self.assertRaises(ValueError):
            label_segment(directory, outcome="INCOMPLETE", full_attempt=True)
        label_segment(directory, outcome="KNOCKOUT", full_attempt=True, notes="Witnessed results screen")
        replay = ReplayReader(self.root, directory.name).read()
        self.assertEqual(replay.meta.outcome, "KNOCKOUT")
        self.assertTrue(replay.meta.extras["full_attempt"])

    def test_idle_input_cannot_be_mislabeled_as_no_input(self):
        directory = make_segment(self.root, kind="idle", outcome="IDLE")
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["raw_input"]["keys"]["304"] = 1
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(self.audit()["total_frame_count"], 0)

    def test_keyboard_snapshots_are_validated_and_idle_controls_rejected(self):
        from cuphead.control.keyboard_input import KEYBOARD_KEYS, KEYBOARD_BINDINGS
        from cuphead.control.action_space import Action
        directory = make_segment(self.root, kind="idle", outcome="IDLE")
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["extras"].update(input_source="keyboard", keyboard_bindings=KEYBOARD_BINDINGS)
        meta_path.write_text(json.dumps(meta))
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["raw_input"] = {"backend": "x11_keyboard", "keys": dict.fromkeys(KEYBOARD_KEYS, 0),
                                "axes": {}, "focused": True}
        def save():
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (directory / "actions.jsonl").write_text("".join(
            json.dumps({"frame": i, "action": Action().to_buttons()}) + "\n" for i in range(len(rows))))
        save()
        self.assertEqual(self.audit()["total_frame_count"], 100)
        # Even keys omitted by the pruned action schema invalidate idle labels.
        for key in ("v", "Tab", "Return", "Escape"):
            rows[0]["raw_input"]["keys"][key] = 1
            save()
            self.assertEqual(self.audit()["total_frame_count"], 0)
            rows[0]["raw_input"]["keys"][key] = 0
        rows[0]["raw_input"]["keys"]["z"] = 1
        save()
        self.assertIn("does not match", self.audit()["invalid_segments"][0]["reason"])
        rows[0]["raw_input"]["keys"]["z"] = 0
        rows[0]["raw_input"]["focused"] = False
        save()
        self.assertEqual(self.audit()["total_frame_count"], 0)
        rows[0]["raw_input"]["focused"] = True
        del rows[0]["raw_input"]["keys"]["z"]
        save()
        self.assertEqual(self.audit()["total_frame_count"], 0)

    def test_repeated_idle_pixels_are_retained(self):
        directory = make_segment(self.root, kind="idle", outcome="IDLE")
        path = directory / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        image = (directory / "frames/00000000.png").read_bytes()
        for row in rows:
            row["checksum"] = 1
            (directory / row["image"]).write_bytes(image)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = self.audit()
        self.assertEqual(report["total_frame_count"], 100)
        self.assertEqual(report["segments"][0]["repeated_images"], 99)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class VarianceTests(unittest.TestCase):
    def test_collapsed_and_diverse_embeddings(self):
        import numpy as np
        constant = np.zeros((240, 384))
        constant[:, 0] = 1
        self.assertEqual(variance_metrics(constant)["mean_output_variance"], 0)
        self.assertTrue(variance_metrics(constant)["still_collapsed_or_low_diversity"])
        diverse = np.random.default_rng(17).normal(size=(240, 384))
        diverse /= np.linalg.norm(diverse, axis=1, keepdims=True)
        self.assertFalse(variance_metrics(diverse)["still_collapsed_or_low_diversity"])
        with self.assertRaises(ValueError):
            variance_metrics(diverse * 100)


class RecorderTests(unittest.TestCase):
    @staticmethod
    def module():
        spec = importlib.util.spec_from_file_location("record_session", REPO / "scripts/record_session.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_synthetic_records_aligned_timestamps_and_cannot_overwrite(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(mode="synthetic", root=Path(tmp), run_id="demo", boss="forest_follies",
                          loadout={}, game_build="test", fps=30, outcome="INCOMPLETE", frames=5, max_seconds=None)
            directory = module.record(**kwargs)
            replay = ReplayReader(Path(tmp), "demo").read()
            times = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame"] for r in times], list(range(5)))
            self.assertEqual(replay.meta.frame_count, 5)
            self.assertFalse(replay.meta.extras["labels_verified"])
            with self.assertRaises(FileExistsError):
                module.record(**kwargs)

    @unittest.skipUnless(HAS_PIXELS, "Pillow required")
    def test_real_path_saves_pixels_and_retains_static_frames(self):
        from cuphead.perception.capture import Frame
        from cuphead.control.action_space import Action
        from PIL import Image

        class Source:
            def __init__(self):
                self.index = 0
            def read(self):
                import time
                frame = Frame(self.index, time.perf_counter(), bytes(16 * 16 * 3), 1, 16, 16)
                self.index += 1
                return frame
            def close(self):
                pass
        class Input:
            def poll(self):
                return Action()
            def snapshot(self):
                return {"backend": "evdev", "keys": {}, "axes": {}}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cuphead.perception.window_capture.open_game_source", Source), patch(
                "cuphead.control.human_input.open_gamepad_source", return_value=Input()):
            directory = self.module().record(mode="real", root=Path(tmp), run_id="idle", boss="forest",
                loadout={}, game_build="test", fps=30, outcome="IDLE", frames=3, max_seconds=None,
                segment_type="idle", width=16, height=16, input_source="gamepad")
            with Image.open(directory / "frames/00000002.png") as image:
                self.assertEqual(image.size, (16, 16))
            self.assertEqual(ReplayReader(Path(tmp), "idle").read().meta.extras["repeated_images"], 2)

    @unittest.skipUnless(HAS_PIXELS, "Pillow required")
    def test_keyboard_recording_focus_boundary_preserves_alignment_and_schema(self):
        import time
        from cuphead.perception.capture import Frame
        from cuphead.control.keyboard_input import KEYBOARD_KEYS, KeyboardFocusLost, keyboard_action
        class Source:
            window_id = 42
            def __init__(self):
                self.index = 0
            def read(self):
                frame = Frame(self.index, time.perf_counter(), bytes(16 * 16 * 3), 1, 16, 16)
                self.index += 1
                return frame
            def close(self):
                pass
        class Input:
            def __init__(self):
                self.count = 0
                self.closed = False
                self.waited = False
                self.keys = dict.fromkeys(KEYBOARD_KEYS, 0)
                self.keys.update(Right=1, z=1, x=1)
            def wait_for_focus(self):
                self.waited = True
            def poll(self):
                self.count += 1
                if self.count > 3:
                    raise KeyboardFocusLost("test segment boundary")
                return keyboard_action(self.keys)
            def snapshot(self):
                return {"backend": "x11_keyboard", "keys": dict(self.keys), "axes": {}, "focused": True}
            def close(self):
                self.closed = True
        module = self.module()
        inputs = Input()
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cuphead.perception.window_capture.open_game_source", Source), patch.object(
                module, "X11KeyboardSource", return_value=inputs) as factory:
            directory = module.record(mode="real", root=Path(tmp), run_id="keys", boss="forest_follies",
                loadout={}, game_build="test", fps=30, outcome="INCOMPLETE", frames=10,
                max_seconds=None, width=16, height=16)
            factory.assert_called_once_with(window_id=42)
            self.assertTrue(inputs.waited)
            self.assertTrue(inputs.closed)
            replay = ReplayReader(Path(tmp), "keys").read()
            self.assertEqual(replay.meta.frame_count, 3)
            self.assertEqual(replay.meta.extras["stop_reason"], "focus_lost")
            for row in replay.actions:
                self.assertEqual(row["action"], keyboard_action(inputs.keys).to_buttons())
            rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame"] for r in rows], [0, 1, 2])
            self.assertTrue(all(r["capture_t"] <= r["action_t"] for r in rows))
            label_segment(directory, outcome="INCOMPLETE", notes="test")
            report, _ = inspect_dataset(Path(tmp), width=16, height=16, fps=30, boss="forest_follies")
            self.assertEqual(report["total_frame_count"], 3)
            self.assertEqual(report["invalid_segments"], [])


class PhysicalInputTests(unittest.TestCase):
    def source(self):
        from cuphead.control.human_input import open_gamepad_source
        codes = SimpleNamespace(EV_SYN=0, EV_KEY=1, EV_ABS=3, SYN_DROPPED=3,
                                ABS_X=0, ABS_Y=1, ABS_HAT0X=16, ABS_HAT0Y=17,
                                BTN_GAMEPAD=304, BTN_SOUTH=304, BTN_WEST=307,
                                BTN_EAST=305, BTN_TR=311)
        class Device:
            path = "fixture"
            def __init__(self):
                self.events = []
            def absinfo(self, code):
                return SimpleNamespace(min=-32768, max=32767, value=0)
            def active_keys(self):
                return [307, 315]  # shoot and menu/start held before capture
            def capabilities(self, absinfo=True):
                axes = [(0, self.absinfo(0)), (1, self.absinfo(1))] if absinfo else [0, 1]
                return {1: [304, 307, 315], 3: axes}
            def read_one(self):
                return self.events.pop(0) if self.events else None
            def close(self):
                pass
        device = Device()
        module = SimpleNamespace(ecodes=codes, InputDevice=lambda path: device, list_devices=lambda: ["fixture"])
        with patch.dict(sys.modules, evdev=module):
            source = open_gamepad_source(backend="linux")
        return source, device, codes

    def test_initial_held_menu_state_and_dpad_are_preserved(self):
        source, device, codes = self.source()
        self.assertTrue(source.poll().shoot)
        self.assertEqual(source.snapshot()["keys"]["315"], 1)
        device.events.append(SimpleNamespace(type=codes.EV_ABS, code=codes.ABS_HAT0X, value=-1))
        self.assertEqual(source.poll().move, "left")
        self.assertEqual(source.snapshot()["axes"]["16"], -1)
        device.events.append(SimpleNamespace(type=codes.EV_ABS, code=codes.ABS_HAT0X, value=0))
        self.assertEqual(source.poll().move, "none")

    def test_dropped_input_events_abort_instead_of_reusing_stale_state(self):
        source, device, codes = self.source()
        device.events.append(SimpleNamespace(type=codes.EV_SYN, code=codes.SYN_DROPPED, value=0))
        with self.assertRaisesRegex(RuntimeError, "dropped"):
            source.poll()


if __name__ == "__main__":
    unittest.main()
