"""CPU ownership/deadline tests and opt-in real CUDA replay integration tests.

CUPHEAD_TEST_CUDA=1 enables CUDA tests; CUPHEAD_TEST_CUDA_COMPILE=1 also
exercises Inductor compilation (requires a supported compiler/toolchain).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.orchestration.gpu_recorded_agent import RingBufferedGPURecordedAgentEngine


class _HostArray:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        value = self.values[index]
        return _HostArray(value) if isinstance(value, list) else value

    def tolist(self):
        return list(self.values)


class RingOwnershipTests(unittest.TestCase):
    def engine(self):
        engine = object.__new__(RingBufferedGPURecordedAgentEngine)
        engine._owner = threading.get_ident()
        engine._closed = engine._faulted = False
        engine._budget_ms = 33.0
        engine._pending = deque([0])
        slot = Mock()
        slot.done.query.return_value = True
        slot.sequence, slot.frame_id = 4, 90
        slot.submitted_at = 100.0
        slot.host_outputs = (
            _HostArray([[2, 1]]), _HostArray([[0.9, 0.5]]),
            _HostArray([7]), _HostArray([True]),
        )
        for event in (slot.copy_start, slot.graph_start, slot.read_start):
            event.elapsed_time.return_value = 1.0
        engine._slots = [slot]
        return engine, slot

    def test_not_ready_never_reads_or_releases_slot(self):
        engine, slot = self.engine()
        slot.done.query.return_value = False
        slot.host_outputs = None  # Accessing it before completion would fail.
        self.assertIsNone(engine.poll())
        self.assertEqual(engine.in_flight, 1)
        slot.done.synchronize.assert_not_called()

    def test_result_is_owned_and_slot_is_released(self):
        engine, slot = self.engine()
        with patch("cuphead.orchestration.gpu_recorded_agent.time.perf_counter", return_value=100.01):
            result = engine.poll()
        self.assertEqual(result.action_id, 7)
        self.assertEqual((result.sequence, result.frame_id), (4, 90))
        slot.host_outputs[0].values[0][0] = 99
        self.assertEqual(result.neighbor_ids, (2, 1))
        self.assertEqual(engine.in_flight, 0)
        self.assertIsNone(engine.poll())

    def test_late_result_suppresses_action(self):
        engine, _ = self.engine()
        with patch("cuphead.orchestration.gpu_recorded_agent.time.perf_counter", return_value=100.04):
            result = engine.poll()
        self.assertTrue(result.deadline_missed)
        self.assertIsNone(result.action_id)

    def test_invalid_embedding_suppresses_action(self):
        engine, slot = self.engine()
        slot.host_outputs[-1].values[0] = False
        with patch("cuphead.orchestration.gpu_recorded_agent.time.perf_counter", return_value=100.01):
            result = engine.poll()
        self.assertFalse(result.valid_embedding)
        self.assertIsNone(result.action_id)

    def test_wrong_thread_closed_and_faulted_guards(self):
        for field, value in (("_owner", -1), ("_closed", True), ("_faulted", True)):
            engine, _ = self.engine()
            setattr(engine, field, value)
            with self.assertRaises(RuntimeError):
                engine.poll()

    def test_close_drains_even_after_submission_failure(self):
        engine, _ = self.engine()
        engine._faulted = True
        streams = [Mock() for _ in range(3)]
        engine._copy_stream, engine._compute_stream, engine._read_stream = streams
        engine.close()
        engine.close()
        for stream in streams:
            stream.synchronize.assert_called_once()
        self.assertEqual(engine._slots, [])

    def test_static_configuration_guards_without_torch(self):
        for options in ({"slots": 1}, {"budget_ms": 34}, {"temperature": 0}, {"warmup_steps": 1}):
            with self.assertRaises(ValueError):
                RingBufferedGPURecordedAgentEngine(None, None, None, input_shape=(1, 2), **options)
        with self.assertRaises(ValueError):
            RingBufferedGPURecordedAgentEngine(None, None, None, input_shape=(2, 2))


@unittest.skipUnless(os.environ.get("CUPHEAD_TEST_CUDA") == "1", "optional CUDA integration suite")
class CUDAReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.torch = torch

    def make_engine(self, **kwargs):
        torch = self.torch

        class Encoder(torch.nn.Module):
            def forward(self, pixels):
                return pixels.float()

        keys = torch.tensor([[3., 4.], [4., -3.], [-3., -4.]])
        actions = torch.tensor([7, 9, 11], dtype=torch.int64)
        engine = RingBufferedGPURecordedAgentEngine(
            Encoder(), keys, actions, input_shape=(1, 2), input_dtype=torch.float32,
            k=2, compile_model=os.environ.get("CUPHEAD_TEST_CUDA_COMPILE") == "1", **kwargs,
        )
        # External mutation must not alter the engine-owned memory snapshot.
        keys.zero_()
        actions.zero_()
        return engine

    def drain_one(self, engine):
        timeout = time.perf_counter() + 10
        while time.perf_counter() < timeout:
            result = engine.poll()
            if result is not None:
                return result
            time.sleep(0)
        self.fail("CUDA result did not become ready within ten seconds")

    def test_wraparound_backpressure_and_independent_outputs(self):
        torch = self.torch
        with self.make_engine() as engine:
            for cycle in range(10):
                a = torch.tensor([[3., 4.]])
                b = torch.tensor([[4., -3.]])
                self.assertEqual(engine.try_submit(a, frame_id=cycle * 2), cycle * 2)
                a.zero_()  # Staging must already own the DMA source.
                self.assertEqual(engine.try_submit(b, frame_id=cycle * 2 + 1), cycle * 2 + 1)
                self.assertIsNone(engine.try_submit(b, frame_id=-1))
                first, second = self.drain_one(engine), self.drain_one(engine)
                self.assertEqual((first.frame_id, second.frame_id), (cycle * 2, cycle * 2 + 1))
                self.assertEqual((first.neighbor_ids[0], second.neighbor_ids[0]), (0, 1))
                for result, expected in ((first, 7), (second, 9)):
                    self.assertAlmostEqual(result.similarities[0], 1.0, places=5)
                    self.assertTrue(result.valid_embedding)
                    self.assertEqual(result.action_id, None if result.deadline_missed else expected)

    def test_zero_nan_deadline_and_shape_guards(self):
        torch = self.torch
        with self.make_engine(slots=3) as engine:
            for pixels in (torch.zeros(1, 2), torch.full((1, 2), float("nan"))):
                engine.try_submit(pixels, frame_id=0)
                result = self.drain_one(engine)
                self.assertFalse(result.valid_embedding)
                self.assertIsNone(result.action_id)
            engine.try_submit(torch.tensor([[3., 4.]]), frame_id=1, captured_at=time.perf_counter() - 1)
            result = self.drain_one(engine)
            self.assertTrue(result.deadline_missed)
            self.assertIsNone(result.action_id)
            with self.assertRaises(ValueError):
                engine.try_submit(torch.zeros(1, 3), frame_id=0)
            with self.assertRaises(ValueError):
                engine.try_submit(torch.zeros(1, 2, dtype=torch.uint8), frame_id=0)


if __name__ == "__main__":
    unittest.main()
