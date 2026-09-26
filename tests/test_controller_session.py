"""Session failures must not unplug a running game's controller or kill the game."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cuphead.control.controller_lease import ControllerLease
from cuphead.control.session import GamepadSession
from cuphead.control.actuator import open_vgamepad_actuator


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GamepadSessionTests(unittest.TestCase):
    def setUp(self):
        self.pad = Mock()
        self.child = Mock()
        self.child.poll.return_value = 0
        self.open = patch("cuphead.control.session.open_vgamepad_actuator", return_value=self.pad).start()
        self.launch = patch("cuphead.control.session.subprocess.Popen", return_value=self.child).start()
        patch("cuphead.control.session.platform.system", return_value="Windows").start()
        self.processes = patch("cuphead.control.session.subprocess.run", return_value=Mock(stdout="")).start()
        self.addCleanup(patch.stopall)

    def test_requires_direct_launch_before_creating_pad(self):
        for command in (None, ["steam.exe"], ["cmd", "/c", "Cuphead.exe"]):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "direct Cuphead.exe"):
                with GamepadSession(command):
                    pass
        self.open.assert_not_called()
        self.launch.assert_not_called()

    def test_rejects_existing_game_before_creating_pad(self):
        self.processes.return_value.stdout = '"Cuphead.exe","123","Console","1","200 K"\n'
        with self.assertRaisesRegex(RuntimeError, "already running"):
            with GamepadSession(["Cuphead.exe"]):
                pass
        self.open.assert_not_called()

    def test_process_check_failure_does_not_create_pad(self):
        self.processes.side_effect = subprocess.CalledProcessError(1, "tasklist")
        with self.assertRaises(subprocess.CalledProcessError):
            with GamepadSession(["Cuphead.exe"]):
                pass
        self.open.assert_not_called()

    def test_launch_failure_closes_pad(self):
        self.launch.side_effect = OSError("missing executable")
        with self.assertRaises(OSError):
            with GamepadSession(["Cuphead.exe"]):
                pass
        self.pad.neutral.assert_called_once()
        self.pad.close.assert_called_once()

    def test_pad_created_before_launch(self):
        self.launch.side_effect = lambda *a, **k: (self.open.assert_called_once(), self.child)[1]
        with GamepadSession(["Cuphead.exe"]):
            pass

    def test_failure_releases_input_but_waits_to_disconnect(self):
        self.child.poll.side_effect = [None, None, None, 0]
        def during_wait(_):
            self.pad.neutral.assert_called()
            self.pad.close.assert_not_called()
        with patch("cuphead.control.session.time.sleep", side_effect=during_wait):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with GamepadSession(["Cuphead.exe"], announce=Mock()):
                    raise RuntimeError("capture failed")
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()

    def test_repeated_interrupt_during_wait_does_not_disconnect(self):
        self.child.poll.side_effect = [None, None, None, 0]
        with patch("cuphead.control.session.time.sleep", side_effect=[KeyboardInterrupt, None]):
            with GamepadSession(["Cuphead.exe"], announce=Mock()):
                pass
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()

    def test_neutral_failure_retries_while_game_alive(self):
        self.pad.neutral.side_effect = [OSError("transient"), None]
        self.child.poll.side_effect = [None, None, 0]
        with patch("cuphead.control.session.time.sleep"):
            with GamepadSession(["Cuphead.exe"], announce=Mock()):
                pass
        self.assertEqual(self.pad.neutral.call_count, 2)
        self.pad.close.assert_called_once()

    def test_close_is_idempotent(self):
        with GamepadSession(["Cuphead.exe"]) as session:
            pass
        session.close()
        self.pad.close.assert_called_once()

    def test_canary_capture_setup_failure_keeps_pad_until_game_exit(self):
        canary = script("latency_canary")
        self.child.poll.side_effect = [None, None, 0]
        with patch("cuphead.perception.capture.open_screen_source", side_effect=RuntimeError("capture failed")), \
             patch("cuphead.control.session.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                canary.run("real", 1, 30, 50, ["Cuphead.exe"])
        self.pad.neutral.assert_called()
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()

    def test_launcher_interrupt_never_terminates_game(self):
        launcher = script("launch_with_vgamepad")
        self.child.wait.side_effect = KeyboardInterrupt
        self.child.poll.side_effect = [None, None, 0]
        with patch.object(sys, "argv", ["launcher", "--", "Cuphead.exe"]), \
             patch("cuphead.control.session.time.sleep"):
            self.assertEqual(launcher.main(), 130)
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()
        self.pad.close.assert_called_once()

    def test_capture_pilot_failure_writes_summary_before_controller_wait(self):
        pilot = script("windows_capture_pilot")
        source = Mock()
        source.read.side_effect = RuntimeError("capture failed")
        source.close.side_effect = OSError("capture cleanup failed")
        self.child.poll.side_effect = [None, None, None, 0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            def sleep(duration):
                if duration == .25:
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary['status'], 'cleanup_error')
                    self.assertEqual(summary['frames'], 0)
                    self.pad.neutral.assert_called()
                    self.pad.close.assert_not_called()
            with patch.object(sys, "argv", ["pilot", "--output", str(output), "--launch", "Cuphead.exe"]), \
                 patch.dict(sys.modules, {"numpy": Mock(), "PIL": SimpleNamespace(Image=Mock())}), \
                 patch.object(pilot.threading, "Thread"), \
                 patch.object(pilot, "game_focused", return_value=True), \
                 patch.object(pilot, "open_screen_source", return_value=source), \
                 patch("cuphead.control.session.time.sleep", side_effect=sleep):
                self.assertEqual(pilot.main(), 1)
        source.close.assert_called_once()
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()

    def run_failing_memory_agent(self, output, *, write_fails=False):
        runner = script("run_memory_agent")
        encoder = SimpleNamespace(spec={}, fingerprint="fixture", output_dim=3, backend_name="fixture")
        verifier = SimpleNamespace(document={"edges": [], "goal": "victory",
                                             "landmarks": [{"label": "victory", "mode": "victory"}]},
                                   references={"victory": "fixture"})
        self.child.poll.side_effect = [None, None, None, 0]
        with patch.object(sys, "argv", ["agent", "--real", "--route", "fixture.json",
                                        "--output", str(output), "--bank", str(output.with_suffix('.bank.json')),
                                        "--launch", "Cuphead.exe"]), \
             patch("cuphead.perception.frozen_jepa.FrozenJEPAEncoder", return_value=encoder), \
             patch("cuphead.perception.landmarks.LandmarkVerifier", return_value=verifier), \
             patch("cuphead.perception.window_capture.open_game_source", return_value=Mock()), \
             patch("cuphead.orchestration.memory_agent.MemoryAgent") as agent, \
             patch("cuphead.control.session.time.sleep"):
            agent.return_value.step.side_effect = RuntimeError("capture failed")
            if write_fails:
                with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        runner.main()
            else:
                self.assertEqual(runner.main(), 1)
                self.assertEqual(json.loads(output.read_text())["status"], "error")

    def test_memory_agent_failure_never_kills_game(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_failing_memory_agent(Path(directory) / "run.json")
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()

    def test_memory_agent_report_failure_still_preserves_controller_until_game_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_failing_memory_agent(Path(directory) / "run.json", write_fails=True)
        self.pad.close.assert_called_once()
        self.child.terminate.assert_not_called()


class ControllerLeaseTests(unittest.TestCase):
    def test_duplicate_owner_rejected_and_lock_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            lease = ControllerLease(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "Another Cuphead"):
                    ControllerLease(path)
            finally:
                lease.close()
            ControllerLease(path).close()

    def test_other_process_cannot_acquire_held_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            lease = ControllerLease(path)
            try:
                command = [sys.executable, "-c",
                           "import sys; sys.path.insert(0, sys.argv[1]); "
                           "from cuphead.control.controller_lease import ControllerLease; "
                           "ControllerLease(sys.argv[2]).close()", str(REPO / "src"), str(path)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Another Cuphead", result.stderr)
            finally:
                lease.close()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_device_creation_failure_releases_lease(self):
        lease = Mock()
        with patch("cuphead.control.controller_lease.ControllerLease", return_value=lease), \
             patch("cuphead.control.actuator._open_vgamepad_actuator", side_effect=RuntimeError("driver")):
            with self.assertRaisesRegex(RuntimeError, "driver"):
                open_vgamepad_actuator()
        lease.close.assert_called_once()

    def test_interrupt_during_device_settle_closes_device_and_lease(self):
        lease, actuator = Mock(), Mock()
        with patch("cuphead.control.controller_lease.ControllerLease", return_value=lease), \
             patch("cuphead.control.actuator._open_vgamepad_actuator", return_value=actuator), \
             patch("cuphead.control.actuator.time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                open_vgamepad_actuator()
        actuator.close.assert_called_once()
        lease.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
