"""Contract tests with fake native libraries; no driver or display required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuphead.control.action_space import Action
from cuphead.control.actuator import open_vgamepad_actuator
from cuphead.control.timed_input import ControlInput
from cuphead.control.windows_gamepad import WindowsGamepadActuator
from cuphead.perception.capture import open_screen_source
from cuphead.perception.dxcam_capture import DXCamFrameSource


class WindowsControllerTests(unittest.TestCase):
    def setUp(self):
        self.pad = Mock()
        self.buttons = SimpleNamespace(**{
            'XUSB_GAMEPAD_' + name: name
            for name in ['A', 'B', 'X', 'Y', 'RIGHT_SHOULDER', 'START']})
        self.actuator = WindowsGamepadActuator(self.pad, self.buttons)

    def test_combat_y_up_and_lock_on_both_bindings(self):
        action = Action(shoot=True, lock=True, aim='n')
        result = self.actuator.send(action)
        self.assertEqual(result.action, action)
        self.pad.left_joystick_float.assert_called_with(x_value_float=0.0, y_value_float=1.0)
        self.pad.press_button.assert_any_call(button='X')
        self.pad.press_button.assert_any_call(button='RIGHT_SHOULDER')
        self.pad.right_trigger_float.assert_called_with(value_float=1.0)

    def test_complete_states_release_and_close(self):
        self.actuator.send_buttons(ControlInput('menu', a=True, y=True, start=True, stick_y=-1))
        for button in ['A', 'Y', 'START']:
            self.pad.press_button.assert_any_call(button=button)
        self.pad.reset_mock()
        self.actuator.send_buttons(ControlInput('neutral'))
        self.pad.reset.assert_called_once()
        self.pad.press_button.assert_not_called()
        self.pad.right_trigger_float.assert_called_once_with(value_float=0.0)
        self.pad.update.assert_called_once()
        self.actuator.close()
        self.actuator.close()
        with self.assertRaises(RuntimeError):
            self.actuator.send(Action())

    def test_windows_factory(self):
        library = SimpleNamespace(VX360Gamepad=Mock(return_value=self.pad), XUSB_BUTTON=self.buttons)
        with patch.dict(sys.modules, vgamepad=library), patch('platform.system', return_value='Windows'):
            actuator = open_vgamepad_actuator(settle_seconds=0)
            self.assertIsInstance(actuator, WindowsGamepadActuator)
            actuator.close()


class DXCamTests(unittest.TestCase):
    def test_game_source_platform_dispatch(self):
        from cuphead.perception.window_capture import open_game_source
        with patch('platform.system', return_value='Linux'), patch(
                'cuphead.perception.window_capture.X11WindowSource') as x11:
            self.assertIs(open_game_source(), x11.return_value)
        with patch('platform.system', return_value='Windows'), patch(
                'cuphead.perception.capture.open_screen_source') as screen:
            self.assertIs(open_game_source(), screen.return_value)

    def test_missing_dependency_is_actionable(self):
        with patch.dict(sys.modules, dxcam=None):
            with self.assertRaisesRegex(RuntimeError, 'pip install'):
                DXCamFrameSource()

    def test_rgb_region_freshness_indices_and_cleanup(self):
        pixels = SimpleNamespace(shape=(2, 3, 3), tobytes=lambda: bytes(range(18)))
        camera = Mock()
        camera.grab.side_effect = [None, pixels, pixels]
        library = SimpleNamespace(create=Mock(return_value=camera))
        with patch.dict(sys.modules, dxcam=library), patch('platform.system', return_value='Windows'):
            source = open_screen_source(monitor=2, region=dict(left=4, top=5, width=3, height=2))
            library.create.assert_called_once_with(device_idx=0, output_idx=1, output_color='RGB')
            first, second = source.read(), source.read()
            self.assertEqual((first.index, second.index), (0, 1))
            self.assertEqual((first.width, first.height, first.payload), (3, 2, bytes(range(18))))
            camera.grab.assert_called_with(region=(4, 5, 7, 7))
            source.close()
            source.close()
            camera.release.assert_called_once()
            with self.assertRaises(RuntimeError):
                source.read()

    def test_timeout_does_not_emit_a_frame(self):
        camera = Mock()
        camera.grab.return_value = None
        with patch.dict(sys.modules, dxcam=SimpleNamespace(create=Mock(return_value=camera))):
            source = DXCamFrameSource(timeout=0.01)
            with patch('cuphead.perception.dxcam_capture.time.perf_counter', side_effect=[0, 1]):
                with self.assertRaises(TimeoutError):
                    source.read()
            self.assertEqual(source._index, 0)
            source.close()

    def test_invalid_configuration(self):
        for kwargs in [dict(monitor=0), dict(timeout=0), dict(timeout=float('nan')),
                       dict(region=dict(left=-1, top=0, width=1, height=1))]:
            with self.assertRaises(ValueError):
                DXCamFrameSource(**kwargs)
        with self.assertRaises(ValueError):
            open_screen_source(backend='invalid')


if __name__ == '__main__':
    unittest.main()
