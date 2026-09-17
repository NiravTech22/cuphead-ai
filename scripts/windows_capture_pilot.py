"""Bounded native gameplay collection with explicit, untrained exploration.

Setup commands are JSON in <output>/command.json: {id, buttons, hold_seconds}.
After visually verifying the level, send {id, mode: 'explore'}; stop with
{id, mode: 'stop'}. This is data collection, not a trained combat policy.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import queue
import threading
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from cuphead.control.actuator import open_vgamepad_actuator
from cuphead.control.timed_input import ControlInput
from cuphead.perception.capture import open_screen_source


def game_focused():
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(user32.GetForegroundWindow(), title, 512)
    return title.value == 'Cuphead'


def exploration_action(rng):
    # Forward-biased coverage policy; no claim of learned avoidance or progress.
    choices = [ControlInput('right_shoot', stick_x=1, x=True),
               ControlInput('jump_right_shoot', stick_x=1, a=True, x=True),
               ControlInput('shoot', x=True),
               ControlInput('jump_shoot', a=True, x=True)]
    return rng.choices(choices, weights=[4, 5, 1, 1])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--setup-seconds', type=float, default=900)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--retry-reference', type=Path,
                        help='visually verified Forest Follies death card; enables automatic retry and exploration')
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or not 0 < args.fps <= 60:
        parser.error('seconds must be in (0,3600], fps in (0,60]')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'frames').mkdir()
    incoming = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            try:
                incoming.put(json.loads(line))
            except ValueError:
                print('Ignoring malformed command JSON', flush=True)

    threading.Thread(target=read_commands, daemon=True).start()
    stdin_command = None
    from PIL import Image
    import numpy as np

    def death_patch(image):
        w, h = image.size
        return np.asarray(image.crop((int(.29*w), int(.12*h), int(.69*w), int(.5*h))).resize((64,64)), dtype=np.float32) / 255

    death_reference = death_patch(Image.open(args.retry_reference).convert('RGB')) if args.retry_reference else None
    death_count = 0
    retries = 0
    retry_until = 0

    actuator = source = None
    rng = random.Random(args.seed)
    count = 0
    started = None
    last_id = None
    active = ControlInput('neutral')
    action_t = time.perf_counter()
    release_at = next_action = 0
    latest_at = 0
    mode = 'setup'
    status = 'starting'
    last_capture = None
    gaps = []
    setup_deadline = time.perf_counter() + args.setup_seconds
    try:
        actuator = open_vgamepad_actuator(settle_seconds=0.5)
        if args.launch:
            subprocess.Popen([str(args.launch), '-screen-fullscreen', '1',
                              '-screen-width', '1280', '-screen-height', '720'],
                             cwd=args.launch.parent)
            time.sleep(6)
        source = open_screen_source(timeout=2)
        print(json.dumps({'status': 'ready', 'command': str(args.output / 'command.json')}), flush=True)
        with (args.output / 'frames.jsonl').open('x') as frames, (args.output / 'commands.jsonl').open('x') as commands:
            while True:
                tick = time.perf_counter()
                if not incoming.empty():
                    stdin_command = incoming.get_nowait()
                command_path = args.output / 'command.json'
                if stdin_command is not None or command_path.exists():
                    try:
                        command = stdin_command if stdin_command is not None else json.loads(command_path.read_text())
                    except (ValueError, OSError):
                        command = {}
                    if command.get('id') is not None and command['id'] != last_id and (
                        command.get('mode') == 'stop' or game_focused()
                    ):
                        last_id = command['id']
                        if command.get('mode') == 'stop':
                            status = 'requested_stop'
                            break
                        if command.get('mode') == 'explore':
                            mode = 'explore'
                            started = started or tick
                            next_action = tick
                        if 'buttons' in command:
                            mode = 'setup'
                            active = ControlInput('setup', **command['buttons'])
                            if game_focused():
                                actuator.send_buttons(active)
                                action_t = time.perf_counter()
                                release_at = action_t + min(5, max(0.01, command.get('hold_seconds', 0.1)))
                                commands.write(json.dumps({'t': action_t, 'source': 'setup', **command}) + '\n')
                                commands.flush()
                if started is not None and tick - started >= args.seconds:
                    status = 'time_budget'
                    break
                if started is None and tick >= setup_deadline:
                    status = 'setup_timeout'
                    break
                if not game_focused():
                    actuator.neutral()
                    active = ControlInput('neutral')
                    if started is not None:
                        status = 'focus_lost'
                        break
                    time.sleep(0.1)
                    continue
                if release_at and tick >= release_at:
                    actuator.neutral()
                    active = ControlInput('neutral')
                    action_t = time.perf_counter()
                    release_at = 0
                if mode == 'explore' and tick >= next_action:
                    # Release between jumps, so consecutive jumps are distinct presses.
                    active = exploration_action(rng)
                    actuator.send_buttons(active)
                    action_t = time.perf_counter()
                    duration = rng.uniform(0.18, 0.42)
                    release_at = action_t + duration
                    next_action = release_at + 0.08
                    commands.write(json.dumps({'t': action_t, 'source': 'exploration', 'buttons': asdict(active)}) + '\n')
                    commands.flush()
                captured_input = asdict(active)
                captured_action_t = action_t
                frame = source.read()
                if not game_focused():
                    actuator.neutral()
                    if started is not None:
                        status = 'focus_lost'
                        break
                    continue
                image = Image.frombytes('RGB', (frame.width, frame.height), frame.payload)
                is_death = death_reference is not None and float(np.abs(death_patch(image) - death_reference).mean()) < 0.075
                death_count = death_count + 1 if is_death else 0
                # Require consecutive matches to the inspected death card. Never
                # infer a victory or confirm an unrecognized menu.
                if death_count >= 3 and tick >= retry_until:
                    actuator.neutral()
                    active = ControlInput('retry', a=True)
                    actuator.send_buttons(active)
                    action_t = time.perf_counter()
                    release_at = action_t + 0.12
                    retry_until = action_t + 4
                    next_action = retry_until
                    started = started or action_t
                    mode = 'retry'
                    retries += 1
                    commands.write(json.dumps({'t': action_t, 'source': 'verified_death_retry', 'buttons': asdict(active)}) + '\n')
                    commands.flush()
                elif mode == 'retry' and tick >= retry_until and not is_death:
                    mode = 'explore'
                    next_action = tick
                if tick >= latest_at:
                    preview = args.output / 'latest.tmp.png'
                    image.resize((960, 540)).save(preview, compress_level=1)
                    preview.replace(args.output / 'latest.png')
                    latest_at = tick + 1
                    print(json.dumps({'mode': mode, 'frames': count, 'elapsed': None if started is None else tick-started}), flush=True)
                if started is not None:
                    image_path = f'frames/{count:08d}.png'
                    image.resize((256, 256), Image.Resampling.BILINEAR).save(args.output / image_path, compress_level=1)
                    frames.write(json.dumps({'frame': count, 'capture_t': frame.t_capture,
                                             'action_t': captured_action_t, 'checksum': frame.checksum,
                                             'image': image_path, 'mode': mode,
                                             'held_input': captured_input, 'death_match': bool(is_death)}) + '\n')
                    frames.flush()
                    if last_capture is not None:
                        gaps.append(frame.t_capture-last_capture)
                    last_capture = frame.t_capture
                    count += 1
                time.sleep(max(0, tick + 1 / args.fps - time.perf_counter()))
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        status = f'error: {type(exc).__name__}: {exc}'
        print(status, flush=True)
    finally:
        for resource in (actuator, source):
            if resource is not None:
                resource.close()
        summary = {'status': status, 'frames': count, 'policy': 'untrained_seeded_exploration',
                   'retry_count': retries,
                   'seed': args.seed, 'training_performed': False, 'cuphead_victory_verified': False,
                   'labels_verified': False, 'resolution': [256,256], 'target_fps': args.fps,
                   'elapsed_seconds': None if started is None else time.perf_counter()-started,
                   'max_capture_gap_seconds': max(gaps, default=0),
                   'mean_capture_gap_seconds': sum(gaps)/len(gaps) if gaps else None,
                   'action_semantics': 'held controller report at capture; action_t is report publish timestamp'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
