#!/usr/bin/env python3
"""Run the frozen-encoder episodic agent, a synthetic integration fixture, or a benchmark."""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--real", action="store_true")
    parser.add_argument("--repository", type=Path, default=REPO / "checkpoints/vjepa2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
    )
    parser.add_argument(
        "--variant", choices=["base", "large", "giant", "gigantic"], default="base"
    )
    parser.add_argument("--dimension", type=int, choices=[384, 768], default=384)
    parser.add_argument("--quantization", choices=["int8", "none"], default="int8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--route",
        type=Path,
        help="calibrated landmark graph JSON; required for real control",
    )
    parser.add_argument(
        "--navigation-only",
        action="store_true",
        help="stop at route landmark; does not claim a level victory",
    )
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-video", type=Path, help="record the game drawable to MJPG .avi with capture timestamps")
    parser.add_argument(
        "--controller", choices=["gamepad", "keyboard"], default="gamepad"
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument(
        "--launch", nargs=argparse.REMAINDER, help="launch command (must be last)"
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_seconds <= 0:
        parser.error("positive step/time budgets required")
    if args.real and args.route is None:
        parser.error("--real requires --route with calibrated game screenshots")
    if args.launch and not args.real:
        parser.error("--launch is only supported with --real")
    if args.record_video and (not args.real or args.record_video.suffix.lower() != ".avi"):
        parser.error("--record-video requires --real and an .avi output path")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = (
        "synthetic" if args.synthetic else ("benchmark" if args.benchmark else "real")
    )
    output = args.output or REPO / "experiments" / f"memory_{mode}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        from cuphead.evaluation.synthetic_agent import FixtureEncoder, run_synthetic
        from cuphead.memory.latent_bank import LatentBank

        bank = (
            LatentBank.load(args.bank, encoder_id=FixtureEncoder.fingerprint)
            if args.bank and args.bank.exists()
            else None
        )
        result, bank = run_synthetic(bank=bank, max_steps=args.max_steps)
        if args.bank:
            bank.save(args.bank)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["fixture_completed"] else 1

    from cuphead.perception.frozen_jepa import FrozenJEPAEncoder

    started = time.perf_counter()
    encoder = FrozenJEPAEncoder(
        repository=args.repository,
        checkpoint=args.checkpoint,
        variant=args.variant,
        output_dim=args.dimension,
        quantization=args.quantization,
        device=args.device,
    )
    print(
        f"Loaded {encoder.backend_name} in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    if args.benchmark:
        import torch
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (960, 540), (70, 100, 150))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 420, 960, 540), fill=(50, 150, 70))
        draw.ellipse((100, 320, 160, 410), fill="white")
        latencies = {}
        for low in (True, False):
            encoder.reset()
            encoder.encode(image, low_detail=low)
            durations = []
            for _ in range(10):
                begin = time.perf_counter()
                latent = encoder.encode(image, low_detail=low)
                durations.append(1000 * (time.perf_counter() - begin))
            latencies["image128" if low else "video256"] = {
                "median_ms": statistics.median(durations),
                "max_ms": max(durations),
                "dimension": len(latent),
                "l2_norm": sum(x * x for x in latent) ** 0.5,
                "input_shape": encoder.last_input_shape,
            }
        result = {
            "environment": "pretrained_encoder_benchmark",
            "encoder": encoder.spec,
            "all_parameters_frozen": all(
                not p.requires_grad for p in encoder._model.parameters()
            ),
            "quantized_linear_count": sum(
                isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
                for m in encoder._model.modules()
            ),
            "latency": latencies,
            "cuphead_victory_verified": False,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0

    from cuphead.control.actuator import open_vgamepad_actuator
    from cuphead.control.timed_input import TimedExecutor
    from cuphead.control.x11_keyboard import X11KeyboardActuator
    from cuphead.memory.latent_bank import LatentBank
    from cuphead.orchestration.memory_agent import MemoryAgent
    from cuphead.perception.landmarks import LandmarkVerifier
    from cuphead.perception.window_capture import X11WindowSource
    from cuphead.planner.memory_planner import MemoryPlanner
    from cuphead.strategist.landmark_route import LandmarkRoute, RouteEdge
    from cuphead.world_model.knn_dynamics import KNNDynamics

    verifier = LandmarkVerifier(args.route)
    route = LandmarkRoute(
        [RouteEdge(**edge) for edge in verifier.document["edges"]],
        verifier.document["goal"],
    )
    if route.goal not in verifier.references:
        raise ValueError("route goal must have a calibrated reference image")
    goal_node = next(
        node for node in verifier.document["landmarks"] if node["label"] == route.goal
    )
    if not args.navigation_only and goal_node.get("mode") != "victory":
        parser.error(
            "this route only verifies navigation; use --navigation-only "
            "or provide a route ending at a calibrated victory landmark"
        )
    bank_path = args.bank or REPO / "data/memory" / f"{encoder.fingerprint}.json"
    bank = (
        LatentBank.load(bank_path, encoder_id=encoder.fingerprint)
        if bank_path.exists()
        else LatentBank(encoder.output_dim, encoder.fingerprint)
    )
    if bank.dimension != encoder.output_dim:
        raise ValueError("bank dimension differs from encoder")
    source = actuator = child = recorder = None
    result = {
        "environment": "live_cuphead",
        "cuphead_victory_verified": False,
        "status": "starting",
        "encoder": encoder.spec,
        "controller": args.controller,
        "steps": 0,
    }
    log_path = output.with_suffix(".jsonl")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.controller == "gamepad":
            actuator = open_vgamepad_actuator()
        if args.launch:
            child = subprocess.Popen(
                args.launch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        deadline = time.perf_counter() + min(45, args.max_seconds)
        while source is None:
            try:
                source = X11WindowSource()
            except RuntimeError:
                if time.perf_counter() >= deadline or (
                    child and child.poll() is not None
                ):
                    raise
                time.sleep(0.25)
        if actuator is None:
            actuator = X11KeyboardActuator()
        if args.record_video:
            from cuphead.perception.video_recorder import WindowVideoRecorder

            recorder = WindowVideoRecorder(args.record_video)
        with log_path.open("w") as stream:

            def emit(row):
                row = {"t_monotonic": time.perf_counter(), **row}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)

            agent = MemoryAgent(
                source=source,
                encoder=encoder,
                verifier=verifier,
                executor=TimedExecutor(actuator),
                bank=bank,
                planner=MemoryPlanner(KNNDynamics(bank, radius=args.radius)),
                route=route,
                emit=emit,
                navigation_only=args.navigation_only,
            )
            deadline = time.perf_counter() + args.max_seconds
            unknown = 0
            for step in range(args.max_steps):
                if recorder:
                    recorder.check()
                if time.perf_counter() >= deadline:
                    result["status"] = "time_budget"
                    break
                if child and child.poll() is not None:
                    result["status"] = "game_exited"
                    break
                observation = agent.step()
                result["steps"] = step + 1
                if step % 20 == 0 or observation.status in {
                    "observed",
                    "goal_reached",
                    "won",
                }:
                    from cuphead.perception.latent_encoder import _as_pil_image

                    _as_pil_image(agent.last_frame).save(
                        output.with_suffix(".last.png")
                    )
                if observation.status == "goal_reached":
                    result["status"] = "goal_reached"
                    break
                if observation.status == "won":
                    result["status"] = "won"
                    result["cuphead_victory_verified"] = True
                    break
                unknown = unknown + 1 if observation.status == "unknown" else 0
                if unknown >= 300:
                    result["status"] = "unrecognized_screen"
                    break
                if (step + 1) % 10 == 0:
                    bank.save(bank_path)
                if observation.status in {"unknown", "verifying"}:
                    time.sleep(0.1)
            else:
                result["status"] = "step_budget"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except Exception as exc:  # noqa: BLE001 -- persist the run failure before cleanup
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = []
        for resource in (actuator, recorder, source):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001 -- finish all remaining cleanup
                    cleanup_errors.append(f"{type(resource).__name__}: {exc}")
        if recorder:
            result["video"] = recorder.summary()
        if child and child.poll() is None:
            try:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            except OSError as exc:
                cleanup_errors.append(f"game process: {exc}")
        signal.signal(signal.SIGTERM, previous)
        try:
            bank.save(bank_path)
        except OSError as exc:
            cleanup_errors.append(f"saving bank: {exc}")
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["status"] = "error"
        result["transitions"] = len(bank)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"won", "goal_reached"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
