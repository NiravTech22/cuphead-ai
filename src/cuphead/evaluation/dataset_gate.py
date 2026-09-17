"""Coverage and encoder-diversity gate for human gameplay replays. No training."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from cuphead.memory.replay import ReplayReader, _atomic_write_text
from cuphead.control.keyboard_input import KEYBOARD_BINDINGS, KEYBOARD_KEYS, keyboard_action

OUTCOMES = {"attempt": {"DEATH", "KNOCKOUT", "INCOMPLETE"},
            "menu": {"NAVIGATION", "INCOMPLETE"}, "idle": {"IDLE"}}
BASELINE_VARIANCE = 0.00012801
# A conservative operational definition of 'comparable', not a calibrated
# scientific boundary between healthy representations and collapsed ones.
VARIANCE_FLOOR = 2 * BASELINE_VARIANCE
MIN_FRAMES = 100_000
MIN_SAMPLES = 200


def label_segment(directory: Path, *, outcome: str, full_attempt: bool = False,
                  notes: str = "") -> None:
    path = directory / "meta.json"
    doc = json.loads(path.read_text())
    extra = doc["extras"]
    kind = extra["segment_type"]
    if extra.get("schema") != "dataset_segment_v1" or outcome not in OUTCOMES[kind]:
        raise ValueError("invalid dataset segment or outcome")
    if full_attempt and (kind != "attempt" or outcome not in {"DEATH", "KNOCKOUT"}):
        raise ValueError("a full attempt must have a witnessed death or clear")
    doc["outcome"] = outcome
    extra.update(labels_verified=True, full_attempt=full_attempt, notes=notes)
    _atomic_write_text(path, json.dumps(doc, indent=2, allow_nan=False) + "\n")


def select_samples(segments: list[dict], count: int = 240) -> list[tuple[dict, int]]:
    """Evenly cover type/boss/phase/outcome strata and time within each stratum."""
    if count < MIN_SAMPLES:
        raise ValueError("variance probe requires at least 200 distinct frames")
    groups = defaultdict(list)
    for segment in segments:
        groups[(segment["segment_type"], segment["boss"], segment["phase"], segment["outcome"])].append(segment)
    if not groups:
        return []
    sizes = {key: sum(s["frame_count"] for s in items) for key, items in groups.items()}
    quotas = {key: 0 for key in groups}
    remaining = min(max(count, len(groups)), sum(sizes.values()))
    while remaining:
        for key in sorted(groups):
            if remaining and quotas[key] < sizes[key]:
                quotas[key] += 1
                remaining -= 1
    selected = []
    for key in sorted(groups):
        size, quota = sizes[key], quotas[key]
        for i in range(quota):
            # Include both ends; integer spacing never selects a frame twice.
            offset = i * (size - 1) // (quota - 1) if quota > 1 else size // 2
            for segment in groups[key]:
                if offset < segment["frame_count"]:
                    selected.append((segment, offset))
                    break
                offset -= segment["frame_count"]
    return selected


def variance_metrics(latents) -> dict:
    import numpy as np

    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2 or z.shape[1] != 384 or not np.isfinite(z).all():
        raise ValueError("expected finite 384-dimensional normalized embeddings for at least two frames")
    norms = np.linalg.norm(z, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("variance comparison requires unit-normalized embeddings")
    variance = float(z.var(axis=0).mean())
    return {"samples": len(z), "dimension": z.shape[1], "mean_output_variance": variance,
            "min_dimension_variance": float(z.var(axis=0).min()),
            "mean_norm": float(norms.mean()), "ratio_to_previous_probe": variance / BASELINE_VARIANCE,
            "still_collapsed_or_low_diversity": variance <= VARIANCE_FLOOR}


def inspect_dataset(root: Path, *, width: int, height: int, fps: float,
                    boss: str, samples: int = 240) -> tuple[dict, list[dict]]:
    """Validate actual stored pixels/actions/times; reject legacy/synthetic rows."""
    from PIL import Image

    if min(width, height) < 1 or not math.isfinite(fps) or fps <= 0 or not boss.strip():
        raise ValueError("positive capture settings and a target boss are required")
    if samples < MIN_SAMPLES:
        raise ValueError("at least 200 samples required")
    digest = hashlib.sha256()
    segments, excluded, errors = [], [], []
    for directory in sorted(root.iterdir()) if root.exists() else []:
        if not directory.is_dir():
            continue
        try:
            path = directory / "meta.json"
            if not path.exists():
                excluded.append({"segment": directory.name, "reason": "unfinalized"})
                continue
            doc = json.loads(path.read_text())
            extra = doc.get("extras", {})
            if extra.get("schema") != "dataset_segment_v1" or extra.get("mode") != "real":
                excluded.append({"segment": directory.name, "reason": "legacy or synthetic; not verified pixel/action data"})
                continue
            replay = ReplayReader(root, directory.name).read()
            n = replay.meta.frame_count
            kind = extra["segment_type"]
            if n <= 0 or kind not in OUTCOMES or replay.meta.outcome not in OUTCOMES[kind]:
                raise ValueError("invalid count/type/outcome")
            if extra.get("labels_verified") is not True or not extra.get("phase"):
                raise ValueError("segment outcome/phase has not been reviewed")
            if extra.get("resolution") != [width, height] or replay.meta.fps != fps:
                raise ValueError("resolution/rate differs from declared training capture settings")
            rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            if len(rows) != n or [r["frame"] for r in rows] != list(range(n)):
                raise ValueError("image/timestamp/action indices are misaligned")
            stamps = [r["capture_t"] for r in rows]
            if not all(math.isfinite(t) for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
                raise ValueError("capture timestamps must be finite and strictly increasing")
            gaps = [b - a for a, b in zip(stamps, stamps[1:])]
            late = sum(gap > 1.5 / fps for gap in gaps)
            if gaps and (abs((n - 1) / (stamps[-1] - stamps[0]) / fps - 1) > .05
                         or max(gaps) > 3 / fps or late / len(gaps) > .01):
                raise ValueError("capture cadence fails: >5% rate error, >3-frame gap, or >1% late intervals")
            for row in rows:
                if (not math.isfinite(row["action_t"]) or row["action_t"] < row["capture_t"]
                        or row["action_t"] - row["capture_t"] > 1 / fps):
                    raise ValueError("input sample is more than one frame late or precedes capture")
                raw = row.get("raw_input")
                if (not isinstance(raw, dict) or raw.get("backend") not in {"evdev", "x11_keyboard"}
                        or not all(isinstance(raw.get(k), dict) for k in ("keys", "axes"))):
                    raise ValueError("missing physical input snapshot (menu controls cannot be inferred)")
                action = replay.actions[row["frame"]]["action"]
                if raw["backend"] == "x11_keyboard":
                    if (extra.get("input_source") != "keyboard"
                            or extra.get("keyboard_bindings") != KEYBOARD_BINDINGS
                            or raw.get("focused") is not True or raw["axes"]
                            or set(raw["keys"]) != set(KEYBOARD_KEYS)
                            or any(type(v) not in (int, bool) or v not in (0, 1) for v in raw["keys"].values())):
                        raise ValueError("invalid keyboard snapshot, focus or default bindings")
                    if action != keyboard_action(raw["keys"]).to_buttons():
                        raise ValueError("keyboard state does not match normalized action")
                elif extra.get("input_source") == "keyboard":
                    raise ValueError("keyboard segment contains non-keyboard input")
                if kind == "idle" and (any(raw["keys"].values()) or any(
                        action.get(k) for k in ("a", "b", "x", "rt", "stick_x", "stick_y"))):
                    raise ValueError("idle segment contains input")
                expected = f"frames/{row['frame']:08d}.png"
                if row["image"] != expected:
                    raise ValueError("unexpected or missing image path")
                image_path = directory / expected
                with Image.open(image_path) as im:
                    if im.size != (width, height) or im.mode != "RGB":
                        raise ValueError("stored image size/color mode mismatch")
                    im.verify()
                digest.update(directory.name.encode())
                digest.update(expected.encode())
                digest.update(image_path.read_bytes())
            for filename in ("meta.json", "actions.jsonl", "frames.jsonl", "events.json", "hud.jsonl"):
                digest.update((directory / filename).read_bytes())
            segments.append({"directory": str(directory), "segment": directory.name,
                             "segment_type": kind, "boss": replay.meta.boss,
                             "phase": extra["phase"], "outcome": replay.meta.outcome,
                             "full_attempt": extra.get("full_attempt") is True,
                             "frame_count": n, "captured_seconds": stamps[-1] - stamps[0] + 1 / fps,
                             "capture_span_seconds": stamps[-1] - stamps[0],
                             "late_intervals": late,
                             "repeated_images": sum(a["checksum"] == b["checksum"] for a, b in zip(rows, rows[1:]))})
        except Exception as exc:
            errors.append({"segment": directory.name, "reason": f"{type(exc).__name__}: {exc}"})
    counts = Counter()
    outcomes = Counter()
    for segment in segments:
        counts[segment["segment_type"]] += segment["frame_count"]
        outcomes[segment["outcome"]] += segment["frame_count"]
    total = sum(counts.values())
    attempts = [s for s in segments if s["segment_type"] == "attempt" and s["boss"] == boss
                and s["full_attempt"] and s["outcome"] in {"DEATH", "KNOCKOUT"}]
    reasons = []
    if total < MIN_FRAMES:
        reasons.append(f"only {total} eligible frames; need at least {MIN_FRAMES}")
    if len(attempts) < 5:
        reasons.append(f"only {len(attempts)} full {boss} attempts; target five")
    if sum(s["outcome"] == "DEATH" for s in attempts) < 2:
        reasons.append("need at least two witnessed full attempts ending in death")
    if counts["menu"] < 100 or counts["idle"] < 100:
        reasons.append("need at least 100 reviewed frames each of menu and idle")
    if not total or counts["attempt"] < .7 * total:
        reasons.append("at least 70% of eligible frames must be gameplay attempts; idle/menu cannot pad the target")
    if errors:
        reasons.append("one or more real segments failed data integrity or label checks")
    selected = select_samples(segments, samples)
    if len(selected) < MIN_SAMPLES:
        reasons.append(f"only {len(selected)} sample frames; need at least 200")
    return {
        "schema": "gameplay_dataset_gate_v1", "root": str(root),
        "training_capture_settings": {"width": width, "height": height, "fps": fps, "pixel_format": "RGB"},
        "dataset_sha256": digest.hexdigest(), "total_frame_count": total,
        "frames_by_segment_type": {k: counts[k] for k in OUTCOMES},
        "frames_by_outcome": dict(outcomes), "full_target_attempts": len(attempts),
        "successful_clears": sum(s["outcome"] == "KNOCKOUT" for s in attempts),
        "clear_note": "A successful clear is desired if manually achievable; never inferred from level entry.",
        "captured_wall_clock_seconds": sum(s["captured_seconds"] for s in segments),
        "capture_span_seconds": sum(s["capture_span_seconds"] for s in segments),
        "time_definition": "sum of per-segment last-first capture timestamps plus one nominal frame interval; excludes breaks",
        "segments": segments, "excluded": excluded, "invalid_segments": errors,
        "coverage_passed": not reasons, "blocking_reasons": reasons,
        "encoder_probe": None, "training_allowed": False, "data_collection_done": False,
        "sample_count": len(selected),
    }, segments


def probe_encoder(segments: list[dict], encoder, *, samples: int = 240) -> dict:
    from PIL import Image

    selected = select_samples(segments, samples)
    if len(selected) < MIN_SAMPLES:
        raise ValueError("fewer than 200 eligible dataset frames; no substitute calibration screenshots")
    comparable, temporal, records = [], [], []
    by_type = defaultdict(list)
    for number, (segment, index) in enumerate(selected):
        directory = Path(segment["directory"])
        with Image.open(directory / f"frames/{index:08d}.png") as im:
            encoder.reset()
            comparable.append(encoder.encode(im, low_detail=True))
        # Consecutive clip from the same segment only; do not mix distant samples.
        clip = []
        for frame in range(max(0, index - encoder.clip_frames + 1), index + 1):
            with Image.open(directory / f"frames/{frame:08d}.png") as im:
                clip.append(im.copy())
        z = encoder.encode_clip(clip, low_detail=False)
        temporal.append(z)
        by_type[segment["segment_type"]].append(z)
        records.append({"segment": segment["segment"], "frame": index,
                        "segment_type": segment["segment_type"], "outcome": segment["outcome"]})
        if (number + 1) % 10 == 0:
            print(f"Encoder gate: {number + 1}/{len(selected)} dataset samples", flush=True)
    old_mode = variance_metrics(comparable)
    video_mode = variance_metrics(temporal)
    per_type = {k: variance_metrics(v) for k, v in by_type.items() if len(v) >= 2}
    failed = old_mode["still_collapsed_or_low_diversity"] or video_mode["still_collapsed_or_low_diversity"]
    failed = failed or "attempt" not in per_type or per_type["attempt"]["still_collapsed_or_low_diversity"]
    return {
        "encoder": encoder.spec, "samples": records,
        "variance_definition": "mean across dimensions of population variance across unit-normalized embeddings",
        "previous_eight_screenshot_variance": BASELINE_VARIANCE,
        "comparable_threshold": VARIANCE_FLOOR,
        "threshold_note": "Conservative 2x prior-probe heuristic; absolute normalized variance alone does not prove encoder collapse.",
        "independent_image_128_comparable_to_previous": old_mode,
        "consecutive_clip_256": video_mode, "clip_variance_by_segment_type": per_type,
        "passed": not failed,
        "status": "still_collapsed_or_low_diversity" if failed else "diversity_check_passed",
        "recommendation": "Inspect encoder/preprocessing before collecting more of the same data" if failed else None,
    }


def finish_report(report: dict, probe: dict | None) -> dict:
    report["encoder_probe"] = probe
    if probe is None:
        report["blocking_reasons"].append("encoder variance has not been measured on >=200 new dataset frames")
    elif not probe["passed"]:
        report["blocking_reasons"].append("STILL-COLLAPSED / low-diversity encoder output: inspect encoder before proceeding")
    report["training_allowed"] = report["coverage_passed"] and probe is not None and probe["passed"]
    report["data_collection_done"] = report["training_allowed"]
    return report
