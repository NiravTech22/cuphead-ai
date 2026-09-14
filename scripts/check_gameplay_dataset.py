#!/usr/bin/env python3
"""Report dataset coverage and run the mandatory frozen-encoder variance gate.

Exit 0 only when BOTH coverage and measured diversity pass; exit 2 otherwise.
No training, predictor loading, action selection, or navigation is performed.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cuphead.evaluation.dataset_gate import inspect_dataset, probe_encoder, finish_report, label_segment
from cuphead.memory.replay import _atomic_write_text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="Record a human-reviewed outcome after capturing a segment")
    label.add_argument("directory", type=Path)
    label.add_argument("--outcome", required=True, choices=["DEATH", "KNOCKOUT", "INCOMPLETE", "NAVIGATION", "IDLE"])
    label.add_argument("--full-attempt", action="store_true")
    label.add_argument("--notes", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--root", type=Path, default=REPO / "data/replays")
    audit.add_argument("--output", type=Path, default=REPO / "experiments/gameplay_dataset_gate.json")
    audit.add_argument("--width", type=int, default=256)
    audit.add_argument("--height", type=int, default=256)
    audit.add_argument("--fps", type=float, default=30)
    audit.add_argument("--boss", default="forest_follies")
    audit.add_argument("--samples", type=int, default=240)
    audit.add_argument("--coverage-only", action="store_true", help="Always leaves training blocked")
    args = ap.parse_args()
    if args.command == "label":
        label_segment(args.directory, outcome=args.outcome, full_attempt=args.full_attempt, notes=args.notes)
        return 0
    # Invalidate an earlier passing result before any potentially failing work.
    _atomic_write_text(args.output, json.dumps({"training_allowed": False, "data_collection_done": False,
                                               "status": "audit_in_progress"}) + "\n")
    report, segments = inspect_dataset(args.root, width=args.width, height=args.height,
                                      fps=args.fps, boss=args.boss, samples=args.samples)
    probe = None
    if not args.coverage_only and report["sample_count"] >= 200:
        try:
            from cuphead.perception.frozen_jepa import FrozenJEPAEncoder
            encoder = FrozenJEPAEncoder(repository=REPO / "checkpoints/vjepa2",
                                        checkpoint=REPO / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")
            probe = probe_encoder(segments, encoder, samples=args.samples)
        except Exception as exc:
            report["blocking_reasons"].append(f"encoder probe failed: {type(exc).__name__}: {exc}")
    report = finish_report(report, probe)
    _atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"segments", "encoder_probe"}}, indent=2))
    if probe:
        print(json.dumps({k: v for k, v in probe.items() if k != "samples"}, indent=2))
    print(f"Full report: {args.output}")
    return 0 if report["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
