#!/usr/bin/env python3
"""Measure the existing state without treating upstream weights as Cuphead training."""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    import numpy as np
    import torch
    from PIL import Image

    from cuphead.memory.latent_bank import Experience, LatentBank
    from cuphead.perception.frozen_jepa import FrozenJEPAEncoder
    from cuphead.planner.memory_planner import MemoryPlanner
    from cuphead.world_model.knn_dynamics import KNNDynamics

    checkpoint = REPO / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    report = {
        "cuphead_training_resumed": False,
        "resumed_step": None,
        "resumed_epoch": None,
        "cuphead_training_loss": None,
        "reason": "No Cuphead predictor training checkpoint or Phase 3 evaluation artifact found",
        "checkpoint_inventory": [str(p.relative_to(REPO)) for p in (REPO / "checkpoints").rglob("*.pt")],
        "upstream_pretraining_only": {
            "path": str(checkpoint), "bytes": checkpoint.stat().st_size,
            "keys": list(weights), "epoch": weights["epoch"], "loss": weights["loss"],
        },
    }
    del weights
    encoder = FrozenJEPAEncoder(repository=REPO / "checkpoints/vjepa2", checkpoint=checkpoint)
    paths = sorted((REPO / "data/landmarks").glob("*.png"))
    latents = []
    for path in paths:
        encoder.reset()
        with Image.open(path) as im:
            latents.append(encoder.encode(im))
    z = np.asarray(latents)
    report["encoder_probe"] = {
        "strict_load_succeeded": True, "encoder": encoder.spec,
        "samples": [str(p.relative_to(REPO)) for p in paths],
        "variance_definition": "mean across dimensions of population variance across independent screenshots, after output normalization",
        "mean_output_variance": float(z.var(axis=0).mean()),
        "min_dimension_variance": float(z.var(axis=0).min()),
        "mean_norm": float(np.linalg.norm(z, axis=1).mean()),
        "limitation": f"{len(paths)} calibration screenshots, mostly menus; not a combat collapse check at a training resume step",
    }
    report["banks"] = []
    for path in sorted((REPO / "data/memory").glob("*.json")):
        bank = json.loads(path.read_text())
        rows = bank["rows"]
        report["banks"].append({
            "path": str(path.relative_to(REPO)), "rows": len(rows),
            "scopes": dict(Counter(r["scope"] for r in rows)),
            "hits": sum(r["hit"] for r in rows),
            "terminal": sum(r["terminal"] for r in rows),
        })

    # Run 02 added seven rows after the first 16. Freeze the bank at that boundary;
    # never train on this held-out run or silently use future appended transitions.
    doc = json.loads((REPO / "data/memory/navigation.json").read_text())
    train, heldout = doc["rows"][:16], doc["rows"][16:23]
    bank = LatentBank(doc["dimension"], doc["encoder_id"])
    for row in train:
        bank.add(Experience(**row))
    model = KNNDynamics(bank)
    rng = np.random.default_rng(17)
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    metrics = []
    for row in heldout:
        prediction = model.predict(row["state"], row["action"], row["scope"])
        if prediction is None:
            continue
        actual = np.asarray(row["next_state"])
        random = rng.normal(size=actual.shape)
        metrics.append({
            "scope": row["scope"],
            "predicted_cosine": cosine(prediction.next_state, actual),
            "no_change_cosine": cosine(row["state"], actual),
            "random_unit_cosine": cosine(random, actual),
            "predicted_mse": float(np.square(np.asarray(prediction.next_state) - actual).mean()),
            "no_change_mse": float(np.square(np.asarray(row["state"]) - actual).mean()),
        })
    report["navigation_sanity_probe"] = {
        "predictor": "existing kNN consequence bank, not the untrained T5 adapters",
        "train_rows": len(train), "heldout_rows": len(heldout), "supported_rows": len(metrics),
        "rows": metrics,
        "means": {key: float(np.mean([r[key] for r in metrics])) for key in (
            "predicted_cosine", "no_change_cosine", "random_unit_cosine", "predicted_mse", "no_change_mse"
        )} if metrics else {},
        "combat_training_gate_passed": False,
        "limitation": "Small navigation-only probe; cannot validate combat predictions or replace the missing Phase 3 check",
    }
    # Known consequence scenarios test the selection logic independently of learning.
    examples = []
    for name, rows, goal, expected in [
        ("avoid_observed_hit", [Experience((0., 0.), "shoot", (1., 0.), "fixture", reward=.5, hit=True),
                               Experience((0., 0.), "duck", (0., 0.), "fixture", reward=.1)], None, "duck"),
        ("prefer_predicted_progress", [Experience((0., 0.), "left", (1., 0.), "fixture"),
                                       Experience((0., 0.), "right", (-1., 0.), "fixture")], (1., 0.), "left"),
    ]:
        fixture = LatentBank(2, "fixture")
        for row in rows:
            fixture.add(row)
        decision = MemoryPlanner(KNNDynamics(fixture)).choose((0., 0.), [r.action for r in rows], "fixture", goal=goal)
        examples.append({"scenario": name, "expected": expected, "decision": asdict(decision), "passed": decision.action == expected})
    report["planner_checks"] = {"signal": "predicted reward + predicted distance reduction to goal - 5 * hit_probability - 0.25 * uncertainty", "synthetic_known_scenarios": examples}
    output = REPO / "experiments/v1_state_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
