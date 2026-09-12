# Frozen latent consequence memory

The implementation is runnable through `scripts/run_memory_agent.py`. Its visual
encoder is frozen, its action-conditioned world model is a local kNN transition
bank, and its planner scores predicted consequences. No action classifier or
online encoder training is involved.

## What runs

- `perception/frozen_jepa.py`: Meta's pretrained V-JEPA 2.1 encoder, strict checkpoint
  loading, frozen parameters, CPU dynamic INT8 Linear layers, normalized 384/768-D
  outputs. Static scenes use the image branch at 128×128. Combat uses consecutive
  four-frame clips at 256×256. Native features are compressed with a seeded fixed
  random projection where needed; this projection is not learned distillation.
- `memory/latent_bank.py`: bounded transition memory with exact scenario/action
  filtering, distance-radius rejection, k nearest neighbors, validated frame order,
  observed duration, and atomic JSON persistence. An action's identity includes its
  button bindings, hold duration and release duration. Banks reject encoder changes
  using checkpoint/source/preprocessing/projection fingerprints.
- `world_model/knn_dynamics.py`: weighted local latent deltas, observed reward,
  hit/terminal probabilities, and disagreement/distance uncertainty. An unsupported
  action returns `None`, never an invented confident prediction.
- `planner/memory_planner.py`: designated-action retrieval when local evidence is
  usable, otherwise locally untried actions and consequence scoring. Goal progress,
  hit risk and uncertainty affect the score. This is one-step prediction, not a
  trained long-horizon MPC or reflex policy.
- `strategist/landmark_route.py` and `perception/landmarks.py`: visual breadcrumbs
  through menus and the overworld. Short edges form a graph so resets can re-localize.
  Calibrated image patches verify landmarks; movement time alone never advances the
  route. Supplemental patches prevent an HP label alone from identifying a level.
- `orchestration/memory_agent.py`: capture, classify, encode, retrieve, actuate,
  observe and remember. Unknown/loading screens release input. An unverified
  transition does not contaminate the bank. Terminal results need repeated visual
  verification. Input release is protected by `finally`.

The existing FLAN-T5 sequencing experiment remains available independently.

## Setup

Use the existing project virtual environment:

```bash
.venv/bin/python -m pip install -r requirements-memory.txt
mkdir -p checkpoints/vjepa2
git -C checkpoints/vjepa2 init
git -C checkpoints/vjepa2 fetch --depth 1 https://github.com/facebookresearch/vjepa2.git 204698b45b3712590f06245fbfba32d3be539812
git -C checkpoints/vjepa2 checkout --detach FETCH_HEAD
curl -L --fail https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt -o checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

The official checkpoint is 1,664,223,428 bytes. SHA-256:
`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.
It includes training components; the loader memory-maps the file and loads only the
encoder. Downloads and upstream source live under ignored `checkpoints/`.

This CPU configuration uses **ViT-B distilled from ViT-G**, as published in
[Meta's V-JEPA repository](https://github.com/facebookresearch/vjepa2).
It is not the full ViT-G running in 384 dimensions. `--variant` also exposes the
large, giant and gigantic constructors, but they require their own matching
checkpoints and sufficient RAM during floating-point loading and quantization.
Only the base variant has been exercised here. Quantization is CPU Linear INT8;
convolutions and normalization remain floating point. `--quantization none`
supports a selected `--device`, including CUDA when available.

## Run and inspect evidence

```bash
# No ML packages, game, or controller required; explicitly a synthetic fixture.
python3 scripts/run_memory_agent.py --synthetic --bank /tmp/cuphead-fixture-bank.json
python3 scripts/run_memory_agent.py --synthetic --bank /tmp/cuphead-fixture-bank.json

# Actual pretrained inference, frozen/INT8 checks, and per-resolution timings.
.venv/bin/python scripts/run_memory_agent.py --benchmark

# Pure-logic regression suite, then optional pretrained/integration tests.
python3 -m unittest discover -s tests -q
CUPHEAD_TEST_VJEPA=1 .venv/bin/python -m unittest discover -s tests -q
```

Every run writes an `experiments/` JSON summary. Live runs also write a JSONL
transition trace and a last-frame image. A synthetic fixture always records
`cuphead_victory_verified: false`. The `--navigation-only` flag can report
`goal_reached`; that also leaves the victory field false.

On the current Linux/Wine machine, direct X11 **window** capture works while MSS
capture of the XWayland root returns black pixels. The virtual Xbox controller
can be created, but its confirm input did not visibly affect the game. Keyboard
confirm/movement were verified, so use the explicit keyboard fallback:

```bash
.venv/bin/python scripts/run_memory_agent.py \
  --real --controller keyboard --navigation-only \
  --route data/landmarks/forest_follies.json \
  --bank data/memory/navigation.json \
  --max-seconds 90 --max-steps 500 \
  --launch wine /home/nirav/Downloads/Cuphead/game_info/data/Cuphead.exe \
  -screen-fullscreen 0 -screen-width 960 -screen-height 540
```

Omit `--launch` to attach to an already open game. Gamepad mode creates the pad
before launch. Keyboard mode targets the Cuphead X11 window; customized game
bindings require updating its explicit key mapping. The real runner owns and
cleans up only the game process it launched, saves its bank on interruption, and
releases controller input on exit.

## Live navigation evidence

`experiments/boot_to_forest_follies.json` and its JSONL trace record an autonomous
fresh boot through title → Start → save slot → character → overworld → interaction
prompt → Forest Follies entrance → playable level. The run used the actual frozen
INT8 encoder and reused previously observed menu consequences. It ended with
`goal_reached`, with `cuphead_victory_verified: false`. Calibration screenshots
were collected interactively beforehand; the validation run executed the route
without manual input. This is one navigation run, not a reliability estimate.

## Limits of the current result

The included reference screenshots describe this machine's existing save,
English UI, and 960×540 window. They are a route calibration, not a general menu
recognizer for arbitrary save slots or arbitrary overworld starting positions.
An unknown screen stops the run rather than triggering blind confirmation.

The CPU benchmark measured roughly 46 ms median for image128 and 300 ms for
video256 while Cuphead was running. The video path does **not** meet the 66 ms
15 Hz tactical budget, and this change does not supply a trained 60 Hz reflex
policy. The pretrained encoder is not a Cuphead combat world model: hit and
reward predictions require actual local transition data.

A completed first-level combat policy, calibrated HP2/HP1/death/victory detectors,
and statistically validated level clears are still missing. The included
`forest_ready` landmark verifies arrival in Forest Follies; it does not verify
completion. Do not interpret a synthetic completion, navigation result, or
passing test suite as a Cuphead clear.
