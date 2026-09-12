"""Frozen pretrained V-JEPA 2.1; no action classifier and no trained encoder.

Load only the encoder from Meta's checkpoint (not its large pretraining
predictor/optimizer). CPU INT8 quantizes Linear weights; patch convolutions and
normalization stay floating point. A seeded, fixed projection compresses the
native features, and is deliberately not described as distillation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

from .latent_encoder import EncoderUnavailableError, _as_pil_image


class FrozenJEPAEncoder:
    def __init__(
        self,
        *,
        repository: str | Path,
        checkpoint: str | Path,
        variant: str = "base",
        output_dim: int = 384,
        quantization: str = "int8",
        device: str = "cpu",
        clip_frames: int = 4,
        seed: int = 17,
        threads: int = 4,
    ):
        if variant not in {"base", "large", "giant", "gigantic"}:
            raise ValueError("unknown V-JEPA 2.1 variant")
        if output_dim not in {384, 768}:
            raise ValueError("output_dim must be 384 or 768")
        if clip_frames < 2 or clip_frames % 2:
            raise ValueError(
                "clip_frames must be positive and divisible by tubelet size 2"
            )
        if quantization not in {"none", "int8"}:
            raise ValueError("quantization must be none or int8")
        if quantization == "int8" and device != "cpu":
            raise ValueError("dynamic INT8 is a CPU backend; use none for CUDA")
        if threads < 1:
            raise ValueError("threads must be positive")
        repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if not (repository / "app/vjepa_2_1/models/vision_transformer.py").is_file():
            raise EncoderUnavailableError(
                f"missing Meta V-JEPA 2.1 source: {repository}"
            )
        if not checkpoint.is_file():
            raise EncoderUnavailableError(
                f"missing pretrained checkpoint: {checkpoint}"
            )
        try:
            import torch

            sys.path.insert(0, str(repository))
            from app.vjepa_2_1.models import vision_transformer as vit
        except ImportError as exc:
            raise EncoderUnavailableError(
                "V-JEPA 2.1 requires torch, torchvision, timm and einops"
            ) from exc
        self._torch = torch
        torch.set_num_threads(threads)
        self.device, self.output_dim = device, output_dim
        self.clip_frames, self.seed = clip_frames, seed
        self._history: deque[Any] = deque(maxlen=clip_frames)
        self._projection = None
        self.last_input_shape: tuple[int, ...] | None = None
        name = {
            "base": "vit_base",
            "large": "vit_large",
            "giant": "vit_giant_xformers",
            "gigantic": "vit_gigantic_xformers",
        }[variant]
        # mmap avoids materializing the entire training checkpoint in physical RAM.
        weights = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        key = "ema_encoder" if variant in {"base", "large"} else "target_encoder"
        if key not in weights:
            raise EncoderUnavailableError(
                f"checkpoint lacks {key!r}; check the variant"
            )
        state = {
            k.removeprefix("module.").removeprefix("backbone."): v
            for k, v in weights[key].items()
        }
        model = getattr(vit, name)(
            img_size=(384, 384),
            patch_size=16,
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        model.load_state_dict(state, strict=True)
        del state, weights
        model.eval().requires_grad_(False)
        if quantization == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        self._model = model.to(device).eval().requires_grad_(False)
        self.backend_name = f"vjepa2.1:{variant}:{quantization}"
        # Hash actual weights, not a mutable filename, to prevent incompatible banks.
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        source_digest = hashlib.sha256()
        for root in (repository / "app/vjepa_2_1/models", repository / "src"):
            for path in sorted(root.rglob("*.py")):
                source_digest.update(str(path.relative_to(repository)).encode())
                source_digest.update(path.read_bytes())
        self.spec = {
            "backend": self.backend_name,
            "checkpoint_sha256": digest.hexdigest(),
            "source_sha256": source_digest.hexdigest(),
            "dimension": output_dim,
            "clip_frames": clip_frames,
            "projection_seed": seed,
            "pooling": "mean_tokens",
            "preprocess": "rgb-square-imagenet-v1",
            "low_size": 128,
            "high_size": 256,
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(self.spec, sort_keys=True).encode()
        ).hexdigest()

    def reset(self) -> None:
        """Clear motion context at menu/level boundaries and episode resets."""
        self._history.clear()

    def encode_clip(self, frames, *, low_detail: bool = False) -> tuple[float, ...]:
        """Encode consecutive captured frames, rather than widely spaced decisions."""
        frames = list(frames)
        if not frames:
            raise ValueError("at least one frame required")
        self.reset()
        for frame in frames[-self.clip_frames : -1]:
            self._history.append(_as_pil_image(frame))
        return self.encode(frames[-1], low_detail=low_detail)

    def encode(
        self, frame_or_payload: object, *, low_detail: bool = True
    ) -> tuple[float, ...]:
        import numpy as np
        from PIL import Image

        torch = self._torch
        image = _as_pil_image(frame_or_payload)
        self._history.append(image)
        images = list(self._history)
        images = [images[0]] * (self.clip_frames - len(images)) + images
        # Menu/overworld screenshots use the trained image branch, combat the video branch.
        if low_detail:
            images = [images[-1]]
        size = 128 if low_detail else 256
        pixels = np.stack(
            [
                np.asarray(
                    im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
                )
                / 255
                for im in images
            ]
        )
        batch = (
            torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        )
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.last_input_shape = tuple(batch.shape)
        with torch.inference_mode():
            tokens = self._model((batch - mean) / std)
            if not torch.is_tensor(tokens):
                raise RuntimeError("expected one V-JEPA token tensor")
            latent = tokens.mean(dim=1)[0].float()
            if latent.numel() != self.output_dim:
                if self._projection is None:
                    generator = torch.Generator(device="cpu").manual_seed(self.seed)
                    self._projection = (
                        torch.randn(
                            latent.numel(), self.output_dim, generator=generator
                        )
                        / self.output_dim**0.5
                    ).to(self.device)
                latent = latent @ self._projection
            if not torch.isfinite(latent).all() or latent.norm() <= 1e-12:
                raise RuntimeError("encoder returned an invalid latent")
            latent = torch.nn.functional.normalize(latent, dim=0)
        return tuple(latent.cpu().tolist())
