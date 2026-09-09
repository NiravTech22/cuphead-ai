"""Frozen visual feature extraction for latent planning.

The navigation world model deliberately does not train an image encoder. A
pretrained visual transformer is frozen, and only a small action-conditioned
dynamics head adapts online to Cuphead's menus.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Protocol


class EncoderUnavailableError(RuntimeError):
    """Raised when no requested frozen visual-encoder backend is available."""


class LatentEncoder(Protocol):
    def encode(self, frame_or_payload: object) -> tuple[float, ...]: ...


def _as_pil_image(payload: object) -> Any:
    """Convert an MSS shot, PIL image, ndarray, or encoded image bytes to PIL."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise EncoderUnavailableError("Pillow is required for frozen visual encoding") from exc

    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if hasattr(payload, "rgb") and hasattr(payload, "width") and hasattr(payload, "height"):
        return Image.frombytes("RGB", (payload.width, payload.height), payload.rgb)
    if hasattr(payload, "shape"):
        return Image.fromarray(payload).convert("RGB")
    if isinstance(payload, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(payload))).convert("RGB")
    raise TypeError(
        "unsupported frame payload; expected an MSS screenshot, PIL image, ndarray, or encoded bytes"
    )


class FrozenVisualEncoder:
    """A frozen V-JEPA-compatible encoder with a pretrained ViT fallback.

    Set CUPHEAD_VJEPA_MODEL_ID to the local/Hugging Face identifier of an
    installed V-JEPA 2.1-compatible visual model. backend="auto" tries that
    explicit V-JEPA configuration first, then falls back to torchvision's
    pretrained ViT-B/16. No parameters are ever made trainable.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        if backend not in {"auto", "vjepa2", "vit"}:
            raise ValueError("backend must be 'auto', 'vjepa2', or 'vit'")
        self._device = device
        self.backend_name: str
        requested_vjepa = model_id or os.environ.get("CUPHEAD_VJEPA_MODEL_ID")

        if backend in {"auto", "vjepa2"} and requested_vjepa:
            try:
                self._load_transformers(requested_vjepa)
                return
            except (ImportError, OSError, ValueError) as exc:
                if backend == "vjepa2":
                    raise EncoderUnavailableError(
                        f"could not load requested frozen V-JEPA model {requested_vjepa!r}: {exc}"
                    ) from exc

        if backend == "vjepa2":
            raise EncoderUnavailableError(
                "V-JEPA was requested but no model ID was configured. Set CUPHEAD_VJEPA_MODEL_ID "
                "or pass model_id=..."
            )
        self._load_torchvision_vit()

    def _freeze(self, model: Any) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def _load_transformers(self, model_id: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._freeze(self._model)
        self._torch = torch
        self.backend_name = f"vjepa2:{model_id}"
        self._transformers = True

    def _load_torchvision_vit(self) -> None:
        try:
            import torch
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except ImportError as exc:
            raise EncoderUnavailableError(
                "no V-JEPA model was configured and torchvision is unavailable. "
                "Install torchvision or configure CUPHEAD_VJEPA_MODEL_ID."
            ) from exc

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self._model = vit_b_16(weights=weights)
        self._model.heads = torch.nn.Identity()
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._freeze(self._model)
        self._preprocess = weights.transforms()
        self._torch = torch
        self.backend_name = "torchvision:vit_b_16_imagenet1k"
        self._transformers = False

    def encode(self, frame_or_payload: object) -> tuple[float, ...]:
        """Return a stable, L2-normalized frozen visual latent for one frame."""
        image = _as_pil_image(getattr(frame_or_payload, "payload", frame_or_payload))
        torch = self._torch
        with torch.inference_mode():
            if self._transformers:
                batch = self._processor(images=image, return_tensors="pt")
                batch = {key: value.to(self._device) for key, value in batch.items()}
                output = self._model(**batch)
                latent = getattr(output, "pooler_output", None)
                if latent is None:
                    latent = output.last_hidden_state.mean(dim=1)
            else:
                latent = self._model(self._preprocess(image).unsqueeze(0).to(self._device))
            latent = latent.reshape(latent.shape[0], -1)[0]
            latent = latent / latent.norm().clamp_min(1e-12)
        return tuple(float(x) for x in latent.detach().cpu().tolist())
