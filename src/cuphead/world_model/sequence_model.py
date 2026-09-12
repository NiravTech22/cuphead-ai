"""Experimental frozen T5 latent sequencing, for offline training/evaluation.

Torch and Transformers are loaded only by the factory. No tokenizer is needed:
trainable adapters map continuous visual latents/actions into T5 embeddings.
"""

from __future__ import annotations

from typing import Any


def build_frozen_t5_sequence_model(
    latent_dim: int,
    action_dim: int,
    *,
    model_id: str = "google/flan-t5-small",
    backbone: Any = None,
    max_context: int = 64,
    max_horizon: int = 20,
) -> Any:
    """Return a torch module with frozen backbone and trainable adapters.

    ``backbone`` may be an already loaded ``transformers.T5Model`` (including
    a tiny random model for tests). Otherwise pretrained weights are loaded.
    Inputs are float tensors: context [B,C,latent_dim], actions [B,H,action_dim].
    Masks are boolean [B,C] / [B,H], with True for valid, right-padded steps.
    Action features must include duration when held inputs have variable lengths.
    """
    if min(latent_dim, action_dim, max_context, max_horizon) < 1:
        raise ValueError("dimensions and sequence limits must be positive")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("T5 sequencing requires torch and transformers") from exc
    if backbone is None:
        try:
            from transformers import T5Model
        except ImportError as exc:
            raise RuntimeError("T5 sequencing requires transformers") from exc
        backbone = T5Model.from_pretrained(model_id)
    if getattr(backbone.config, "model_type", None) != "t5":
        raise ValueError("backbone must be a T5Model encoder-decoder")

    class FrozenT5SequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.backbone.requires_grad_(False)
            width = backbone.config.d_model
            self.context_adapter = nn.Linear(latent_dim, width)
            self.decoder_adapter = nn.Linear(latent_dim + action_dim, width)
            self.output_adapter = nn.Linear(width, latent_dim)
            # Match an injected backbone's device/dtype before the first call.
            reference = next(backbone.parameters())
            for adapter in self.adapters:
                adapter.to(device=reference.device, dtype=reference.dtype)
            self.backbone.eval()

        @property
        def adapters(self):
            return (self.context_adapter, self.decoder_adapter, self.output_adapter)

        def train(self, mode: bool = True):
            super().train(mode)
            # Freezing weights alone does not disable T5's dropout.
            self.backbone.eval()
            return self

        def _mask(self, values, mask, width, limit, name):
            if values.ndim != 3 or values.shape[-1] != width:
                raise ValueError(f"{name} must have shape [batch, steps, {width}]")
            if values.shape[0] < 1 or not 1 <= values.shape[1] <= limit:
                raise ValueError(f"{name} requires a nonempty batch and 1..{limit} steps")
            if not values.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if mask is None:
                mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if mask.shape != values.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"{name} mask must be boolean [batch, steps]")
            if mask.device != values.device:
                raise ValueError(f"{name} mask must be on the input device")
            if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
                raise ValueError(f"{name} mask must describe nonempty right-padded sequences")
            if not torch.isfinite(values[mask]).all():
                raise ValueError(f"{name} contains nonfinite valid values")
            return mask

        def _prepare(self, context, actions, context_mask, action_mask):
            cm = self._mask(context, context_mask, latent_dim, max_context, "context")
            am = self._mask(actions, action_mask, action_dim, max_horizon, "actions")
            if context.shape[0] != actions.shape[0]:
                raise ValueError("context and actions batch sizes differ")
            if context.device != actions.device or context.dtype != actions.dtype:
                raise ValueError("context and actions must share device and dtype")
            context = context.masked_fill(~cm.unsqueeze(-1), 0)
            actions = actions.masked_fill(~am.unsqueeze(-1), 0)
            last = context[torch.arange(context.shape[0], device=context.device), cm.sum(1) - 1]
            return context, actions, cm, am, last

        def _decode(self, encoded, cm, previous, actions, am):
            result = self.backbone(
                encoder_outputs=encoded,
                attention_mask=cm,
                decoder_inputs_embeds=self.decoder_adapter(torch.cat((previous, actions), dim=-1)),
                decoder_attention_mask=am,
                use_cache=False,
                return_dict=True,
            )
            return previous + self.output_adapter(result.last_hidden_state)

        def forward(self, context, actions, targets, *, context_mask=None, action_mask=None):
            """Teacher-forced predictions; target t is never input to step t.

            Returns [B,H,latent_dim]. Padded predictions are zero. Use ``loss``
            for masked MSE; use ``rollout`` for evaluation without target leakage.
            """
            context, actions, cm, am, last = self._prepare(context, actions, context_mask, action_mask)
            if targets.shape != (*actions.shape[:2], latent_dim):
                raise ValueError("targets must have shape [batch, horizon, latent_dim]")
            if targets.device != context.device or targets.dtype != context.dtype:
                raise ValueError("targets must share context device and dtype")
            if not torch.isfinite(targets[am]).all():
                raise ValueError("targets contains nonfinite valid values")
            targets = targets.masked_fill(~am.unsqueeze(-1), 0)
            previous = torch.cat((last.unsqueeze(1), targets[:, :-1]), dim=1)
            # Keep autograd through frozen T5 so gradients reach input adapters.
            encoded = self.backbone.encoder(
                inputs_embeds=self.context_adapter(context), attention_mask=cm, return_dict=True
            )
            predictions = self._decode(encoded, cm, previous, actions, am)
            return predictions.masked_fill(~am.unsqueeze(-1), 0)

        def loss(self, context, actions, targets, *, context_mask=None, action_mask=None):
            predictions = self(context, actions, targets, context_mask=context_mask, action_mask=action_mask)
            mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device) if action_mask is None else action_mask
            return (predictions[mask] - targets[mask]).square().mean()

        @torch.no_grad()
        def rollout(self, context, actions, *, context_mask=None, action_mask=None):
            """Autoregress H future latents, encoding observed context only once."""
            context, actions, cm, am, last = self._prepare(context, actions, context_mask, action_mask)
            encoded = self.backbone.encoder(
                inputs_embeds=self.context_adapter(context), attention_mask=cm, return_dict=True
            )
            previous = last.unsqueeze(1)
            predictions = []
            for step in range(actions.shape[1]):
                predicted = self._decode(encoded, cm, previous, actions[:, :step + 1], am[:, :step + 1])[:, -1:]
                predicted = predicted.masked_fill(~am[:, step:step + 1].unsqueeze(-1), 0)
                predictions.append(predicted)
                previous = torch.cat((previous, predicted), dim=1)
            return torch.cat(predictions, dim=1)

    return FrozenT5SequenceModel()
