# Frozen T5 latent sequencing

The optional sequence model defaults to `google/flan-t5-small`. It encodes observed
visual latent history and causally decodes action-conditioned future latents.
Both T5 stacks stay frozen and in evaluation mode; three linear adapters train.
No tokenizer is needed. Continuous latents enter through learned embeddings.

Install `pip install -r requirements-sequence.txt`. First construction downloads
pretrained weights. Pass a local model directory as `model_id` or inject an
already loaded `transformers.T5Model` with `backbone` to avoid a download.

```python
import torch
from cuphead.world_model import build_frozen_t5_sequence_model

model = build_frozen_t5_sequence_model(latent_dim=768, action_dim=5)
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=1e-3
)
# context: [batch, observed_steps, 768], ending at z_t
# actions: [batch, horizon, 5], starting at a_t
# targets: [batch, horizon, 768], starting at z_{t+1}
# Menu features: (*action.vector(), action.hold_frames / 60.0)
model.train()
optimizer.zero_grad()
loss = model.loss(context, actions, targets)
loss.backward()
optimizer.step()
future_latents = model.rollout(context, actions)
torch.save(model.state_dict(), "sequence_weights.pt")
```

Construct examples within a single episode, preserving frame/action alignment.
Teacher forcing shifts targets right, beginning with the last observed latent.
The decoder's causal attention prevents future actions or targets leaking into
earlier predictions. Rollout encodes context once and feeds predictions back.
Optional boolean `context_mask` and `action_mask` describe nonempty right-padded
sequences; padding is excluded from attention and loss. Defaults allow 64 context
steps and 20 predicted steps. Inputs must share the model's device and dtype.

Autograd stays enabled through frozen T5 during training so input adapters learn.
Language pretraining alone does not provide visual dynamics; adapters need data.
This is an offline experiment: the existing navigator keeps its current predictor.
Measure held-out autoregressive error, behavioral metrics and device latency
before using this model in navigation. No gameplay improvement is established.

Tests use a tiny random real T5 without downloading pretrained weights:
`python -m unittest discover -s tests -p test_sequence_model.py -v`.
They check freezing, input gradients, causal alignment, padding and rollout.
They skip when optional dependencies are absent.

API reference: [Hugging Face T5](https://huggingface.co/docs/transformers/model_doc/t5).
