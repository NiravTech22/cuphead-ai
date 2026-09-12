"""World model — latent dynamics with reward, value and hit-probability heads.

Owner: the ``world-model`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 4.

TD-MPC2-flavoured: deterministic latent, no decoder. Dropping reconstruction frees
capacity for the six-pixel bullet that actually kills you, instead of spending it
on background parallax.

    z_{t+1} = D(z_t, a_t)      latent dynamics
    r_hat   = R(z_t, a_t)      dense reward
    v_hat   = V(z_t)           value bootstrap past the horizon
    h_hat   = H(z_t, a_t)      P(hit within k steps)   <- the gated head
    pi_hat  = P(z_t)           policy prior seeding CEM

Acceptance is behavioural, never reconstructive:

    hit-prediction AUC >= 0.85 at k = 8 steps, on held-out replays, per boss.

A model with poor reconstruction and strong hit prediction plans well. The reverse
is never true.
"""

from .sequence_model import build_frozen_t5_sequence_model

from .knn_dynamics import Consequence, KNNDynamics
