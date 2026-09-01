"""LLM strategist — loadout, phase priors, post-mortems, routing, experiments.

Owner: the ``architect`` agent (with ``qa`` holding the veto).
See ``docs/SPEEDRUN_PLAN.md`` section 7.

Five jobs, none of them in the control loop:

1. loadout selection per boss, from failure memory;
2. structured phase policy priors (``planner.objective.PhasePrior``);
3. failure post-mortems over event traces, producing hypotheses;
4. route and category strategy across bosses;
5. experiment authoring — configs, curricula, ablations.

Every output is typed, schema-validated and clamped before it reaches the planner,
and carries a numeric prediction (``expected_ttk_delta_s``) that the system scores
it on. That prediction record is what turns "the LLM suggested something" into a
calibration curve you can trust.

    The strategist proposes. The evaluation harness disposes.

Operating rate is 0.02-1 Hz, offline, between attempts. The LLM is never in the
60 Hz loop — no exceptions, not even for debugging.
"""
