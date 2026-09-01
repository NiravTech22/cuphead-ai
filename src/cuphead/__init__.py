"""CUPHEAD-AI — autonomous Cuphead speedrunning agent.

Eleven layers, three timescales. See ``CLAUDE.md`` for the rules and
``docs/SPEEDRUN_PLAN.md`` for the technical breakdown.

Nothing here imports numpy or torch at module load. The orchestration, control,
event and evaluation layers are stdlib-only so the project is inspectable and
testable before any environment is set up.
"""

__version__ = "0.1.0"
