"""
Per-subtask DIFFICULTY classifier (experiment 4, see experiments/subtask_router.md).

Rebuilt to use observed evidence instead of invented thresholds. The first version scored
call-count and file-count thresholds (6 calls, 4 targets, 2 files) that were guesses about
what correlates with difficulty. This version scores error_severity and spinning from
trajectory_signals.py (vendored from BerriAI/litellm PR #39976), which are observations of
task state -- a tool call actually erroring, actually repeating -- not proxies for it.

The distinction from subtask_type_classifier.py, which is the whole point of this experiment:
that one maps phase -> fixed slot, so every implement subtask gets the same model. This one
scores how hard the CURRENT subtask is, so a clean run of edits and a run full of repeated
failed edits are both "implement" and route differently.

Phase is a prior, not the answer. It sets a floor and a ceiling; evidence from the subtask's
own recent tool-call trajectory moves the tier inside that band:

  explore   SIMPLE..MEDIUM     reading is usually cheap
  implement MEDIUM..REASONING  editing is usually hard, but a clean run doesn't need REASONING
  verify    SIMPLE..COMPLEX    parsing output is cheap until the output is a failure

evidence = error_severity + spinning, both in [0, 1] fractions over the trajectory window.
The tier moves up one step when evidence > (1 - difficulty_sensitivity). One dial instead of
five invented constants: 0 means never move off the phase's starting tier, 1 means move on
any evidence at all. Provisional default 0.5, not yet calibrated against real traffic --
see experiments/subtask_router.md for the plan to set it from observed data instead of guessing.

No LLM call. Every signal is already in the payload the request carries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.router_strategy.complexity_router.config import ComplexityTier
from litellm.types.router import RoutingContext

sys.path.insert(0, str(Path(__file__).parent))

from subtask_signal import Phase, current_phase  # noqa: E402  # needs the sys.path insert above
from trajectory_signals import compute_trajectory_signals  # noqa: E402  # same reason

_LADDER: Final = (
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
)

# (floor, start, ceiling) as ladder indices.
_PHASE_BAND: Final = {
    Phase.EXPLORE: (0, 0, 1),
    Phase.IMPLEMENT: (1, 2, 3),
    Phase.VERIFY: (0, 0, 2),
}

_TRAJECTORY_WINDOW: Final = 12
_DEFAULT_SENSITIVITY: Final = 0.5


class PhaseDifficultyClassifier:
    """Scores the difficulty of the subtask happening now, from observed trajectory evidence."""

    def __init__(self, difficulty_sensitivity: float = _DEFAULT_SENSITIVITY) -> None:
        if not 0.0 <= difficulty_sensitivity <= 1.0:
            raise ValueError(f"difficulty_sensitivity must be in [0, 1], got {difficulty_sensitivity}")
        self.difficulty_sensitivity: Final = difficulty_sensitivity

    async def classify(self, context: RoutingContext) -> str | None:
        try:
            messages: Final = context.structured_messages
            phase, _boundary = current_phase(messages)
            if phase is None or phase not in _PHASE_BAND:
                return None

            floor, start, ceiling = _PHASE_BAND[phase]
            trajectory: Final = compute_trajectory_signals(messages, window=_TRAJECTORY_WINDOW)
            evidence: Final = trajectory.error_severity + trajectory.spinning
            # >= not >: at sensitivity=0.5, "half the recent calls are duplicates or errors" is
            # exactly the canonical case this classifier exists to catch, and a boundary that
            # excludes its own midpoint would silently swallow it.
            move_up: Final = evidence >= (1.0 - self.difficulty_sensitivity)
            index: Final = min(ceiling, start + 1) if move_up else start
            clamped: Final = max(floor, min(ceiling, index))
            tier: Final = _LADDER[clamped]

            verbose_proxy_logger.info(
                "PhaseDifficultyClassifier: phase=%s tier=%s (start=%s, band=%s..%s) "
                "evidence=%.3f (error_severity=%.3f spinning=%.3f) sensitivity=%.2f observed_calls=%s",
                phase.value,
                tier.value,
                _LADDER[start].value,
                _LADDER[floor].value,
                _LADDER[ceiling].value,
                evidence,
                trajectory.error_severity,
                trajectory.spinning,
                self.difficulty_sensitivity,
                trajectory.observed_calls,
            )
            return tier.value
        except Exception as e:  # noqa: BLE001  # a classifier must never fail the request
            verbose_proxy_logger.error("PhaseDifficultyClassifier failed: %s", e, exc_info=True)
            return None


# get_instance_fn (litellm/proxy/types_utils/utils.py) is a plain getattr on the module: it
# never instantiates, unlike the guardrail loader. classifier_plugin must therefore name a
# module-level instance, not the class.
phase_difficulty_classifier = PhaseDifficultyClassifier()
