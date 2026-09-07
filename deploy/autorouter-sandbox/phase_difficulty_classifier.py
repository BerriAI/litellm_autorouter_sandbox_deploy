"""
Phase-based difficulty classifier for experiment 4 (deploy/autorouter-sandbox/experiments/subtask_router.md).

ClassifierPlugin that classifies difficulty per detected phase/subtask rather than per user
message. Keeps built-in SIMPLE/MEDIUM/COMPLEX/REASONING tiers, reclassifies at phase
boundaries using the subtask's own tool activity instead of anchoring on the human ask.

Phase-to-tier mapping:
- explore: SIMPLE (Haiku) — file read/search is input-dominated, low reasoning
- implement: COMPLEX (Opus 5) — code editing is risky, proven on the codebase
- verify: MEDIUM (Sonnet 5) — test output parsing is structured, low reasoning
- other/no phase yet: fallback_tier (set in config, typically REASONING or COMPLEX)

Revalidate this mapping against real traffic escalations before shipping. First version logs
every tier decision for analysis.
"""

from __future__ import annotations

import sys
from typing import Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.router_strategy.complexity_router.config import ComplexityTier
from litellm.types.router import RoutingContext
from litellm.types.router import ClassifierPlugin as ClassifierPluginProtocol

# Inject subtask_signal into the path so we can import from it
sys.path.insert(0, __file__.rsplit("/", 1)[0])

try:
    from subtask_signal import current_phase, Phase
except ImportError as e:
    verbose_proxy_logger.error("PhaseDifficultyClassifier: could not import subtask_signal: %s", e)
    raise


class PhaseDifficultyClassifier:
    """ClassifierPlugin that routes based on detected subtask phase and maps phase to difficulty tier."""

    _PHASE_TO_TIER: Final = {
        Phase.EXPLORE: ComplexityTier.SIMPLE,
        Phase.IMPLEMENT: ComplexityTier.COMPLEX,
        Phase.VERIFY: ComplexityTier.MEDIUM,
        Phase.OTHER: None,  # Signals "no determined phase", falls to fallback
    }

    async def classify(self, context: RoutingContext) -> str | None:
        """
        Detect the current subtask phase and return the corresponding tier.

        Args:
            context: RoutingContext with structured_messages (full message history).

        Returns:
            Tier name (SIMPLE/MEDIUM/COMPLEX/REASONING) or None to fall back.
        """
        try:
            phase, boundary = current_phase(context.structured_messages)

            if phase is None:
                verbose_proxy_logger.debug(
                    "PhaseDifficultyClassifier: no confirmed phase yet (possibly first turn), "
                    "falling back to classifier_fallback"
                )
                return None

            tier: Final = self._PHASE_TO_TIER.get(phase)
            if tier is None:
                verbose_proxy_logger.warning(
                    "PhaseDifficultyClassifier: phase %s mapped to None tier, falling back",
                    phase.value if phase else "???",
                )
                return None

            # Log the decision for analysis: did this phase-to-tier mapping hold up?
            lag: Final = boundary.lag if boundary else -1
            verbose_proxy_logger.info(
                "PhaseDifficultyClassifier: phase=%s tier=%s boundary_at_index=%s lag=%s",
                phase.value,
                tier.value,
                boundary.detected_at if boundary else -1,
                lag,
            )

            return tier.value

        except Exception as e:
            verbose_proxy_logger.error("PhaseDifficultyClassifier: exception during classify: %s", e, exc_info=True)
            return None
