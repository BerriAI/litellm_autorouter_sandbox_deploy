"""
Subtask TYPE classifier (experiment 3, see experiments/subtask_router.md).

Maps the detected work phase to a tier whose model is chosen for that KIND of work, not for
its difficulty: a cheap long-context model for reading, a coding model for editing, a fast
model for interpreting test output.

Tiers here are the built-in names used as arbitrary slots. They carry no severity meaning on
this router, and nothing reads them as a difficulty ladder: SIMPLE is "the explore model",
COMPLEX is "the implement model", MEDIUM is "the verify model". `tier_definitions` would name
them honestly, but it also bans escalation_keywords, so the built-in names are the cheaper
trade while the manual LITELLM ESCALATE override stays available.

Pairs with phase_difficulty_classifier.py, which answers the different question of how hard
the current subtask is.
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


class SubtaskTypeClassifier:
    """Routes on what KIND of work the current subtask is."""

    _PHASE_TO_SLOT: Final = {
        Phase.EXPLORE: ComplexityTier.SIMPLE,
        Phase.IMPLEMENT: ComplexityTier.COMPLEX,
        Phase.VERIFY: ComplexityTier.MEDIUM,
    }

    async def classify(self, context: RoutingContext) -> str | None:
        try:
            phase, boundary = current_phase(context.structured_messages)
            slot: Final = self._PHASE_TO_SLOT.get(phase) if phase is not None else None
            if slot is None:
                # No confirmed phase (opening turn) or a neutral phase: decline and let
                # classifier_fallback decide rather than guessing a specialist.
                return None

            verbose_proxy_logger.info(
                "SubtaskTypeClassifier: phase=%s slot=%s detected_at=%s lag=%s",
                phase.value,
                slot.value,
                boundary.detected_at if boundary else -1,
                boundary.lag if boundary else -1,
            )
            return slot.value
        except Exception as e:  # noqa: BLE001  # a classifier must never fail the request
            verbose_proxy_logger.error("SubtaskTypeClassifier failed: %s", e, exc_info=True)
            return None


# get_instance_fn (litellm/proxy/types_utils/utils.py) is a plain getattr on the module: it
# never instantiates, unlike the guardrail loader. classifier_plugin must therefore name a
# module-level instance, not the class.
subtask_type_classifier = SubtaskTypeClassifier()
