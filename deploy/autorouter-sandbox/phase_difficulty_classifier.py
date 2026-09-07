"""
Per-subtask DIFFICULTY classifier (experiment 4, see experiments/subtask_router.md).

The distinction from subtask_type_classifier.py, which is the whole point of this experiment:
that one maps phase -> fixed slot, so every implement subtask gets the same model. This one
scores how hard the CURRENT subtask is, so a one-line typo fix and a cross-file refactor are
both "implement" and route differently.

Phase is a prior, not the answer. It sets a floor and a ceiling, then evidence from the
subtask's own tool activity moves the tier inside that band:

  explore   SIMPLE..MEDIUM     reading is usually cheap, but wide reading is a real search
  implement MEDIUM..REASONING  editing is usually hard, but a one-file touch-up is not
  verify    SIMPLE..COMPLEX    parsing output is cheap until the output is a failure

Signals are counted over the current subtask only (calls since the last confirmed boundary),
because the whole premise is that the session's opening ask stopped being informative many
turns ago.

No LLM call. The signals below are free, already in the payload, and the router's own
classifier costs ~3s of latency per turn; adding a second model call to save model cost is
the wrong trade until the free version is shown to be too coarse.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.router_strategy.complexity_router.config import ComplexityTier
from litellm.types.router import RoutingContext

sys.path.insert(0, str(Path(__file__).parent))

from subtask_signal import (  # noqa: E402  # needs the sys.path insert above
    Phase,
    ToolCall,
    classify_tool_call,
    extract_tool_calls,
    replay_online,
)

_LADDER: Final = (
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
)

# (floor, start, ceiling) as ladder indices. `start` is where a subtask with no other
# evidence lands; signals move it within [floor, ceiling].
_PHASE_BAND: Final = {
    Phase.EXPLORE: (0, 0, 1),
    Phase.IMPLEMENT: (1, 2, 3),
    Phase.VERIFY: (0, 0, 2),
}

_FAILURE_MARKERS: Final = (
    "traceback",
    "error:",
    "exception",
    "failed",
    "assertionerror",
    "syntaxerror",
    "typeerror",
    "no such file",
    "command not found",
    "fatal:",
)

_LONG_SUBTASK_CALLS: Final = 6
_WIDE_EXPLORE_FILES: Final = 4


def _subtask_calls(messages: Sequence[dict[str, object]]) -> tuple[tuple[ToolCall, ...], Phase | None]:
    """Tool calls belonging to the current subtask, i.e. since the last confirmed boundary."""
    calls: Final = extract_tool_calls(messages)
    boundaries: Final = replay_online(calls)
    if not boundaries:
        return (), None
    latest: Final = boundaries[-1]
    return calls[latest.started_at :], latest.to_phase


def _distinct_targets(calls: Sequence[ToolCall]) -> int:
    """How many distinct files/targets this subtask has touched, read out of tool arguments."""
    targets: set[str] = set()  # mutable-ok: set built by one scan, membership is the point
    for call in calls:
        try:
            args = json.loads(call.detail) if call.detail else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        for key in ("file_path", "path", "notebook_path", "pattern"):
            value = args.get(key)
            if isinstance(value, str) and value:
                targets.add(value)
    return len(targets)


def _recent_output_text(messages: Sequence[dict[str, object]], limit: int = 3) -> str:
    """Text of the most recent tool results, where a failure would show up."""
    texts: list[str] = []  # mutable-ok: append-only accumulator over one reverse scan
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content.lower())
        if len(texts) >= limit:
            break
    return "\n".join(texts)


class PhaseDifficultyClassifier:
    """Scores the difficulty of the subtask happening now, rather than of the opening ask."""

    async def classify(self, context: RoutingContext) -> str | None:
        try:
            messages: Final = context.structured_messages
            calls, phase = _subtask_calls(messages)
            if phase is None or phase not in _PHASE_BAND:
                return None

            floor, start, ceiling = _PHASE_BAND[phase]
            index = start  # rebind-ok: accumulates signal adjustments before clamping
            reasons: list[str] = []  # mutable-ok: append-only, for the decision log

            if len(calls) >= _LONG_SUBTASK_CALLS:
                index += 1
                reasons.append(f"long-subtask({len(calls)}calls)")

            targets: Final = _distinct_targets(calls)
            if phase is Phase.EXPLORE and targets >= _WIDE_EXPLORE_FILES:
                index += 1
                reasons.append(f"wide-search({targets}targets)")
            if phase is Phase.IMPLEMENT and targets >= 2:
                index += 1
                reasons.append(f"multi-file-edit({targets}files)")
            if phase is Phase.IMPLEMENT and targets <= 1 and len(calls) <= 2:
                index -= 1
                reasons.append("single-small-edit")

            output: Final = _recent_output_text(messages)
            if any(marker in output for marker in _FAILURE_MARKERS):
                index += 1
                reasons.append("failure-in-output")

            clamped: Final = max(floor, min(ceiling, index))
            tier: Final = _LADDER[clamped]

            verbose_proxy_logger.info(
                "PhaseDifficultyClassifier: phase=%s subtask_calls=%s tier=%s (start=%s -> %s, band=%s..%s) signals=[%s]",
                phase.value,
                len(calls),
                tier.value,
                _LADDER[start].value,
                tier.value,
                _LADDER[floor].value,
                _LADDER[ceiling].value,
                ",".join(reasons) or "none",
            )
            return tier.value
        except Exception as e:  # noqa: BLE001  # a classifier must never fail the request
            verbose_proxy_logger.error("PhaseDifficultyClassifier failed: %s", e, exc_info=True)
            return None


# See subtask_type_classifier.py: the classifier_plugin loader does not instantiate, so this
# must be a module-level instance.
phase_difficulty_classifier = PhaseDifficultyClassifier()
