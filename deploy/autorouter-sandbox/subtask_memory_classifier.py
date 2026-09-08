"""
Subtask-memory-aware difficulty classifier (experiment 5, see experiments/subtask_memory.md).

LLM-based, replacing an earlier arithmetic version (phase band + trajectory-evidence
threshold). One judge call per request, given the phase and recent tool activity, decides
the tier directly. Chosen over the arithmetic version because a fixed formula over
free-text tool output is exactly the kind of judgment call an LLM is suited to and a formula
is not: "is this subtask actually stuck" or "is this actually hard" doesn't reduce to a
threshold on two fractions.

Runs on every classified request, same as the router's own built-in `classifier_type: llm`
path (moe-router's Haiku call) -- this replaces that call, it does not add to it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.types.router import RoutingContext

sys.path.insert(0, str(Path(__file__).parent))

from subtask_signal import Phase, current_phase  # noqa: E402  # needs the sys.path insert above

_DEFAULT_MODEL: Final = "anthropic/claude-haiku-4-5"
# 3000 was too tight in practice: a structured-output call intermittently timed out and fell
# back to default_model. This is on the hot path, so it cannot go much higher without being
# felt as latency; on timeout the classifier returns None and the router falls back safely.
_DEFAULT_TIMEOUT_MS: Final = 6000
_RECENT_CALLS_SHOWN: Final = 8
_RESPONSE_CHARS_SHOWN: Final = 300

_TierName = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]

_PHASE_BAND_DESCRIPTION: Final = {
    Phase.EXPLORE: "explore work (reading/searching). Usually SIMPLE or MEDIUM; never COMPLEX or REASONING.",
    Phase.IMPLEMENT: "implement work (editing code). Usually MEDIUM or COMPLEX; REASONING only if genuinely stuck or the change is architecturally risky.",
    Phase.VERIFY: "verify work (running/reading test or build output). Usually SIMPLE; COMPLEX only if failures need real diagnosis.",
}


class _DifficultyVerdict(BaseModel):
    tier: _TierName = Field(description="The tier this subtask should route to.")
    reasoning: str = Field(description="One sentence: why this tier, referencing what you actually saw.")


def _recent_tool_summary(messages: list[dict]) -> str:
    """Plain-text summary of the last few tool calls and their results, for the judge prompt.
    Reads both wire shapes directly rather than importing trajectory_signals, since this only
    needs display text, not the signature/error-flag structure that module computes."""
    lines: list[str] = []  # mutable-ok: built once by one forward scan, never mutated after
    for message in messages[-40:]:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    lines.append(f"CALL {block.get('name')}({block.get('input')})")
        elif role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                fn = call.get("function", {})
                lines.append(f"CALL {fn.get('name')}({fn.get('arguments')})")
        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = str(block.get("content", ""))[:_RESPONSE_CHARS_SHOWN]
                    err = " [ERROR]" if block.get("is_error") else ""
                    lines.append(f"RESULT{err}: {text}")
        elif role == "tool":
            text = str(content)[:_RESPONSE_CHARS_SHOWN]
            lines.append(f"RESULT: {text}")
    return "\n".join(lines[-_RECENT_CALLS_SHOWN * 2 :]) or "(no tool activity yet)"


class SubtaskMemoryClassifier:
    """Difficulty per subtask, judged by an LLM given the phase and recent tool trajectory,
    plus whatever learnings the memory guardrail already injected into this same turn."""

    def __init__(self, model: str = _DEFAULT_MODEL, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> None:
        self.model: Final = model
        self.timeout_ms: Final = timeout_ms

    async def classify(self, context: RoutingContext) -> str | None:
        try:
            messages: Final = list(context.structured_messages)
            phase, _boundary = current_phase(messages)
            if phase is None or phase not in _PHASE_BAND_DESCRIPTION:
                return None

            similarity: Final = float(context.metadata.get("subtask_memory_similarity", 0.0))
            match_count: Final = int(context.metadata.get("subtask_memory_match_count", 0))
            summary: Final = _recent_tool_summary(messages)

            memory_line: Final = (
                f"{match_count} past learning(s) matched this subtask, top cosine similarity "
                f"{similarity:.2f}, and their procedures are ALREADY injected into the model's "
                f"prompt for this turn. Above ~0.6 that is a close match, so a cheaper tier is "
                f"reasonable; below ~0.45 treat it as weak and ignore it."
                if match_count
                else "No past learnings matched this subtask; nothing was injected."
            )

            prompt: Final = (
                f"Current subtask phase: {phase.value} -- {_PHASE_BAND_DESCRIPTION[phase]}\n\n"
                f"Memory: {memory_line}\n\n"
                f"Recent tool activity (most recent last):\n{summary}\n\n"
                f"Pick the tier this subtask should route to right now. Signs of being stuck "
                f"(repeated identical calls, errors, retries) outweigh memory: route up even if "
                f"a learning matched."
            )

            response: Final = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=_DifficultyVerdict,
                    timeout=self.timeout_ms / 1000,
                    num_retries=0,
                ),
                timeout=self.timeout_ms / 1000,
            )
            content: Final = response.choices[0].message.content
            if not content:
                return None
            verdict: Final = _DifficultyVerdict.model_validate_json(content)

            verbose_proxy_logger.info(
                "SubtaskMemoryClassifier: phase=%s matches=%d similarity=%.3f -> tier=%s (%s)",
                phase.value,
                match_count,
                similarity,
                verdict.tier,
                verdict.reasoning,
            )
            return verdict.tier
        except Exception as e:  # noqa: BLE001  # a classifier must never fail the request
            verbose_proxy_logger.error("SubtaskMemoryClassifier failed: %s", e, exc_info=True)
            return None


# get_instance_fn is a plain getattr on the module, so it cannot pass constructor arguments:
# `classifier_plugin` must name an already-built instance. To change the judge model, edit the
# line below (or add another named instance and point classifier_plugin at that one) rather
# than looking for a YAML key for it.
#
# Env var override so the model can be changed on the deploy without a code edit.
subtask_memory_classifier = SubtaskMemoryClassifier(
    model=os.environ.get("SUBTASK_CLASSIFIER_MODEL", _DEFAULT_MODEL),
    timeout_ms=int(os.environ.get("SUBTASK_CLASSIFIER_TIMEOUT_MS", _DEFAULT_TIMEOUT_MS)),
)
