"""
Subtask-scoped learning (experiment 5, see experiments/subtask_memory.md).

Combines experiment 1 (learned coverage) with experiment 2 (subtask boundaries): learning is
extracted per SUBTASK rather than per whole ask, and retrieval is keyed on what the CURRENT
subtask is doing rather than on the session's opening ask.

Three configurable models, because they have genuinely different latency budgets:
  - `embedding_model`   retrieval, on the hot path, must be fast
  - `extraction_model`  distills a finished subtask into a learning, runs in the background
                        after the response is already streaming, so it can be slower/stronger
  - (the tier decision itself lives in subtask_memory_classifier.py, configured separately)

Retrieval is embeddings + cosine similarity, replacing an earlier word-overlap version that
was demonstrably broken: it matched on the phase name and tool name ("implement", "edit")
while missing the one token that identified the work, because the descriptor held it as JSON
(`{"file_path": "auth.py"}`) and the stored learning held it bare (`auth.py`).

Read path (async_pre_call_hook): embed the current subtask, cosine-rank stored learnings,
inject the top matches into the current turn, and record the top similarity into request
metadata. The classifier reads that on the same request, because pre_call_hook mutates the
same `data` dict route_request later reads (common_request_processing.py:2026 before :2385).

Write path (async_post_call_success_hook): when the phase detector says a NEW subtask just
started, the one that ended is a complete unit worth learning from. Extraction is fired as a
background task so it never sits in the response path.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral
from litellm.utils import token_counter

sys.path.insert(0, str(Path(__file__).parent))

from learning_retrieval import cosine, rank_candidates  # noqa: E402  # needs the sys.path insert above
from subtask_signal import (  # noqa: E402  # same reason
    Phase,
    ToolCall,
    extract_tool_calls,
    replay_online,
)

# Defaults to /tmp so the guardrail constructs anywhere; Render sets the env var to the
# persistent disk. A hardcoded /var/data default kills the proxy at boot on any machine
# without that mount, which is every machine except the deploy.
_STORE_DIR: Final = Path(os.environ.get("SUBTASK_MEMORY_STORE_DIR", "/tmp/subtask_memory"))

_DEFAULT_EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
_DEFAULT_EXTRACTION_MODEL: Final = "anthropic/claude-sonnet-5"
_DEFAULT_EMBEDDING_TIMEOUT_MS: Final = 2000
_DEFAULT_EXTRACTION_TIMEOUT_MS: Final = 30000

_MAX_CANDIDATES: Final = 3
_MIN_CALLS_TO_LEARN: Final = 2
_RESULT_CHARS_SHOWN: Final = 400


class _Learning(BaseModel):
    """What gets distilled out of a finished subtask. Deliberately procedural: the point is
    something a cheaper model can follow next time, not a summary of what happened."""

    summary: str = Field(description="One line naming the kind of work this was.")
    procedure: str = Field(description="The reusable steps, concrete enough to follow without the original context.")
    gotchas: str = Field(default="", description="Anything that went wrong or was surprising. Empty if nothing did.")
    worth_keeping: bool = Field(
        description="False if this subtask was trivial, failed, or teaches nothing reusable. "
        "Be strict: a store full of noise is worse than an empty one."
    )


def _subtask_text(calls: tuple[ToolCall, ...], phase: Phase) -> str:
    """Natural-language description of the subtask, for embedding. Values are pulled out of
    the tool arguments rather than left as raw JSON, so a path embeds as a path."""
    parts: list[str] = []  # mutable-ok: built once by one forward scan
    for call in calls:
        try:
            args = json.loads(call.detail) if call.detail else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        values = " ".join(str(v) for v in args.values()) if isinstance(args, dict) else ""
        parts.append(f"{call.name} {values}".strip())
    return f"{phase.value} phase: " + "; ".join(parts)


class SubtaskMemoryGuardrail(CustomGuardrail):
    """Retrieves and extracts learnings at subtask granularity, embeddings for retrieval and
    a configurable LLM for extraction."""

    def __init__(
        self,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        extraction_model: str = _DEFAULT_EXTRACTION_MODEL,
        embedding_timeout_ms: int = _DEFAULT_EMBEDDING_TIMEOUT_MS,
        extraction_timeout_ms: int = _DEFAULT_EXTRACTION_TIMEOUT_MS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_model: Final = embedding_model
        self.extraction_model: Final = extraction_model
        self.embedding_timeout_ms: Final = embedding_timeout_ms
        self.extraction_timeout_ms: Final = extraction_timeout_ms
        self.store_dir: Final[Path] = _STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        try:
            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                return data

            calls, phase = self._current_subtask(messages)
            if phase is None or not calls:
                return data

            query: Final = _subtask_text(calls, phase)
            matches = await self._retrieve(query)

            metadata_key = "litellm_metadata" if "litellm_metadata" in data else "metadata"
            metadata = data.setdefault(metadata_key, {})
            if not isinstance(metadata, dict):
                return data

            top_similarity = matches[0][0] if matches else 0.0
            metadata["subtask_memory_similarity"] = round(top_similarity, 4)
            metadata["subtask_memory_phase"] = phase.value
            metadata["subtask_memory_match_count"] = len(matches)

            if matches:
                injected = self._inject(messages, tuple(text for _score, text in matches))
                if injected is not None:
                    injected_messages, injected_text = injected
                    data["messages"] = injected_messages
                    metadata["subtask_memory_injected_tokens"] = token_counter(text=injected_text)

            verbose_proxy_logger.info(
                "SubtaskMemoryGuardrail: phase=%s matches=%d top_similarity=%.3f",
                phase.value,
                len(matches),
                top_similarity,
            )
            return data
        except Exception as e:  # noqa: BLE001  # a guardrail must never fail the request
            verbose_proxy_logger.error("SubtaskMemoryGuardrail read path error: %s", e)
            return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> None:
        try:
            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                return

            completed = self._just_completed_subtask(messages)
            if completed is None:
                return
            calls, phase = completed
            if len(calls) < _MIN_CALLS_TO_LEARN:
                return

            # Off the response path entirely: the client is already being written to, so the
            # extraction model can be slower and stronger than anything on the hot path.
            asyncio.create_task(  # noqa: RUF006  # fire-and-forget by design, matches proxy/utils.py
                self._extract_and_store(calls, phase, self._recent_results(messages))
            )
        except Exception as e:  # noqa: BLE001  # a guardrail must never fail the request
            verbose_proxy_logger.error("SubtaskMemoryGuardrail write path error: %s", e)

    def _current_subtask(self, messages: list[dict[str, Any]]) -> tuple[tuple[ToolCall, ...], Phase | None]:
        calls: Final = extract_tool_calls(messages)
        boundaries: Final = replay_online(calls)
        if not boundaries:
            return (), None
        latest: Final = boundaries[-1]
        return calls[latest.started_at :], latest.to_phase

    def _just_completed_subtask(self, messages: list[dict[str, Any]]) -> tuple[tuple[ToolCall, ...], Phase] | None:
        """The subtask that ended, but only on the turn a new one actually began.

        Returning it on every turn would re-extract the same in-progress work repeatedly; a
        boundary is the one moment a subtask is both complete and freshly known.
        """
        calls: Final = extract_tool_calls(messages)
        boundaries: Final = replay_online(calls)
        if len(boundaries) < 2:
            return None
        latest: Final = boundaries[-1]
        if latest.detected_at != len(calls) - 1:
            return None
        previous: Final = boundaries[-2]
        completed_calls: Final = calls[previous.started_at : latest.started_at]
        if not completed_calls:
            return None
        return completed_calls, previous.to_phase

    def _recent_results(self, messages: list[dict[str, Any]]) -> str:
        """Tool results from the tail of the conversation, so extraction sees outcomes and not
        just the calls that produced them."""
        lines: list[str] = []  # mutable-ok: built once by one forward scan
        for message in messages[-30:]:
            content = message.get("content")
            if message.get("role") == "tool":
                lines.append(str(content)[:_RESULT_CHARS_SHOWN])
            elif message.get("role") == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        err = "[ERROR] " if block.get("is_error") else ""
                        lines.append(err + str(block.get("content", ""))[:_RESULT_CHARS_SHOWN])
        return "\n".join(lines[-10:])

    async def _embed(self, text: str) -> list[float] | None:
        try:
            response = await asyncio.wait_for(
                litellm.aembedding(model=self.embedding_model, input=[text]),
                timeout=self.embedding_timeout_ms / 1000,
            )
            return list(response.data[0]["embedding"])
        except Exception as e:  # noqa: BLE001  # retrieval is best-effort; no learnings beats a failed request
            verbose_proxy_logger.error("SubtaskMemoryGuardrail embed failed: %s", e)
            return None

    async def _retrieve(self, query: str) -> list[tuple[float, str]]:
        """Top learnings, best first: see learning_retrieval.rank_candidates for the ranking
        rule (a noise floor plus a gap below the best hit, not a single fixed cutoff).

        Vectors are stored alongside each learning at write time, so retrieval costs exactly
        one embedding call regardless of how many learnings exist.
        """
        query_vector: Final = await self._embed(query)
        if query_vector is None:
            return []
        scored: list[tuple[float, str]] = []  # mutable-ok: append-only, ranked once below
        for path in self.store_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                score = cosine(query_vector, record["vector"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            scored.append((score, record["text"]))
        return rank_candidates(scored, limit=_MAX_CANDIDATES)

    def _inject(
        self, messages: list[dict[str, Any]], learnings: tuple[str, ...]
    ) -> Optional[tuple[list[dict[str, Any]], str]]:
        """Prepended to the current turn, not the system prompt: the system prompt is the
        most-cached prefix in the session, so injecting there invalidates the cache every
        turn. The current turn sits after every cache breakpoint."""
        injected_text: Final = "[Learned from similar subtasks]\n" + "\n\n".join(learnings) + "\n\n[Current work]\n"
        copied: Final = [dict(msg) for msg in messages]
        for msg in reversed(copied):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{injected_text}{content}"
                return copied, injected_text
            if isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = f"{injected_text}{block.get('text', '')}"
                        msg["content"] = blocks
                        return copied, injected_text
        return None

    async def _extract_and_store(self, calls: tuple[ToolCall, ...], phase: Phase, results: str) -> None:
        """Distill a finished subtask into a reusable learning, then store it with its vector.

        Background task: the response is already on its way to the client, so this can use a
        slower, stronger model than anything on the hot path.
        """
        try:
            work: Final = _subtask_text(calls, phase)
            prompt: Final = (
                f"A coding agent just finished a {phase.value} subtask.\n\n"
                f"Tool calls it made:\n{work}\n\n"
                f"Tool results it saw:\n{results or '(none captured)'}\n\n"
                f"Extract a reusable learning: what kind of work this was, and the procedure a "
                f"different model could follow to do the same kind of work next time. Set "
                f"worth_keeping=false if this was trivial, failed, or teaches nothing reusable."
            )
            response: Final = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.extraction_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=_Learning,
                    num_retries=0,
                ),
                timeout=self.extraction_timeout_ms / 1000,
            )
            content: Final = response.choices[0].message.content
            if not content:
                return
            learning: Final = _Learning.model_validate_json(content)
            if not learning.worth_keeping:
                verbose_proxy_logger.info(
                    "SubtaskMemoryGuardrail: %s subtask judged not worth keeping", phase.value
                )
                return

            text: Final = (
                f"[{phase.value}] {learning.summary}\n\nProcedure:\n{learning.procedure}"
                + (f"\n\nGotchas:\n{learning.gotchas}" if learning.gotchas else "")
            )
            # Embedded on the learning's own text, which is what a future query is compared
            # against; embedding the raw tool calls instead would rank on plumbing.
            vector: Final = await self._embed(text)
            if vector is None:
                return

            timestamp: Final = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path: Final = self.store_dir / f"{timestamp}_{phase.value}.json"
            path.write_text(
                json.dumps({"phase": phase.value, "text": text, "vector": vector}),
                encoding="utf-8",
            )
            verbose_proxy_logger.info(
                "SubtaskMemoryGuardrail: stored %s learning: %s", phase.value, learning.summary
            )
        except Exception as e:  # noqa: BLE001  # background task, must not raise into the loop
            verbose_proxy_logger.error("SubtaskMemoryGuardrail extraction failed: %s", e)
