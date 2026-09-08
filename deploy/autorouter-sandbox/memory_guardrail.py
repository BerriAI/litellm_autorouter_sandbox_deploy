"""
MemoryGuardrail for the learned-coverage routing experiment (deploy/autorouter-sandbox/experiments/memory.md).

Read path (async_pre_call_hook, mode=pre_call): retrieves learnings for the current ask and
injects them into the current user turn, before the request reaches the router, so the
built-in classifier scores the augmented ask directly. No custom classifier here, deliberately:
this experiment's whole premise is that an ask reading "here is how this was done before"
scores as easier to whatever classifier reads it next, without a second mechanism.

Write path (async_post_call_success_hook, mode=post_call): extracts a learning from turns that
routed to a strong tier, after the response is already on its way back.

Retrieval is embeddings + cosine similarity, extraction is a configurable LLM producing a
procedural learning with a worth_keeping filter -- the same mechanism as
subtask_memory_guardrail.py (experiment 5), applied to the whole ask instead of a subtask. This
used to be word-overlap retrieval plus verbatim response storage, replaced for the same reason
experiment 5 was: word overlap matched on structural noise while missing the identifying
token, and verbatim storage never distinguishes a real learning from "agents tend to find
something, always" (see the Agno article this experiment is based on). Keeping experiment 1 on
the old mechanism while experiment 5 got the new one would have made any 1-vs-5 comparison
confound "ask-scoped vs subtask-scoped" with "keyword vs embeddings", which is not the question
either experiment is trying to answer.

Registered twice in proxy_config.yaml under the same guardrail_name and different modes, same
pattern as the pre/post custom guardrail examples in
litellm/proxy/example_config_yaml/otel_test_config.yaml. Scoped to moe-learning-router only via
that model's litellm_params.guardrails.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

from pydantic import BaseModel, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.complexity_router.complexity_router import (
    _extract_current_ask_and_system_prompt,
)
from litellm.types.utils import CallTypesLiteral
from litellm.utils import token_counter

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent))

from learning_retrieval import cosine, rank_candidates  # noqa: E402  # needs the sys.path insert above

_STRONG_TIERS: Final = frozenset({"COMPLEX", "REASONING"})
_STORE_DIR: Final = Path(os.environ.get("MEMORY_STORE_DIR", "/tmp/memory"))

_DEFAULT_EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
_DEFAULT_EXTRACTION_MODEL: Final = "anthropic/claude-sonnet-5"
_DEFAULT_EMBEDDING_TIMEOUT_MS: Final = 2000
_DEFAULT_EXTRACTION_TIMEOUT_MS: Final = 30000

_MAX_CANDIDATES: Final = 3


class _Learning(BaseModel):
    """What gets distilled out of a strong-tier turn. Deliberately procedural: the point is
    something a cheaper model can follow next time, not a summary of what happened."""

    summary: str = Field(description="One line naming the kind of task this was.")
    procedure: str = Field(description="The reusable steps, concrete enough to follow without the original context.")
    gotchas: str = Field(default="", description="Anything that went wrong or was surprising. Empty if nothing did.")
    worth_keeping: bool = Field(
        description="False if this ask was trivial, failed, or teaches nothing reusable. "
        "Be strict: a store full of noise is worse than an empty one."
    )


class MemoryGuardrail(CustomGuardrail):
    """Injects learned context into requests and extracts learnings from strong-tier turns,
    embeddings for retrieval and a configurable LLM for extraction."""

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

            current_ask, _system_prompt = _extract_current_ask_and_system_prompt(messages)
            if not current_ask:
                return data

            matches = await self._retrieve(current_ask)

            metadata_key = "litellm_metadata" if "litellm_metadata" in data else "metadata"
            metadata = data.setdefault(metadata_key, {})
            if not isinstance(metadata, dict):
                return data

            top_similarity = matches[0][0] if matches else 0.0
            # Not read by any routing decision on this router (see module docstring): the
            # built-in classifier reads the augmented ask directly. Recorded for visibility
            # and for a future classifier that might want it, same as
            # subtask_memory_guardrail.py's equivalent field.
            metadata["memory_similarity"] = round(top_similarity, 4)
            metadata["memory_match_count"] = len(matches)

            if matches:
                injection_result = self._inject(messages, tuple(text for _score, text in matches))
                if injection_result is not None:
                    injected_messages, injected_text = injection_result
                    data["messages"] = injected_messages
                    # Savings baseline prices the whole request against the counterfactual
                    # model, so without this the baseline gets billed for tokens it would
                    # never have needed: see experiments/memory.md, "Injected tokens inflate
                    # the savings baseline".
                    metadata["memory_injected_tokens"] = token_counter(text=injected_text)

            verbose_proxy_logger.info(
                "MemoryGuardrail: matches=%d top_similarity=%.3f", len(matches), top_similarity
            )
            return data
        except Exception as e:
            verbose_proxy_logger.error("MemoryGuardrail read path error: %s", e)
            return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> None:
        try:
            metadata_key = "litellm_metadata" if "litellm_metadata" in data else "metadata"
            metadata = data.get(metadata_key)
            if not isinstance(metadata, dict):
                return

            routing_decision = metadata.get("routing_decision")
            if not isinstance(routing_decision, dict) or routing_decision.get("tier") not in _STRONG_TIERS:
                return

            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                return

            current_ask, _system_prompt = _extract_current_ask_and_system_prompt(messages)
            if not current_ask:
                return

            response_text = self._extract_response_text(response)
            if not response_text:
                return

            # Off the response path entirely: the client is already being written to, so
            # extraction can use a slower, stronger model than anything on the hot path.
            asyncio.create_task(  # noqa: RUF006  # fire-and-forget by design, matches proxy/utils.py
                self._extract_and_store(current_ask, response_text, routing_decision["tier"])
            )
        except Exception as e:
            verbose_proxy_logger.error("MemoryGuardrail write path error: %s", e)

    async def _embed(self, text: str) -> list[float] | None:
        try:
            response = await asyncio.wait_for(
                litellm.aembedding(model=self.embedding_model, input=[text]),
                timeout=self.embedding_timeout_ms / 1000,
            )
            return list(response.data[0]["embedding"])
        except Exception as e:
            verbose_proxy_logger.error("MemoryGuardrail embed failed: %s", e)
            return None

    async def _retrieve(self, ask: str) -> list[tuple[float, str]]:
        """Top learnings, best first: see learning_retrieval.rank_candidates for the ranking
        rule (a noise floor plus a gap below the best hit, not a single fixed cutoff).

        Vectors are stored alongside each learning at write time, so retrieval costs exactly
        one embedding call regardless of how many learnings exist.
        """
        query_vector: Final = await self._embed(ask)
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
        """Prepend learnings to the last user turn, not the system prompt: the system
        prompt is Claude Code's most-cached prefix, so injecting there would invalidate
        the cache on every turn. The current turn sits after every cache breakpoint.

        Returns the injected text alongside the messages so the caller can price it out
        of the savings baseline: see experiments/memory.md, "Injected tokens inflate the
        savings baseline"."""
        injection = "\n\n".join(learnings)
        injected_text = f"[Learned context]\n{injection}\n\n[Current ask]\n"
        messages_copy = [dict(msg) for msg in messages]

        for msg in reversed(messages_copy):
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{injected_text}{content}"
                return messages_copy, injected_text

            if isinstance(content, list):
                new_content = [dict(block) if isinstance(block, dict) else block for block in content]
                for block in new_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = f"{injected_text}{block.get('text', '')}"
                        msg["content"] = new_content
                        return messages_copy, injected_text

        return None

    def _extract_response_text(self, response: Any) -> Optional[str]:
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    return content if isinstance(content, str) else None
            return None

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            return content if isinstance(content, str) else None

        return None

    async def _extract_and_store(self, ask: str, response: str, tier: str) -> None:
        """Distill a strong-tier turn into a reusable learning, then store it with its vector.

        Background task: the response is already on its way to the client, so this can use a
        slower, stronger model than anything on the hot path.
        """
        try:
            prompt: Final = (
                f"A coding agent just completed a task on the {tier} tier.\n\n"
                f"The ask:\n{ask}\n\n"
                f"Its response:\n{response[:4000]}\n\n"
                f"Extract a reusable learning: what kind of task this was, and the procedure a "
                f"different model could follow to do the same kind of task next time. Set "
                f"worth_keeping=false if this was trivial, failed, or teaches nothing reusable."
            )
            llm_response: Final = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.extraction_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=_Learning,
                    num_retries=0,
                ),
                timeout=self.extraction_timeout_ms / 1000,
            )
            content: Final = llm_response.choices[0].message.content
            if not content:
                return
            learning: Final = _Learning.model_validate_json(content)
            if not learning.worth_keeping:
                verbose_proxy_logger.info("MemoryGuardrail: %s-tier ask judged not worth keeping", tier)
                return

            text: Final = (
                f"[{tier}] {learning.summary}\n\nProcedure:\n{learning.procedure}"
                + (f"\n\nGotchas:\n{learning.gotchas}" if learning.gotchas else "")
            )
            # Embedded on the learning's own text, which is what a future ask is compared
            # against; embedding the raw ask+response instead would rank on phrasing.
            vector: Final = await self._embed(text)
            if vector is None:
                return

            timestamp: Final = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path: Final = self.store_dir / f"{timestamp}_{tier}.json"
            path.write_text(
                json.dumps({"tier": tier, "text": text, "vector": vector}),
                encoding="utf-8",
            )
            verbose_proxy_logger.info("MemoryGuardrail: stored %s-tier learning: %s", tier, learning.summary)
        except Exception as e:
            verbose_proxy_logger.error("MemoryGuardrail extraction failed: %s", e)
