"""
MemoryGuardrail for the learned-coverage routing experiment (deploy/autorouter-sandbox/experiments/memory.md).

Read path (async_pre_call_hook, mode=pre_call): retrieves learnings for the current ask and
injects them into the current user turn, before the request reaches the router, so the
classifier scores the augmented ask directly.

Write path (async_post_call_success_hook, mode=post_call): extracts a learning from turns that
routed to a strong tier, after the response is already on its way back.

Registered twice in proxy_config.yaml under the same guardrail_name and different modes, same
pattern as the pre/post custom guardrail examples in
litellm/proxy/example_config_yaml/otel_test_config.yaml. Scoped to moe-router only via that
model's litellm_params.guardrails.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.complexity_router.complexity_router import (
    _extract_current_ask_and_system_prompt,
)
from litellm.types.utils import CallTypesLiteral

_STRONG_TIERS: Final = frozenset({"COMPLEX", "REASONING"})
_MIN_WORD_LEN: Final = 4
_MAX_CANDIDATES: Final = 3
_FULL_COVERAGE_RATIO: Final = 0.7
_PARTIAL_COVERAGE_RATIO: Final = 0.3

Coverage = Literal["full", "partial", "none"]


def _content_words(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in text.split() if len(word) >= _MIN_WORD_LEN)


class MemoryGuardrail(CustomGuardrail):
    """Injects learned context into requests and extracts learnings from strong-tier turns."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store_dir: Final[Path] = Path(os.environ.get("MEMORY_STORE_DIR", "/tmp/memory"))
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

            candidates = self._retrieve_learnings(current_ask)
            if not candidates:
                return data

            coverage = self._estimate_coverage(current_ask, candidates)
            injected = self._inject_into_turn(messages, candidates)
            if injected is not None:
                data["messages"] = injected

            metadata_key = "litellm_metadata" if "litellm_metadata" in data else "metadata"
            metadata = data.setdefault(metadata_key, {})
            if isinstance(metadata, dict):
                metadata["memory_coverage"] = coverage
                metadata["memory_injected"] = injected is not None

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

            self._store_learning(current_ask, response_text, routing_decision["tier"])
        except Exception as e:
            verbose_proxy_logger.error("MemoryGuardrail write path error: %s", e)

    def _retrieve_learnings(self, ask: str) -> tuple[str, ...]:
        """Keyword-overlap retrieval. Dumb on purpose: the classifier reading the
        augmented ask is the real filter, so first-stage retrieval can be sloppy."""
        ask_words = _content_words(ask)
        if not ask_words or not self.store_dir.exists():
            return ()

        scored: list[tuple[int, str]] = []
        for learning_file in self.store_dir.glob("*.md"):
            try:
                content = learning_file.read_text(encoding="utf-8")
            except OSError:
                continue
            overlap = len(ask_words & _content_words(content))
            if overlap > 0:
                scored.append((overlap, content))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return tuple(content for _score, content in scored[:_MAX_CANDIDATES])

    def _estimate_coverage(self, ask: str, learnings: tuple[str, ...]) -> Coverage:
        ask_words = _content_words(ask)
        if not learnings or not ask_words:
            return "none"

        covered = frozenset().union(*(_content_words(learning) for learning in learnings)) & ask_words
        ratio = len(covered) / len(ask_words)
        if ratio >= _FULL_COVERAGE_RATIO:
            return "full"
        if ratio >= _PARTIAL_COVERAGE_RATIO:
            return "partial"
        return "none"

    def _inject_into_turn(
        self, messages: list[dict[str, Any]], learnings: tuple[str, ...]
    ) -> Optional[list[dict[str, Any]]]:
        """Prepend learnings to the last user turn, not the system prompt: the system
        prompt is Claude Code's most-cached prefix, so injecting there would invalidate
        the cache on every turn. The current turn sits after every cache breakpoint."""
        injection = "\n\n".join(learnings)
        messages_copy = [dict(msg) for msg in messages]

        for msg in reversed(messages_copy):
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"[Learned context]\n{injection}\n\n[Current ask]\n{content}"
                return messages_copy

            if isinstance(content, list):
                new_content = [dict(block) if isinstance(block, dict) else block for block in content]
                for block in new_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = f"[Learned context]\n{injection}\n\n[Current ask]\n{block.get('text', '')}"
                        msg["content"] = new_content
                        return messages_copy

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

    def _store_learning(self, ask: str, response: str, tier: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        filename = self.store_dir / f"{timestamp}.md"
        filename.write_text(
            f"# Learning from {tier} tier\n\n**Ask:**\n{ask}\n\n**Response:**\n{response}\n",
            encoding="utf-8",
        )
