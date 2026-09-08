"""
Test-only guardrail that prefixes every assistant reply with `[MODEL_NAME] `, so a router's
model choice is visible in a normal Claude Code session without opening the proxy logs.

Post-call only: never touches the request, never affects inference, tool calls, or routing.
Purely cosmetic on the response text. Attach to a router's litellm_params.guardrails only when
actively testing; strip it before treating a router as anything but an experiment, since it
puts a literal string into every reply a real client would otherwise see clean.

Covers both response shapes a client can get back:
- non-streaming: async_post_call_success_hook, prefixes response.choices[0].message.content
- streaming (what Claude Code actually uses): async_post_call_streaming_iterator_hook,
  prefixes the first chunk that carries non-empty delta.content

Model name is read from response.model, which is the model that actually served the request
(e.g. "anthropic/claude-opus-5"), not the router name the client requested.
"""

from __future__ import annotations

from typing import Any, Final, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse


def _display_model_name(model: str | None) -> str:
    """Strips the `litellm_proxy/` prefix the "*" wildcard route adds: that is this proxy's
    own forwarding plumbing, not part of the upstream model's identity."""
    if model is None:
        return "unknown"
    return model.removeprefix("litellm_proxy/")


class ModelTagGuardrail(CustomGuardrail):
    """Prefixes assistant text with `[MODEL_NAME]` for visibility during manual testing."""

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Optional[Any]:
        try:
            if not isinstance(response, ModelResponse):
                return None
            choices = response.choices
            if not choices:
                return None
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content:
                return None

            tag: Final = f"[{_display_model_name(response.model)}] "
            if content.startswith(tag):
                # Same response object reaching this hook a second time (e.g. once via a
                # model-scoped guardrail merge, once via the general callback loop): tag
                # once, not twice.
                return None

            message.content = f"{tag}{content}"
            return response
        except Exception as e:
            verbose_proxy_logger.error("ModelTagGuardrail: post_call_success error: %s", e)
            return None

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ):
        tagged = False
        async for chunk in response:
            try:
                if not tagged:
                    delta = getattr(chunk.choices[0], "delta", None) if chunk.choices else None
                    content = getattr(delta, "content", None) if delta is not None else None
                    if isinstance(content, str) and content:
                        tag: Final = f"[{_display_model_name(getattr(chunk, 'model', None))}] "
                        if not content.startswith(tag):
                            delta.content = f"{tag}{content}"
                        tagged = True
            except Exception as e:
                verbose_proxy_logger.error("ModelTagGuardrail: streaming error: %s", e)
            yield chunk
