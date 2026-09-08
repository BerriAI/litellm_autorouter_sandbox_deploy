"""
Read-only viewer for the learned-coverage routing experiment's learning store.

Experiment-only instrumentation for the autorouter sandbox fork, see
deploy/autorouter-sandbox/experiments/memory.md. Not for upstream: it reads the guardrail's
plain-file store directly rather than going through any storage abstraction, because the
experiment has no storage abstraction yet and should not grow one before it is worth keeping.
"""

import html
import json
import os
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter(tags=["autorouter memory"])

_MAX_LEARNINGS: Final = 200


def _store_dir() -> Path:
    return Path(os.environ.get("MEMORY_STORE_DIR", "/tmp/memory"))


def _read_learnings() -> tuple[tuple[str, str], ...]:
    """(name, text) newest first. Unreadable or malformed files are skipped rather than
    failing the whole listing, since one bad file should not hide the rest. The vector is
    left out: it is retrieval plumbing, not something worth reading."""
    store: Final = _store_dir()
    if not store.exists():
        return ()

    entries: list[tuple[str, str]] = []
    for path in sorted(store.glob("*.json"), reverse=True)[:_MAX_LEARNINGS]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            entries.append((path.name, record["text"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return tuple(entries)


@router.get("/autorouter/memory", dependencies=[Depends(user_api_key_auth)])
async def list_learnings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> dict[str, object]:
    """Every stored learning, newest first, as JSON."""
    learnings: Final = _read_learnings()
    return {
        "store_dir": str(_store_dir()),
        "count": len(learnings),
        "learnings": [{"name": name, "content": content} for name, content in learnings],
    }


@router.get("/autorouter/memory/ui", dependencies=[Depends(user_api_key_auth)])
async def learnings_ui(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> HTMLResponse:
    """The same listing as a plain page, for reading in a browser."""
    learnings: Final = _read_learnings()
    if learnings:
        body = "".join(
            f"<details open><summary>{html.escape(name)}</summary>"
            f"<pre>{html.escape(content)}</pre></details>"
            for name, content in learnings
        )
    else:
        body = f"<p>No learnings stored yet in <code>{html.escape(str(_store_dir()))}</code></p>"

    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><title>autorouter learnings</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:60rem}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:6px;overflow-x:auto}"
        "summary{cursor:pointer;font-weight:600;margin-top:1rem}</style>"
        f"<h1>Learnings ({len(learnings)})</h1>{body}"
    )


@router.delete("/autorouter/memory", dependencies=[Depends(user_api_key_auth)])
async def clear_learnings(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> dict[str, int]:
    """Wipe the store, so an experiment run can start from a known-empty state."""
    store: Final = _store_dir()
    if not store.exists():
        return {"deleted": 0}

    deleted = 0
    for path in store.glob("*.json"):
        try:
            path.unlink()
            deleted += 1
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"could not delete {path.name}: {e}"
            ) from e
    return {"deleted": deleted}
