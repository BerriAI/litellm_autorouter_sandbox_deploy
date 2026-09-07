"""
Subtask boundary detection from tool-call composition (experiment 2, see
deploy/autorouter-sandbox/experiments/subtask_signal.md).

Classifies each tool call into a phase by name (Read/Grep/Glob -> explore, Write/Edit ->
implement) or, for Bash, by keywords in the command. A boundary is confirmed only once the
new phase holds for `debounce` consecutive calls, so a single stray Read mid-edit doesn't
register as a subtask switch.

This module only detects boundaries. It says nothing about whether a boundary is a good
place to change models, or whether the work on either side of it was done correctly -- that
is a separate, later question.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final


class Phase(str, Enum):
    EXPLORE = "explore"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ClassifiedCall:
    call: ToolCall
    phase: Phase


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    index: int
    from_phase: Phase | None
    to_phase: Phase


_EXPLORE_TOOLS: Final = frozenset(
    {
        "Read",
        "Grep",
        "Glob",
        "WebSearch",
        "WebFetch",
        "TaskList",
        "TaskGet",
        "ListAgents",
        "ReadMcpResourceTool",
        "ReadMcpResourceDirTool",
        "ListMcpResourcesTool",
    }
)
_IMPLEMENT_TOOLS: Final = frozenset({"Write", "Edit", "NotebookEdit"})

_SHIP_KEYWORDS: Final = ("git commit", "git push", "git add")
_VERIFY_KEYWORDS: Final = (
    "curl ",
    "pytest",
    "npm test",
    "npm run build",
    "vitest",
    "uvicorn",
    "health/liveliness",
    "npm ci",
)
_EXPLORE_BASH_KEYWORDS: Final = ("grep ", "find ", " ls ", "cat ", "sed -n", "python3 -c")


def classify_tool_call(call: ToolCall) -> Phase:
    if call.name in _EXPLORE_TOOLS:
        return Phase.EXPLORE
    if call.name in _IMPLEMENT_TOOLS:
        return Phase.IMPLEMENT
    if call.name != "Bash":
        return Phase.OTHER

    detail: Final = call.detail.lower()
    if any(kw in detail for kw in _SHIP_KEYWORDS):
        return Phase.OTHER
    if any(kw in detail for kw in _VERIFY_KEYWORDS):
        return Phase.VERIFY
    if any(kw in detail for kw in _EXPLORE_BASH_KEYWORDS):
        return Phase.EXPLORE
    return Phase.OTHER


def classify_trace(trace: Sequence[ToolCall]) -> tuple[ClassifiedCall, ...]:
    return tuple(ClassifiedCall(call=call, phase=classify_tool_call(call)) for call in trace)


def _phase_runs(classified: Sequence[ClassifiedCall]) -> tuple[tuple[Phase, int, int], ...]:
    """(phase, start_index, length) for each maximal run of consecutive same-phase calls."""
    index = 0
    runs: list[tuple[Phase, int, int]] = []  # mutable-ok: built once by a single forward scan
    for phase, group in itertools.groupby(classified, key=lambda c: c.phase):
        length = sum(1 for _ in group)
        runs.append((phase, index, length))
        index += length
    return tuple(runs)


def _confirm_boundaries(
    runs: tuple[tuple[Phase, int, int], ...],
    debounce: int,
    confirmed_phase: Phase | None = None,
) -> tuple[BoundaryEvent, ...]:
    """A run only flips the confirmed phase once it is at least `debounce` calls long, or it
    is the very first run. Shorter runs are noise: absorbed without a boundary."""
    if not runs:
        return ()

    phase, start, length = runs[0]
    rest: Final = runs[1:]

    if confirmed_phase is None:
        return (BoundaryEvent(index=start, from_phase=None, to_phase=phase),) + _confirm_boundaries(
            rest, debounce, phase
        )
    if phase != confirmed_phase and length >= debounce:
        return (BoundaryEvent(index=start, from_phase=confirmed_phase, to_phase=phase),) + _confirm_boundaries(
            rest, debounce, phase
        )
    return _confirm_boundaries(rest, debounce, confirmed_phase)


def detect_boundaries(classified: Sequence[ClassifiedCall], debounce: int = 2) -> tuple[BoundaryEvent, ...]:
    return _confirm_boundaries(_phase_runs(classified), debounce)


def render_timeline(classified: Sequence[ClassifiedCall], boundaries: Sequence[BoundaryEvent]) -> str:
    boundary_at: Final = {b.index: b for b in boundaries}
    lines: list[str] = []  # mutable-ok: built once by a single forward scan
    for i, c in enumerate(classified):
        if i in boundary_at:
            b = boundary_at[i]
            arrow = f"{b.from_phase.value if b.from_phase else 'start'} -> {b.to_phase.value}"
            lines.append(f"\n--- subtask boundary ({arrow}) ---")
        detail = f" {c.call.detail}" if c.call.detail else ""
        lines.append(f"[{i:>3}] {c.phase.value:<9} {c.call.name}{detail}")
    return "\n".join(lines)


# This session's own tool calls, condensed to name + a short detail, in order. Real, not
# synthetic: it is what actually happened while building the memory experiment above.
_THIS_SESSION_TRACE: Final = (
    ToolCall("Bash", "grep app.include_router customer_router proxy_server.py"),
    ToolCall("Write", "litellm/proxy/autorouter_memory_endpoints.py"),
    ToolCall("Bash", "grep import style routers proxy_server.py"),
    ToolCall("Read", "proxy_server.py:490"),
    ToolCall("Edit", "proxy_server.py add import"),
    ToolCall("Edit", "proxy_server.py add include_router"),
    ToolCall("Bash", "uvicorn boot, grep Traceback"),
    ToolCall("Bash", "grep traceback details"),
    ToolCall("Bash", "grep routing.py:1450"),
    ToolCall("Edit", "autorouter_memory_endpoints.py fix tags/dependencies tuple->list"),
    ToolCall("Bash", "uvicorn reboot"),
    ToolCall("Bash", "curl /autorouter/memory, write learning, curl again, curl no-auth, curl ui"),
    ToolCall("Bash", "kill proxy, rm -rf /tmp"),
    ToolCall("Bash", "git add commit push"),
    ToolCall("Bash", "python3 -c check model_prices for fable/haiku"),
    ToolCall("Read", "proxy_config.yaml"),
    ToolCall("Edit", "proxy_config.yaml classifier+fallback+reasoning tier"),
    ToolCall("Bash", "git add commit push"),
    ToolCall("Read", "render.yaml"),
    ToolCall("Bash", "grep LITELLM_LICENSE proxy_server.py"),
    ToolCall("Bash", "grep LITELLM_SALT_KEY encrypt_decrypt_utils.py"),
    ToolCall("WebFetch", "render.com blueprint spec"),
    ToolCall("Edit", "render.yaml add database + DATABASE_URL + SALT_KEY"),
    ToolCall("Bash", "docker run postgres (failed, docker not installed)"),
    ToolCall("Bash", "find prisma migrations"),
    ToolCall("Bash", "git add commit push"),
    ToolCall("Bash", "grep moe-router config"),
    ToolCall("Edit", "proxy_config.yaml remove guardrails from moe-router"),
    ToolCall("Read", "proxy_config.yaml"),
    ToolCall("Edit", "proxy_config.yaml add moe-learning-router"),
    ToolCall("Bash", "node --version, npm --version, ls ui checks"),
    ToolCall("Bash", "npm run build ui (background)"),
    ToolCall("Bash", "git add commit push router split"),
    ToolCall("Bash", "ls out/autorouter-learnings check build output"),
    ToolCall("Bash", "grep leftnav memory page discovery"),
    ToolCall("Read", "MemoryView.tsx"),
    ToolCall("Edit", "networking.tsx add fetch/clear functions"),
    ToolCall("Write", "AutorouterLearningsView.tsx"),
    ToolCall("Bash", "grep DeleteResourceModal props"),
    ToolCall("Edit", "AutorouterLearningsView.tsx fix modal props"),
    ToolCall("Write", "page.tsx"),
    ToolCall("Edit", "leftnav.tsx add nav item"),
    ToolCall("Edit", "page_metadata.ts"),
    ToolCall("Bash", "grep uiHref routeSegmentForPathname"),
    ToolCall("Bash", "npm run build ui (background)"),
    ToolCall("Bash", "npx vitest run leftnav.test.tsx"),
    ToolCall("Bash", "cp -r out/. to served _experimental/out"),
    ToolCall("Bash", "uvicorn boot, curl /v1/models, curl ui page, grep bundle for strings"),
    ToolCall("Bash", "python yaml check guardrail scoping, curl endpoint"),
    ToolCall("Bash", "write learning via guardrail script, curl viewer"),
    ToolCall("Bash", "kill, rm -rf tmp, git add commit push"),
    ToolCall("Bash", "grep autorouter_savings call sites"),
    ToolCall("Read", "litellm_logging.py"),
    ToolCall("Read", "savings.py compute_autorouter_savings"),
    ToolCall("Read", "savings.py autorouter_savings_for_request"),
    ToolCall("Bash", "docker attempt (failed), find migrations (not found)"),
    ToolCall("Edit", "memory_guardrail.py record injected tokens"),
    ToolCall("Edit", "memory_guardrail.py _inject_into_turn return text"),
    ToolCall("Bash", "grep other references to _inject_into_turn"),
    ToolCall("Bash", "python verify guardrail records token count"),
    ToolCall("Edit", "experiments/memory.md add caveats section"),
    ToolCall("Bash", "git add commit push"),
    ToolCall("Bash", "python3 -c check model_prices opus-5 reasoning support"),
    ToolCall("Bash", "python3 -c check opus-5-high model id exists"),
    ToolCall("Read", "proxy_config.yaml"),
    ToolCall("Edit", "proxy_config.yaml COMPLEX to opus-5 high"),
    ToolCall("Bash", "git add commit push"),
)


def main() -> None:
    classified = classify_trace(_THIS_SESSION_TRACE)
    boundaries = detect_boundaries(classified, debounce=2)
    print(render_timeline(classified, boundaries))
    print(f"\n{len(classified)} tool calls, {len(boundaries)} confirmed subtask boundaries")


if __name__ == "__main__":
    main()
