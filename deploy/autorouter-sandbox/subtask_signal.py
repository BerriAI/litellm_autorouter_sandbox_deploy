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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final


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


@dataclass(frozen=True, slots=True)
class OnlineState:
    """Everything the online detector needs to carry between calls. Small and picklable on
    purpose: it has to survive between two HTTP requests."""

    confirmed: Phase | None = None
    pending: Phase | None = None
    pending_run: int = 0
    seen: int = 0


@dataclass(frozen=True, slots=True)
class OnlineBoundary:
    started_at: int
    detected_at: int
    from_phase: Phase | None
    to_phase: Phase

    @property
    def lag(self) -> int:
        return self.detected_at - self.started_at


def advance(state: OnlineState, phase: Phase, debounce: int = 2) -> tuple[OnlineState, OnlineBoundary | None]:
    """Feed one tool call. Returns the next state and a boundary if this call confirmed one.

    Unlike `detect_boundaries`, this never looks at a call it has not seen: a run is confirmed
    the moment it reaches `debounce` length, not once the following run reveals it ended. That
    is the whole difference between deciding live and explaining afterwards, and it is why the
    two disagree.
    """
    index: Final = state.seen
    if state.confirmed is None:
        return (
            OnlineState(confirmed=phase, seen=index + 1),
            OnlineBoundary(started_at=index, detected_at=index, from_phase=None, to_phase=phase),
        )

    if phase == state.confirmed:
        return OnlineState(confirmed=state.confirmed, seen=index + 1), None

    run: Final = state.pending_run + 1 if phase == state.pending else 1
    if run >= debounce:
        return (
            OnlineState(confirmed=phase, seen=index + 1),
            OnlineBoundary(
                started_at=index - run + 1,
                detected_at=index,
                from_phase=state.confirmed,
                to_phase=phase,
            ),
        )
    return OnlineState(confirmed=state.confirmed, pending=phase, pending_run=run, seen=index + 1), None


def replay_online(trace: Sequence[ToolCall], debounce: int = 2, skip_neutral: bool = True) -> tuple[OnlineBoundary, ...]:
    """Feed a trace through `advance` one call at a time, exactly as the live path would.

    `skip_neutral` drops OTHER calls before they reach the state machine. OTHER is the
    catch-all for "this call carries no phase signal" (a git commit, a cleanup), and a call
    with no signal should neither confirm a new phase nor break the current run. Letting it
    act as a phase was making commits fire two boundaries, one in and one back out, churning
    the tier for no routing benefit.

    Boundary indices are remapped back to positions in the original `trace`, so `lag` counts
    real elapsed calls including any neutral ones that were skipped.
    """
    state = OnlineState()  # rebind-ok: fold over the trace, each step depends on the previous
    events: list[OnlineBoundary] = []  # mutable-ok: append-only accumulator over one forward pass
    signal_positions: list[int] = []  # mutable-ok: same, maps filtered index -> original index
    for original_index, call in enumerate(trace):
        phase = classify_tool_call(call)
        if skip_neutral and phase is Phase.OTHER:
            continue
        signal_positions.append(original_index)
        state, boundary = advance(state, phase, debounce)
        if boundary is not None:
            events.append(
                OnlineBoundary(
                    started_at=signal_positions[boundary.started_at],
                    detected_at=signal_positions[boundary.detected_at],
                    from_phase=boundary.from_phase,
                    to_phase=boundary.to_phase,
                )
            )
    return tuple(events)


def extract_tool_calls(messages: Sequence[Mapping[str, Any]]) -> tuple[ToolCall, ...]:
    """Pull the tool-call trace out of an in-flight request's message history, oldest first.

    This is what makes the live path stateless: Claude Code resends the full history every
    turn, and at turn N that history contains only turns 1..N, so reading it is causal by
    construction. Nothing needs to be persisted between requests.

    Delegates to trajectory_signals.iter_tool_call_events_newest_first, which reads both wire
    shapes a request can carry: chat-completions `tool_calls` entries and Anthropic Messages
    `tool_use` content blocks. This module used to parse only the former, which silently
    produced zero tool calls -- and so no phase, ever -- against the Anthropic Messages shape
    Claude Code actually sends through this gateway. Caught by testing against the real shape
    rather than only the one this module happened to be written against.
    """
    from trajectory_signals import iter_tool_call_events_newest_first

    newest_first: Final = tuple(iter_tool_call_events_newest_first(messages))
    return tuple(ToolCall(name=event.signature[0], detail=event.signature[1]) for event in reversed(newest_first))


def current_phase(messages: Sequence[Mapping[str, Any]], debounce: int = 2) -> tuple[Phase | None, OnlineBoundary | None]:
    """The live entry point: current confirmed phase for an in-flight request, plus the most
    recent boundary if there is one. A router would key its tier decision off the phase."""
    boundaries: Final = replay_online(extract_tool_calls(messages), debounce)
    if not boundaries:
        return None, None
    return boundaries[-1].to_phase, boundaries[-1]


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
    # Terminal-Bench question, then the subtask-signal build itself.
    ToolCall("WebSearch", "terminal-bench 2.1 details"),
    ToolCall("WebSearch", "terminal-bench litellm proxy wiring"),
    ToolCall("Write", "subtask_signal.py post-hoc detector"),
    ToolCall("Bash", "python3 subtask_signal.py first run"),
    ToolCall("Write", "experiments/subtask_signal.md"),
    ToolCall("Bash", "git add commit push subtask signal"),
    ToolCall("Bash", "grep RoutingPlugin RoutingContext types/router.py"),
    ToolCall("Bash", "grep tool_calls complexity_router.py"),
    ToolCall("Read", "complexity_router.py:579 newest_turn_is_human_ask"),
    ToolCall("Bash", "grep plugins invocation sites complexity_router.py"),
    ToolCall("Edit", "subtask_signal.py add advance/OnlineState/extract_tool_calls"),
    ToolCall("Edit", "subtask_signal.py imports Mapping/Any"),
    ToolCall("Edit", "subtask_signal.py online vs posthoc comparison in main"),
    ToolCall("Bash", "python3 subtask_signal.py online comparison"),
    ToolCall("Edit", "subtask_signal.py add wire-format live demo"),
    ToolCall("Bash", "python3 subtask_signal.py live path"),
    ToolCall("Edit", "subtask_signal.py fix duplicate confirmed tag"),
    ToolCall("Bash", "python3 subtask_signal.py verify fix"),
    ToolCall("Bash", "python3 subtask_signal.py tail check"),
    ToolCall("Edit", "experiments/subtask_signal.md online section"),
    ToolCall("Bash", "grep stray CJK char in md"),
    ToolCall("Edit", "experiments/subtask_signal.md fix stray char"),
    ToolCall("Bash", "git add commit push online detector"),
    ToolCall("Bash", "python3 -c debounce sweep 1-4"),
    ToolCall("Bash", "python3 -c debounce transitions"),
    ToolCall("Bash", "python3 -c check trace length"),
)


def _wire_message(role: str, tool_name: str | None = None, arguments: str = "") -> dict[str, Any]:
    """One synthetic chat-completions message, shaped like what a real request carries: an
    assistant turn with a `tool_calls` array, or the `tool` role reply that follows it."""
    if tool_name is None:
        return {"role": role, "content": ""}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": tool_name, "arguments": arguments}}],
    }


_WIRE_TRACE: Final = (
    _wire_message("user"),
    _wire_message("assistant", "Read", '{"file_path": "foo.py"}'),
    _wire_message("tool"),
    _wire_message("assistant", "Grep", '{"pattern": "def foo"}'),
    _wire_message("tool"),
    _wire_message("assistant", "Edit", '{"file_path": "foo.py"}'),
    _wire_message("tool"),
    _wire_message("assistant", "Edit", '{"file_path": "foo.py"}'),
    _wire_message("tool"),
)


def _demo_live_path() -> None:
    """Simulates what the guardrail's pre_call hook actually sees: at request N, only turns
    1..N exist, because Claude Code resends the whole history every turn. No hand-labeled
    ToolCall list, no persisted state -- extract_tool_calls reads the wire format directly,
    and current_phase is recomputed fresh from whatever prefix of history the request carries."""
    print("=== LIVE PATH: recomputed fresh from the request's own message history ===\n")
    previous_calls = 0
    for n in range(1, len(_WIRE_TRACE) + 1):
        prefix = _WIRE_TRACE[:n]
        call_count = len(extract_tool_calls(prefix))
        phase, boundary = current_phase(prefix)
        tag = f"-> {phase.value}" if phase else "(no confirmed phase yet)"
        if boundary is not None and call_count > previous_calls and boundary.detected_at == call_count - 1:
            tag += "  <== just confirmed this call"
        print(f"request with {n:>2} messages so far: {tag}")
        previous_calls = call_count
    print()


def main() -> None:
    _demo_live_path()
    classified: Final = classify_trace(_THIS_SESSION_TRACE)
    online: Final = replay_online(_THIS_SESSION_TRACE, debounce=2)
    posthoc: Final = detect_boundaries(classified, debounce=2)

    print("=== ONLINE, OTHER phase-neutral (the shipping config) ===\n")
    detected_at: Final = {b.detected_at: b for b in online}
    for i, c in enumerate(classified):
        marker = ""
        if i in detected_at:
            b = detected_at[i]
            origin = b.from_phase.value if b.from_phase else "start"
            marker = f"   <== BOUNDARY {origin} -> {b.to_phase.value} (began at [{b.started_at}], lag {b.lag})"
        print(f"[{i:>3}] {c.phase.value:<9} {c.call.name}{marker}")

    noisy: Final = replay_online(_THIS_SESSION_TRACE, debounce=2, skip_neutral=False)
    print(f"\n{len(classified)} tool calls")
    print(f"OTHER neutral:    {len(online)} boundaries  (every one a real explore/implement/verify switch)")
    print(f"OTHER as a phase: {len(noisy)} boundaries  (adds git-commit churn: fires leaving work, fires again returning)")

    churn: Final = tuple(b for b in noisy if Phase.OTHER in (b.from_phase, b.to_phase))
    print(f"  of those, {len(churn)} are transitions into or out of OTHER, i.e. tier changes with no routing benefit")

    lags: Final = tuple(b.lag for b in online if b.from_phase is not None)
    if lags:
        print(f"\ndetection lag: min {min(lags)}, max {max(lags)}, mean {sum(lags) / len(lags):.2f} calls")
    print(f"posthoc (legacy, OTHER as a phase): {len(posthoc)} boundaries")


if __name__ == "__main__":
    main()
