# Experiment 2 (signal only): subtask boundary detection from tool-call composition

Status: signal built and demoed online (`subtask_signal.py`), not wired into any router

## Online, not post hoc

The first version of this only worked in retrospect: it took a complete trace and ran
`itertools.groupby` over it, which needs to see a run end before it can report the run. That is
useless for routing, which has to decide during the request

The live path turned out not to need session persistence at all. Claude Code resends the full
message history on every request, and at turn N that history contains only turns 1..N, so
reading it is causal by construction; there is no future in the payload to accidentally peek at.
`extract_tool_calls` reads the wire format (`assistant` turns carrying a `tool_calls` array)
directly, and `current_phase(messages)` recomputes the phase from scratch per request. Every
request is a cold start that happens to already contain its own history. `advance()` is the
actual online step: it confirms a run the moment it reaches `debounce` length, never waiting to
see what follows

Demoed both ways in `main()`: a message-by-message live path over wire-shaped messages, and a
full replay of the session trace comparing online against post hoc

## Online and post hoc agree exactly

13 boundaries each, identical positions, mean detection lag 1.00 calls, zero disagreement

This is forced by the algorithm, not luck. `groupby` looked like it needed the future because it
waits for a run to finish before reporting its length, but "has this run reached 2 calls yet" is
knowable the instant the second call arrives, and nothing later in the trace changes that fact.
A run-length debounce was always computable online; the batch version was a lazier way to compute
the same answer, not a better-informed one

So real time costs nothing in accuracy against this baseline. What it costs is different: the
answer has to be produced inside one request's latency budget, recomputed from scratch each time,
with no ability to revise a call already routed on

## Scope

This is not routing yet. The question on the table is narrower: given a trace of tool calls,
can something cheap and local say where one subtask ends and the next begins. Whether a
boundary is a good place to switch models, and whether the work on either side was done
correctly, are separate questions for later

## Mechanism

Each tool call gets a phase by name: Read/Grep/Glob/WebSearch/etc. -> explore, Write/Edit ->
implement. Bash is classified by keyword in the command (test/curl/build -> verify,
grep/find/cat -> explore, git add/commit/push -> other). No LLM call, no latency, reads only
what the harness already sends

A boundary is confirmed only once the new phase holds for `debounce` (default 2) consecutive
calls. A single stray Read in the middle of an edit run does not flip the confirmed phase.
This is the whole mechanism: classify, run-length encode, debounce

## Example run

`subtask_signal.py` includes this actual session's tool-call history (this session, condensed
to name + short detail, real not synthetic) as `_THIS_SESSION_TRACE`. Running it:

```
python3 deploy/autorouter-sandbox/subtask_signal.py
```

produced 67 tool calls, 13 confirmed boundaries. Spot-checked against what actually happened:
single-call blips (a lone Write right after a grep, a lone Edit inside an explore run) were
correctly absorbed rather than flagged as switches; real phase changes (explore into implement,
implement into verify, verify into explore) were caught

## What this run exposed

A stretch of the session (roughly calls 14-30) was a tight loop of read-one-file,
edit-one-line, commit, read-the-next-file. None of those individual flips lasted two calls in a
row, so the whole stretch rendered as one undifferentiated `explore` run with no internal
boundaries. That is a direct cost of debouncing against noise: it trades away fine-grained
boundaries during rapid alternation to avoid flagging every single-line edit as a new subtask.
Whether that trade is right is a judgment call, not something this run settles

## Known limits, not yet addressed

Detection is reactive by construction: a phase cannot be classified before at least one call
in it has happened, so every boundary is confirmed one call after it started. No debounce
value removes this; a lower debounce only reduces how many calls of lag beyond that first one

The demo trace's tool-call details are paraphrased by hand, not the literal arguments a real
harness would send. This proves the algorithm's shape against a plausible trace, not its
accuracy on raw production payloads

No ground truth. There is no labeled "correct" segmentation to score against, so calling a
boundary right or wrong is a subjective read of the rendered timeline, not a metric

## Deliberately not doing yet

Wiring this into a `RoutingPlugin` or any router. This file is the detector in isolation

Weighting boundaries by routing consequence (a missed boundary between two same-difficulty
subtasks costs nothing; a missed boundary at a difficulty transition costs a lot). Worth
doing once this feeds an actual routing decision, not before

Response-based or LLM-based segmentation, as alternatives to tool-call composition. Not ruled
out, just not the cheapest thing to try first

Predictive segmentation (decompose the ask into an expected subtask plan up front, track
position in it) as a way past the one-call-of-lag floor. A real fourth option, trades reactive
accuracy for upfront planning risk, not attempted here
