# Experiments 3 and 4: routing on the subtask, not the ask

Status: both built and verified against a live proxy

## The two are genuinely different, and an earlier version of this was not

The first build of experiment 4 was a static `phase -> fixed tier` lookup with no difficulty
assessment anywhere in it. That is experiment 3 wearing the built-in tier names as a costume,
and the router name claiming "difficulty" was doing no work. Caught in review, rebuilt

Verified live, same four fixtures through both routers:

| fixture | by TYPE | by DIFFICULTY |
|---|---|---|
| tiny edit (1 file, 2 calls) | COMPLEX (opus-5) | MEDIUM (sonnet-5) |
| big refactor (4 files, 8 calls) | COMPLEX (opus-5) | REASONING (fable-5-1) |
| wide search (6 targets) | SIMPLE (deepseek-v4-flash) | MEDIUM |
| verify passing | MEDIUM (haiku-4-5) | SIMPLE |

The two implement rows are the point: identical phase, and the type router necessarily sends
both to the same model while the difficulty router splits them two tiers apart

## How the difficulty scoring works

Phase sets a band, not an answer. Signals counted over the current subtask only (calls since
the last confirmed boundary) move the tier inside that band, then it is clamped:

    explore    SIMPLE..MEDIUM      start SIMPLE
    implement  MEDIUM..REASONING   start COMPLEX
    verify     SIMPLE..COMPLEX     start SIMPLE

Signals: `long-subtask` (>=6 calls), `multi-file-edit` (>=2 files, implement only),
`wide-search` (>=4 targets, explore only), `single-small-edit` (<=1 file and <=2 calls, moves
down), `failure-in-output` (traceback/FAILED in recent tool results, moves up)

No LLM call. Every signal is read from the payload the request already carries, so the
classifier costs a dict lookup against the ~3s the LLM classifier spends on `moe-router`

## Loader gotcha, cost an outage if it had shipped

`classifier_plugin` resolves through `get_instance_fn`, which is a plain `getattr` on the
module (`litellm/proxy/types_utils/utils.py:55`). It never instantiates, unlike the guardrail
loader. A bare class passes `resolve_classifier_plugin`'s `isinstance(resolved, ClassifierPlugin)`
check because a class satisfies a runtime_checkable Protocol structurally, and
`inspect.iscoroutinefunction` also passes on the unbound function. So `module.ClassName` boots
clean and then fails on the first classified request, in production, mid-session

Both modules therefore export a module-level instance and the config points at that:
`subtask_type_classifier.subtask_type_classifier`, not `...SubtaskTypeClassifier`

Both build on the online phase detector (`subtask_signal.py`, see `subtask_signal.md`). They are
different experiments and should be separate model names so they stay a clean A/B against
`moe-router`:

- `moe-subtask-router` (experiment 3): pick a model per phase TYPE
- `moe-phase-difficulty-router` (experiment 4): pick a tier per phase by DIFFICULTY

## Why both are ClassifierPlugins, not RoutingPlugins

Verified in `complexity_router.py`, and this rules out the obvious first guess:

`RoutingPlugin` cannot express either experiment. `_pick_model_for_tier` (:2104) builds the
plugin's `candidate_models` from `self._tier_pools().get(tier_key)`, i.e. the pool of the tier the
classifier ALREADY chose. A routing plugin only narrows within that tier; it cannot move the
request to a different tier. Both experiments are about choosing the tier, so both are classifiers

`ClassifierPlugin.classify(context) -> str | None` (:1780-1827) returns a tier name, resolved via
`resolve_classified_tier`. It receives `structured_messages` (the full history, which is exactly
what `current_phase()` reads), runs under `classifier_plugin_timeout_ms` (default 3000), and any
failure or a `None` return falls back safely rather than failing the request

Config: `classifier_type: custom` plus `classifier_plugin: <dotted.path.to.instance>`, resolved at
startup like guardrails (:765-772). `classifier_plugin` set without `classifier_type: custom` is a
hard config error (:1241), so this cannot be half-wired by accident

## Experiment 3: model per phase type

Phases are not difficulty levels, so they should not reuse SIMPLE/MEDIUM/COMPLEX/REASONING.
`tier_definitions` (config.py:587) replaces the built-in ladder with arbitrary named tiers, which
is the honest way to express this: the tier names become `explore` / `implement` / `verify`

What `tier_definitions` costs, verified in `_tier_definition_conflicts` (config.py:1389-1401):
`adaptive`, `session_affinity`, `escalation_keywords`, `stall_escalation_enabled` and `plugins`
are all rejected outright, because they are built on the built-in severity order that a custom
tier set has no equivalent for. `tier_labels` and the rubric presets are out too. `fallback_tier`
becomes required. Notably `classifier_plugin` is NOT in the banned list, and `tier_definitions`
explicitly requires `classifier_type` of `llm` or `custom`, so custom-classifier plus custom-tiers
is the supported combination

Losing `escalation_keywords` is the real cost to weigh: that is the "LITELLM ESCALATE" manual
override, gone on this router

The classifier is then trivial and needs no LLM call: read `current_phase(structured_messages)`,
return the phase name, return `None` before the first tool call so `fallback_tier` handles the
opening turn. Cost is zero and latency is a dict lookup, versus the ~3s Haiku classifier call the
other routers make

## Experiment 4: difficulty per subtask

Keeps the built-in tiers, so it keeps escalation and everything else. What changes is what gets
classified

Today `classification_mode: every_request` re-runs the classifier each turn, but
`_extract_current_ask_and_system_prompt` anchors on the last real HUMAN ask, and tool-result
turns never count as an ask (`_newest_turn_is_human_ask`, :579). So turn 15 classifies the same
text as turn 1. Difficulty tracks the ask for the whole session, and intra-session structure is
invisible by construction. That is not a bug, it is what makes it stable, but it means a session
that opens with a hard ask pays the hard tier for its trivial parts

This experiment classifies the CURRENT SUBTASK instead: at a phase boundary, score the difficulty
of the work happening now (the recent tool activity plus the ask as context) rather than the ask
alone. It is the version of `every_request` that actually varies per request

This is the one worth caring about. Experiment 3 is allocative and its ceiling is "always use the
strongest model", which costs nothing here. Experiment 4 can genuinely beat that ceiling, because
routing an explore stretch down is close to free in quality while a difficulty-blind router pays
top tier for it

## Open question, not yet decided

Whether experiment 4's per-subtask classifier needs an LLM call at all. The phase itself is a
strong difficulty prior (explore is usually easy, implement usually hard) and is free. Starting
with a pure phase-to-tier mapping and only adding an LLM call if that proves too coarse would keep
the first version cheap, and would make experiment 4 a strict superset of experiment 3

## Blocked on

Gateway model availability. `/v1/models` endpoint confirmed it only serves `moe-router` and
`moe-learning-router`, forwarding actual model calls upstream to the sandbox. The gateway does
not enumerate frontier models, and experiment 3 needs upstream to add non-Anthropic models to
its config first. Not a router problem, just a prerequisite

## Model picks for experiment 4 (per-subtask difficulty)

Keeping built-in tiers, so these are all Anthropic. Data from Artificial Analysis Intelligence
Index v4.3 (fetched 2026-09-07) and Terminal-Bench 4.0 (updated 2026-09-03).

Phase-to-tier mapping is provisional; would revalidate against real traffic:

| Phase | Suggested tier | Rationale |
|---|---|---|
| **explore** | SIMPLE (Haiku) | File read/search is input-dominated, 41% of calls. Haiku is 20x cheaper on input ($1/$5 vs $10/$50 Fable). No benchmark isolates retrieval, but file-exploration work is high-volume, low-reasoning, where latency matters more than intelligence. AA's Intelligence Index suggests Sonnet handles retrieval fine (no explicit benchmark but appears in task mix). Use Haiku. |
| **implement** | COMPLEX (Opus 5) | Code editing is the risky phase. Terminal-Bench 4.0 shows Opus 5 at 51.8% vs GPT-6 Astra at ~58%, but Opus is proven on your codebase and the risk of a broken edit is higher than the latency savings. Fable 5.1 peaks at 57.9% on Terminal-Bench, so a future experiment could try it, but Opus is the safe starting point. |
| **verify** | MEDIUM (Sonnet 5) | Test output interpretation has no benchmark, but it is high-volume structured parsing with low reasoning load. Sonnet 5 ($2/$10) is 5x cheaper than Opus and should handle test output easily. If verify often triggers escalation (LITELLM ESCALATE), move it up. |

Cost calculation, phase mix (explore 41%, implement 29%, verify 10%, other 20%):
- Baseline (always Opus): 1,000 calls cost ~$50 (Opus $5/$25)
- Per-subtask: explore 410×Haiku + implement 290×Opus + verify 100×Sonnet ≈ $30, 40% savings
- Risk: if Sonnet fails on a verify phase and escalates to Opus, the savings shrink fast

Caveat: these are guesses based on general benchmark positioning, not empirical on your actual
traffic. The first version should log every tier decision and escalation, and validate that the
phase-to-tier mapping is actually correct before shipping

## Phase mix, measured

Over the 93-call session trace: explore 41%, implement 29%, other 20%, verify 10%. About 70% of
tool calls are not implement work, and today every one of them is priced at whatever tier the
human ask classified to
