# Experiments 3 and 4: routing on the subtask, not the ask

Status: design verified against the code, models not yet picked, nothing built

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

Which non-Anthropic models the gateway actually serves. `model_prices_and_context_window.json`
knows plenty of cheap candidates including a coding-specialised `kimi-k2.7-code`, but the price
map is not the gateway's model list. Could not enumerate it: `/v1/models` on the sandbox returns
401 and there is no `.env` in this checkout, only `.env.example`. Experiment 3 is pointless if
every tier has to be an Anthropic model, since phase-specialisation is the entire premise

## Phase mix, measured

Over the 93-call session trace: explore 41%, implement 29%, other 20%, verify 10%. About 70% of
tool calls are not implement work, and today every one of them is priced at whatever tier the
human ask classified to
