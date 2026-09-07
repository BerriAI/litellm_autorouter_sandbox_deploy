# Experiment 1: learned-coverage routing

Status: spec, not built

## Hypothesis

The router learns how tasks get done, and at routing time checks whether what it already
knows covers the task in front of it. Where coverage is high, it drops tiers and injects the
relevant learnings. Where coverage is low, it routes up, and that expensive run becomes the
material for a new learning

Why this experiment and not one of the others: every other routing idea on the list is
allocative. It moves traffic between fixed models and its ceiling is "always use the strongest
model," which is free here because the company funds it. This one is additive. It changes what
the cheap model is capable of, so the routing question stops being "guess whether this is hard"
and becomes "we know how to do this, so it is now easy." That is a claim the router can be
right about, and it is the only version of this that can beat always-Fable rather than tie it

## Source

Based on Agno's learning machines article (https://www.agno.com/articles/learning-machines),
generalized. Their result that matters here is transfer, not compounding: Gemini Flash goes
37.33 to 96.42 and GLM-5.2 goes 7.92 to 83.46 when seeded with a frontier model's notes. Their
stated motivation is "from model-routing to agent-routing"

Generalized means no per-domain schema and no hand-written store like their 137-line ARC
`GameLearningStore`. Free-form learnings over ordinary coding work in whatever repo

What the article does not answer, and what this experiment is actually about: it seeds notes
and then runs the cheap model, but never says how you decide the notes are sufficient. Here
that decision is the routing decision

## Mechanism

Verified against the code: neither of v2's plugin seams (`ClassifierPlugin`, `RoutingPlugin`)
can mutate the outgoing message. Both only ever narrow `candidate_models` or set `signals`
(`complexity_router.py:2119`, `router.py:12804`); `raw_messages`/`structured_messages` go in
for the plugin to read, but nothing downstream writes them back into the request. So injection
cannot be a routing or classifier plugin

`CustomGuardrail.apply_guardrail` does rewrite request text, that is its whole job (PII
masking works this way), and `RoutingContext`'s own docstring says it mirrors that pattern.
Guardrails run in `pre_call_hook` (`common_request_processing.py:2012`), well before
`route_request` reaches the router's `async_pre_routing_hook`. So this experiment is a
guardrail, not a classifier plugin

### Read path, on the hot path, as a guardrail

Extract the current ask, reusing `_extract_current_ask_and_system_prompt`
(`complexity_router.py:536`), which already strips Claude Code's reminder blocks

Retrieve candidate learnings cheaply, by keyword or embedding

Inject the candidates into the current user turn before the request reaches the router. This
means the classifier v2 already runs on every uncached turn sees the augmented ask directly,
with no second mechanism needed to hand it a coverage judgment: an ask that already states "here
is how this was done before" reads as easier to whatever classifier scores it next. No custom
classifier is required for the first version; v2's existing `classifier_type: llm` is the
consumer

Coverage as a tier shift still needs a decision made somewhere. Simplest version: the same
guardrail computes a coverage estimate from retrieval quality (how much of the retrieved
learning matches the ask) and writes it to request metadata as a signal, for later use once we
want the shift to be more deliberate than "the classifier just reads better." Do not build a
second LLM call for this yet; only add one if the classifier reading the augmented prompt
turns out to under- or over-react

### Write path, off the hot path

Also a guardrail method: `async_post_call_success_hook` (see
`litellm/proxy/example_config_yaml/custom_guardrail.py` for the shape), which runs after the
response is already on its way back, so extraction never touches user latency. This is the
article's `process()` step

Only extract from turns that routed to a strong tier. Those are the novel ones, and they are
where learnings come from; a cheap turn cashing in an existing learning has nothing new to
teach. This filter is also what keeps extraction from being noisy and expensive, which is the
failure the article admits to when it says auto-extraction is "still a bit meh" because agents
"tend to find something, always"

### Wiring

One class, `MemoryGuardrail(CustomGuardrail)`, with `apply_guardrail` (or
`async_pre_call_hook`, same shape as the example file) for the read/inject path and
`async_post_call_success_hook` for the write path. Registered like any custom guardrail, a
`guardrails:` block with `guardrail_name` and `litellm_params.guardrail: <module>.MemoryGuardrail`

Scoped to only `moe-router`, not global: attach it via `litellm_params.guardrails` on that one
`model_list` entry in `proxy_config.yaml` (the model-level guardrail mechanism
`_check_and_merge_model_level_guardrails` merges in, `common_request_processing.py:2000`), so
plain model names and the untouched `moe-router` behavior (once this is toggled off) are
unaffected and remain the A/B control

### Coverage maps to a tier shift, not a binary

Four tiers already exist, so this is a gradient rather than cheap-versus-strong

| Coverage | Meaning | Shift |
|---|---|---|
| Full | procedure specified, the work is execution | down 2 |
| Partial | approach known, details novel | down 1 |
| None | novel work | no change, or up |

Start conservative. The asymmetry is savage: routing down wrongly costs trust and can cause a
churn event, routing down rightly saves a few seconds of latency and money nobody is paying
attention to

### Injection point

Prepended to the current user turn, not the system prompt

The article puts learnings in the system prompt via `build_context`. For Claude Code that is
the worst available position: the system prompt is the most stable and most-cached prefix in
the session, so injecting there invalidates the cache on every turn and the added cost can
swamp the routing win. The current user turn sits after every cache breakpoint, so prepending
there is cache-safe

Two consequences to hold onto. Learnings should be stable within a session rather than
recomputed per turn, or the cache never warms. And a model switch is itself a cache miss,
since caches are per-model, so cost accounting has to be per session, not per turn, or the
results will read as a loss even when the decisions were right

### Storage

Free-form markdown per learning, which is the article's format

Resolved: a Render persistent disk, added in `render.yaml` (1GB, mounted at `/var/data`,
`MEMORY_STORE_DIR=/var/data/memory`), so learnings survive every redeploy during the
experiment day instead of resetting each push. `standard` plan supports disks. Attaching a
disk to an already-running service is a one-time change Render may want confirmed in the
dashboard on the next deploy; if the deploy does not pick it up automatically, check
Render > litellm-autorouter-sandbox > Disks after pushing this

## Risks

A wrong learning plus a cheap model is worse than no learning at all. A strong model will
notice advice that does not fit the situation; a cheap one is more likely to follow it
confidently off a cliff. That is precisely the trust-killer that causes churn. Mitigation for
now is a conservative coverage bar, not machinery

Retrieval precision, because similarity is not coverage. "Add a provider" and "add a guardrail"
are semantically close and share no procedure, while "add the Anthropic provider" and "add the
Cohere provider" are close and transfer completely. This is why retrieval is two-stage and the
classifier is the real filter; the first-stage retrieval can be dumb without hurting much

Delayed payoff. There is nothing to cash in until learnings accumulate, and the article's whole
thesis is about repeat attempts. A day of real coding may not repeat a task, so the earliest
honest read on cross-session value is tomorrow

## How we will know it worked

The primary metric needs no infrastructure: how many times did you reach for the model
switcher today. You already hand-tune and resent it, so if that number goes to zero the
experiment won

Secondary signal, once there is traffic: the escalation rate on turns that routed down with
injected learnings. An escalation there is direct negative evidence about that specific
learning, and it is the seed of outcome tracking if this is worth continuing

## Deliberately not doing yet

Subtask-aware retrieval and injection. Subtask boundaries are better retrieval keys than whole
asks, but that is the parked experiment, and this one should not depend on it

Outcome tracking and learning validation, beyond noticing escalations

Agentic write-back. The article prefers it, but the router is an HTTP proxy, so exposing write
tools means injecting tool definitions into the payload and intercepting the calls, which
would fight Claude Code's own tool UI

Cross-user pooling. The router does see the whole team's traffic, which makes shared learnings
possible and is a genuinely good adoption story, but it is not needed to test the idea
