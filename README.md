# Companion

A domain-agnostic agentic companion: persistent per-user memory,
Bayesian belief instead of a bare confidence float, an ask-vs-act
decision that improves online from real feedback, and a pluggable
reasoning backend — a dependency-free stub by default, Google's Agent
Development Kit (or raw Gemini) in production.

```
INPUT --> [ Internal State + External State ] --> BRAIN --> ACTION --> NEW STATE
                   ^                                                       |
                   +-------------------- feeds back in ---------------------+
```

## Quickstart

```bash
cd companion
python main.py --domain general        # zero dependencies -- runs on the built-in stub
python main.py --domain study --live    # real Gemini, via ADK if installed
```

`.env` (copy from `.env.example`) holds `GEMINI_API_KEY` and `GEMINI_MODEL`.
Profile/memory persists to `companion.db` next to `main.py` across runs;
pass `--fresh` for a throwaway in-memory session instead.

### Web app

A browser chat UI over the same `Companion` class the CLI drives — no
separate agent logic, `web/server.py` just calls the same `turn()` /
`give_feedback()` / `profile()` / `live_feed()` / `adaptation_metrics()`
methods `main.py` does, over HTTP.

```bash
pip install fastapi uvicorn   # or: pip install -r requirements.txt
python web/server.py
# open http://localhost:8000
```

The sidebar mirrors the CLI's `:profile` / `:feed` / `:metrics`
commands live as you chat, plus one-click accept/keep on any fact
staged as `pending_confirmation` by the Consolidator.

### Multimodal input (images)

Both interfaces can attach an image to a turn — the web app has a 📎
button next to the message box, the CLI has `:image <path>` (stages
it for your *next* message; bare `:image` clears it). On the stub
backend it's just acknowledged in the reply; with a real Gemini
backend (`--live` / a configured `GEMINI_API_KEY`) it's sent as an
actual multimodal `Part` alongside the text, so the model genuinely
sees it — verified against the real installed `google-genai` API
(`Part.from_bytes(data=..., mime_type=...)`, mixed into the same
`contents` list as the text). Nothing about the Clarifier, memory, or
belief math changed to support this — it's carried as one optional
field on `ExternalState`, threaded through `Guide`'s existing
`context` dict.

## File map

```
companion/
  state.py             ExternalState (Actor), InternalState (Observer), SemanticFact
  belief.py             BetaBelief + KL divergence -- confidence as evidence, not a magic float
  gate.py                 AdaptiveGate -- deviation-triggered "ask" trigger
  domain.py                DomainConfig -- points the machinery at any vertical
  privacy.py                 PII detection/redaction (the trust layer)
  metrics.py                  AdaptationMetrics -- the numbers that prove adaptation happened
  agent.py                     Companion -- the Brain, wires the full loop
  agents/
    clarifier.py                ask-vs-act: learned + entropy-scored + per-user adaptive gate
    retriever.py                  semantic + episodic + domain-knowledge retrieval
    guide.py                       produces the personalized response
    feedback.py                     explicit + implicit signal capture
  memory/
    store.py                        UnifiedMemoryStore -- the one shared, source-tagged database
    consolidator.py                  episodic -> semantic, Bayesian contradiction handling
  llm/
    base.py                           LLMBackend interface
    stub.py                            dependency-free default (what runs out of the box)
    adk_backend.py                      PREFERRED --live backend: routes through Google's
                                         Agent Development Kit (pip install google-adk)
    gemini.py                            fallback --live backend: hardened raw google-genai
                                          SDK call, used automatically if google-adk isn't installed
  domains/
    general.py                          the default: no fixed vertical
    study.py                             a second domain, its domain_knowledge partly sourced
                                          from a real ingested dataset (see below)
    ingest_whatsapp.py                    messy-data ingestion pipeline: parse -> chunk ->
                                           anonymize -> safety-filter -> domain_knowledge
    domain_knowledge_loader.py             loads a domain's ingested knowledge from JSON
  tests/
    test_companion.py                     the loop, memory, privacy, idempotency, metrics, learning
    test_belief.py                         Bayesian belief math + the adaptive gate
    test_ingest.py                          anonymization + safety-filter, on synthetic data
    test_llm_backends.py                     backend fail-fast paths (skipped if the optional
                                              google-adk/google-genai packages aren't installed)
  web/
    server.py                             FastAPI wrapper -- every endpoint calls straight into
                                           agent.py, no logic of its own (pip install fastapi uvicorn)
    static/index.html                      chat UI + a live profile/feed/metrics sidebar
```

## Why two Gemini backends

`llm/adk_backend.py` and `llm/gemini.py` both implement the same
`LLMBackend` interface and are interchangeable — `main.py --live` prefers
ADK and falls back to the raw SDK automatically (`--backend` overrides
this). The reason both exist: calling `google-genai` directly is a real
way to reach Gemini, but it's an SDK call, not a framework. Routing the
same generation through an actual `google.adk.Agent` + `Runner` is an
unambiguous use of Google's Agent Development Kit — useful when a
"Google Agent Framework" requirement needs to be clearly, not just
arguably, satisfied. Nothing upstream (memory, belief, ask-vs-act,
consolidation) needed to change either way; only `llm/` grew a second
file.

## Data ingestion (`domains/ingest_whatsapp.py`)

Turns a raw WhatsApp `.txt` export into the `study` domain's knowledge
base, end to end:

1. **Parse** export lines into timestamped, per-sender messages.
2. **Redact** PII (email/phone/card/SSN) via `privacy.py`.
3. **Chunk** by conversation gap (a real topic boundary, not an arbitrary
   character count), then split further by word budget.
4. **Anonymize** every real name to a consistent `Student NN` alias
   across the whole corpus — including casual first-name mentions inside
   messages, not just the sender-name prefix, and names of people who
   are only ever mentioned, never senders (see `EXTRA_NAMES_TO_ANONYMIZE`
   in that file if you run this on your own export and spot one).
5. **Safety-filter**: drops any chunk containing profanity/explicit
   language from what actually ships — nothing is silently discarded,
   filtered chunks stay in the traceability JSONL with
   `flagged_explicit: true`, just excluded from `domain_knowledge`.

Run it on your own export with `python domains/ingest_whatsapp.py path/to/export.txt`.

## Confidence is a belief, not a float

Every `SemanticFact.confidence` is backed by a `BetaBelief(alpha, beta)`
— alpha counts confirming evidence, beta counts disconfirming evidence.
`concentration` (`alpha + beta`) tells you how much evidence that number
rests on. Contradictions are measured as the KL cost of folding new
evidence into the existing belief (scales automatically with how much
evidence a fact already has) rather than gated by a fixed
`confidence >= 0.6` cutoff — see `memory/consolidator.py`'s docstring.

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Status

Working, tested (33 tests). Both the CLI and the web app run
end-to-end on real Gemini reasoning (via Google's Agent Development
Kit) once a key is set, not just the built-in stub — verified live,
including a full ask-vs-act / memory / contradiction-handling pass
through the actual web UI. Deployment (Cloud Run/Firestore) is
deliberately not live in this build — see the project's own notes for
why and what a real deploy would need.
