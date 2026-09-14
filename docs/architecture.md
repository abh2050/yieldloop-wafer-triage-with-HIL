# Architecture

yieldloop classifies wafer map defect patterns, decides which cases a human
should look at, captures those decisions as training signal, and produces root
cause hypotheses for lots flagged for excursion review. Each hypothesis cites
retrieved evidence, and the agent abstains when the evidence does not support
one.

## Shape

```
   WM811K archive (real, 811,457 wafers)
            │
   ingest ──┴──► Postgres ──► models ──► sampling ──► review queue ──► console
                    │            │                         │              │
                    │            └──► retrieval ──┐        └──► decisions ┘
                    │                              │              │
                    └──────────────────────────────┴──► agent ────┘
                                                    (behind guardrails)
                                                          │
                                                        audit
```

## What each layer owns

| Layer | Owns | Does not |
| --- | --- | --- |
| `ingest` | Reading the real archive, normalizing maps, lot-keyed partitioning | Know about models or the API |
| `models` | Training, calibration, embeddings, the artifact registry | Decide routing |
| `sampling` | Which unlabeled wafers a reviewer sees next | Perform I/O of any kind |
| `retrieval` | The FAISS index and evidence assembly | Interpret evidence |
| `agent` | Prompt construction and one model call | Reach the model unguarded |
| `guardrails` | Every gate on both sides of the agent | Be optional |
| `review` | Queue state, decisions, reason codes | Contain HTTP concerns |
| `api` | Validate, delegate, translate errors | Contain business logic |
| `telemetry` | Override rate, calibration drift, input drift | Change behaviour |

## The decisions that shaped it

### Partitioning keys on the lot, not the wafer

Wafers in one lot share process history and look nearly identical. Splitting at
the wafer level places near-duplicates on both sides of the boundary and
inflates every number in the eval report. The partitioner hashes the lot name,
so no lot straddles a split. A test asserts this against the real archive.

The hash also keeps assignment stable as the dataset grows. Adding lots never
moves an existing one, so someone can evaluate a model trained last month on a
holdout it provably never saw. The partitioner ignores WM811K's own
`trainTestLabel`, which does not respect lot boundaries.

### Exactly one file reads the holdout split

Training fits on train. Temperature scaling fits on validation. Early stopping
watches validation. Only `eval/suites/classifier_suite.py` reads holdout.
Keeping that access in a single file is what gives the holdout numbers meaning.

### Calibration sits upstream of everything

Every routing decision compares a confidence against a threshold, so those
confidences have to mean something. A raw softmax runs systematically
overconfident. An auto-commit threshold set against uncalibrated scores commits
a known fraction of errors unreviewed while looking rigorous.

Temperature scaling fits one parameter, and that constraint carries the point.
One scalar cannot change which class the model predicts, so accuracy stays fixed
by construction and any improvement in calibration error is real rather than a
re-fit.

### The routing bands live in configuration

`confidence_floor` and `auto_commit_threshold` live in `Settings` rather than as
module constants, so the eval harness sweeps them and draws the routing tradeoff
curve without anyone editing code. A validator enforces a non-empty uncertainty
band.

### Below the floor, the API does not send the prediction

The band structure has three regions, and the third one is easy to get wrong.
Showing a low-confidence prediction anchors a reviewer, and the resulting
agreement launders the model's own error into the training set as a human label.

The queue service makes that decision, not the frontend. Below the floor the
prediction fields leave the service as `None`, so no template, debug view, or
API consumer can reach them. `decisions.prediction_was_shown` records which
regime produced each decision, which turns the anchoring effect into a measured
quantity.

### Every path to the agent runs through the guardrails

The transport client knows nothing about grounding, budgets, or the breaker.
`agent/hypothesis.py` composes those gates. Keeping the transport ignorant is
what stops a future caller from reaching the model without them. See
[guardrails.md](guardrails.md).

### The schema carries provenance

Any row that is not a direct record of a real WM811K field or a real human
action carries a `derivation_rule`. The two tables that someone could otherwise
mistake for observed fab telemetry carry a non-nullable `is_derived` column with
a check constraint pinning it true. Documentation goes stale, and a check
constraint does not.

### The database enforces audit immutability

Postgres revokes UPDATE, DELETE, and TRUNCATE on `audit_records`, and a trigger
raises regardless of privilege. The trigger is necessary because the application
connects as the table owner, and an owner retains implicit rights it can
re-grant itself. A hash chain sits on top, so the system still detects tampering
that bypasses the application entirely.

## Request path: one hypothesis

1. `POST /hypothesis` validates the payload and resolves the lot.
2. `retrieval.context_builder` assembles evidence. Every item receives an
   `evidence_id`, and the builder synthesizes nothing to fill a gap.
3. `HypothesisService.generate` runs the gates in order: input filter, breaker,
   empty-bundle check, isolation, budget, model call, strict parse, grounding,
   ledger write, breaker feedback, audit.
4. The service persists only grounded hypotheses, so every citation row resolves
   by construction.
5. The console renders the result, the abstention, or the degraded state.

An empty bundle abstains without calling the model. A call that cannot possibly
be grounded wastes the money.

## Technology

The backend runs on Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2,
PyTorch, FAISS, and the OpenAI SDK. The frontend runs on React 18 with
TypeScript, Vite, and Tailwind. Postgres holds state. Docker Compose runs
everything.

## Testing posture

The suite contains no mocks, stubs, or fixtures standing in for external
services. Database tests start a real Postgres. Dataset tests read the real
archive. OpenAI contract tests call the live API and skip only when the key is
absent. Playwright drives the real API and the real database.

`tests/guardrails/test_guarded_agent_path.py` uses a scripted transport, which
is the one deliberate exception. Every guardrail still runs for real against it.
The scripted transport acts as a controlled model, because a live API will not
produce adversarial behaviour on demand.
