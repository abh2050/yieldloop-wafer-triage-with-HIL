# Architecture

yieldloop is a wafer map defect triage console. It classifies defect patterns,
decides which cases a human should actually look at, captures those decisions as
training signal, and — for lots flagged for excursion review — produces root
cause hypotheses that are forced to cite retrieved evidence or abstain.

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

### Partitioning is keyed on the lot, not the wafer

Wafers in one lot share process history and look nearly identical. Splitting at
the wafer level puts near-duplicates on both sides of the boundary and inflates
every number in the eval report. Partitioning hashes the **lot name**, so no lot
straddles a split — asserted by a test against the real archive.

The hash also makes assignment stable under dataset growth: adding lots never
moves an existing one, so a model trained last month can still be evaluated on a
holdout it provably never saw. WM811K's own `trainTestLabel` is deliberately
unused, because it does not respect lot boundaries.

### Holdout is read in exactly one place

Training fits on train. The temperature fits on validation. Early stopping
watches validation. Only `eval/suites/classifier_suite.py` reads holdout. Keeping
that to a single file is what makes the holdout numbers mean anything.

### Calibration is upstream of everything

Every routing decision is a comparison between a confidence and a threshold, so
those confidences must be meaningful. A raw softmax is systematically
overconfident; an auto-commit threshold set against uncalibrated scores would
commit a known fraction of errors unreviewed while looking rigorous.

Temperature scaling has one parameter, which is the point: it cannot change which
class is predicted, so accuracy is unchanged by construction and any improvement
in calibration error is real rather than a re-fit.

### The routing bands are configuration

`confidence_floor` and `auto_commit_threshold` live in `Settings`, never as
module constants, so the eval harness can sweep them and produce the routing
tradeoff curve without editing code. A validator enforces a non-empty uncertainty
band.

### Below the floor, the prediction is not sent

The band structure has three regions, and the third is the one that is easy to
get wrong. Showing a low-confidence prediction does not help a reviewer, it
anchors them — and the resulting agreement is the model's error laundered into
the training set as a human label.

So the decision is made in the queue service, not the frontend: below the floor
the prediction fields are `None` on the way out, and no template, debug view, or
API consumer can reach them. `decisions.prediction_was_shown` records which
regime each decision was made under, so the anchoring effect is measured rather
than assumed away.

### The agent is unreachable except through the guardrails

The transport client knows nothing about grounding, budgets, or the breaker.
Composing those is `agent/hypothesis.py`'s job, and keeping the transport
ignorant is what stops a future caller from reaching the model without them. See
[guardrails.md](guardrails.md).

### Provenance is carried in the schema

Any row that is not a direct record of a real WM811K field or a real human action
carries a `derivation_rule`. The two tables that could otherwise be mistaken for
observed fab telemetry carry a non-nullable `is_derived` column with a check
constraint pinning it true. Documentation can go stale; a check constraint cannot.

### The audit log is immutable at the database

`audit_records` has UPDATE, DELETE, and TRUNCATE revoked **and** a trigger that
raises regardless of privilege — the trigger is necessary because the application
connects as the table owner, and an owner retains implicit rights it can re-grant
itself. On top sits a hash chain, so tampering by something that bypasses the
application entirely is still detectable.

## Request path: one hypothesis

1. `POST /hypothesis` validates the payload and resolves the lot.
2. `retrieval.context_builder` assembles evidence. Every item gets an
   `evidence_id`; nothing is synthesized to fill a gap.
3. `HypothesisService.generate` runs the gates in order: input filter, breaker,
   empty-bundle check, isolation, budget, model call, strict parse, grounding,
   ledger write, breaker feedback, audit.
4. Only grounded hypotheses are persisted, so every citation row is resolvable by
   construction.
5. The console renders the result, or the abstention, or the degraded state.

An empty bundle abstains **without calling the model**. Paying for a call that
cannot possibly be grounded is pure waste.

## Technology

Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2, PyTorch, FAISS, the
OpenAI SDK. React 18 with TypeScript, Vite, and Tailwind. Postgres for state.
Everything runs under Docker Compose.

## Testing posture

No mocks, no stubs, no fixtures standing in for external services. Database tests
start a real Postgres; the dataset tests read the real archive; OpenAI contract
tests call the live API and skip only when the key is absent; Playwright drives
the real API and database.

The one deliberate exception is the scripted transport in
`tests/guardrails/test_guarded_agent_path.py`. It is not a mock of the
guardrails — every guardrail runs for real — it is a *controlled model*, because
adversarial model behaviour cannot be provoked on demand from a live API.
