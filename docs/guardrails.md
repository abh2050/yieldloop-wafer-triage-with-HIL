# Guardrails

No code calls the agent directly. Every path into the model runs through
`src/yieldloop/guardrails/` on both sides, and no caller flag turns the layer
off.

The layer fails closed. When a guardrail cannot establish that an output is safe
to show, the system returns an explicit abstention. A root cause hypothesis a
process engineer cannot trust costs them the time to investigate it and costs
the console its credibility for everything it says afterwards.

## Order of operations

```
request
  │
  ├─ input_filter      reject malformed input before a token is spent
  ├─ budget            refuse the call if it would breach a ceiling
  ├─ circuit_breaker   refuse if the breaker is open (degrade, don't fail)
  │
  ├─ retrieval ───────► context bundle (every item carries an evidence_id)
  ├─ injection         isolate every piece of untrusted text structurally
  │
  ├─ ══ MODEL CALL ══
  │
  ├─ schema_validator  strict parse, fail closed
  ├─ grounding         drop any claim citing evidence not in the bundle
  ├─ budget.record     write actual usage to the ledger
  ├─ circuit_breaker   feed the outcome back
  └─ audit             append to the tamper-evident chain
```

Validation runs first so that an oversized note or an out-of-window lot costs
nothing. Validating after assembling a context bundle would spend embeddings and
prompt tokens to discover something a regex could report.

## The gates

### `input_filter` bounds everything before the system spends

The filter matches wafer IDs, lot names, and reviewer IDs against anchored,
bounded patterns narrower than the database columns, so anything reaching the
database has already proven well formed. It length-checks free text after NFKC
normalization, because a payload that passes the limit and then expands once
normalized would defeat a pre-normalization check, and the limit exists to bound
prompt cost, which the system pays post-normalization. The filter refuses lots
outside the retention window rather than clamping them.

The filter never modifies text. `injection` neutralizes it.

### `injection` isolates untrusted text

Two kinds of untrusted text reach a prompt: reviewer notes, and the resolution
text on retrieved historical lots.

The gate defends by structural isolation. The text passes through unmodified,
wrapped in a delimited block declared as data, and the gate neutralizes any
occurrence of the fence inside the payload so the block cannot close early.

Stripping matched substrings fails three ways. Obfuscation evades it trivially.
It corrupts legitimate engineering prose, because a reviewer writing "ignore the
previous edge measurement" means it literally. It destroys the evidence that
would otherwise appear in the audit log.

Detection is a separate concern from isolation. A detection raises
`injection_suspected` and writes an audit record, and it does not block the
request. Isolation protects the prompt, and the gate applies it to every piece of
untrusted text whether or not anything was detected.

Matching runs against an NFKC-normalized copy with invisible characters removed,
so zero-width joiners, fullwidth variants, and bidirectional overrides do not
defeat detection. The gate never alters the original string.

### `schema_validator` parses strictly, with no lenient path

The validator rejects output that does not parse rather than recovering it
partially. A half-parsed root cause claim reaches an engineer looking exactly as
authoritative as a complete one, and nothing on screen indicates that a field
was dropped.

The validator does not strip markdown fences either. The agent is called with a
structured output schema, so a fenced response means the model did not honour the
response format. That is a contract failure worth surfacing and worth counting
toward the breaker.

The JSON schema handed to the model derives from the same Pydantic model the
parser enforces, so the two cannot drift apart.

### `grounding` is the hard gate

This guardrail is the one the project exists to demonstrate.

A language model asked for root cause hypotheses produces fluent, plausible,
well-structured hypotheses whether or not it holds evidence, and a process
engineer cannot separate the two by reading them. The defence that survives
contact with a real fab is refusing to render a claim that does not resolve to
evidence the system itself retrieved.

The rule is mechanical. Every cited `evidence_id` must appear in the context
bundle assembled for that request. An ID that does not resolve is a fabrication,
and the gate drops the hypothesis carrying it whole.

The gate does not repair partially. Keeping a hypothesis after discarding its bad
citation would leave a statement partly built on something invented, and the
surviving citations would lend it unearned credibility.

Dropping a middle hypothesis leaves a gap in the rank sequence, so the gate
re-ranks survivors densely while preserving their relative order. A gap would
violate the response schema and corrupt precision-at-k.

When nothing survives, the gate returns an explicit abstention object. The
console renders that as "insufficient evidence", which answers the engineer's
question.

### `thresholds` routes, and enforces the anchoring rule

Three bands sit between two boundaries, and both boundaries live in
configuration so the eval harness can sweep them.

| Band | Condition | Human? | Prediction shown? |
| --- | --- | --- | --- |
| `auto_commit` | `confidence >= auto_commit_threshold` | no | yes |
| `uncertainty_band` | `confidence >= confidence_floor` | yes | yes |
| `below_floor` | otherwise | yes | **no** |

The third band is easy to get wrong. Showing a low-confidence prediction anchors
a reviewer. A reviewer told "the model thinks this is a scratch, but it is not
sure" agrees more often than one shown the same wafer cold, and that agreement
launders the model's own error into the training set as a human label.

So below the floor the console renders no prediction at all, and
`decisions.prediction_was_shown` records which regime produced each decision,
which turns the anchoring effect into a measured quantity.

### `budget` enforces three ceilings

The gate caps spend per request, covering prompt and completion tokens, per
session, and per day.

Session and daily figures come from the `cost_ledger` table rather than from
process memory. An in-process counter resets on restart and does not travel
across workers, which makes it useless as a spend ceiling in the deployment this
system targets.

The gate checks cost before the call using an estimate built from the maximum
completion tokens the call could produce, and records cost after the call using
the usage the API actually reports. Under-estimating beforehand is what lets the
call being checked breach the cap. A call may reach a ceiling and may not exceed
it. `request_id` is unique in the ledger, so a retried write cannot double-count.

### `circuit_breaker` degrades rather than fails

Three independent conditions trip the breaker: a run of consecutive schema
failures, a grounding rejection rate above the configured ratio, or a p95 latency
breach.

Opening the circuit does not fail the page. It degrades the console to
classifier-only mode. Wafer maps, calibrated predictions, and the review queue
keep working, and the hypothesis panel renders as unavailable. A triage console
that returns 503 because a language model is misbehaving takes away the
reviewer's actual job, which does not depend on that model.

Rate and latency conditions require a full window, so a cold process does not
look like an outage. After a cooldown the breaker goes half-open and admits one
probe. A good probe closes it and a bad one reopens it.

### `audit` appends, and stays tamper evident

The gate records four categories: every model output, every human decision, every
guardrail action, and every threshold change.

The database enforces immutability. Postgres revokes UPDATE, DELETE, and TRUNCATE
on `audit_records`, and a `BEFORE UPDATE OR DELETE OR TRUNCATE` trigger raises
regardless of privilege. The trigger is necessary alongside the revoked grants
because the application connects as the table owner in the local stack, and an
owner retains implicit rights it can re-grant to itself.

A hash chain sits on top of that. Each record's digest covers its own content and
its predecessor's digest. Altering one record and recomputing the chain requires
rewriting every record after it, and `verify_chain()` reports exactly where the
break is. The digest covers every field that carries meaning, because someone
could alter a field left out of the digest without breaking the chain.

## What the test suite asserts

| Suite | Asserts |
| --- | --- |
| `test_injection_resistance.py` | 19 attack shapes across 8 signal classes are detected; obfuscated variants survive normalization; legitimate engineering prose is not flagged; no payload can close its own isolation block |
| `test_grounding_enforcement.py` | A single fabricated citation drops the hypothesis; partly fabricated claims drop whole; all-fabricated yields abstention rather than an empty card; ranks re-densify; malformed IDs are rejected a layer earlier |
| `test_threshold_routing.py` | Band boundaries hold; the prediction is hidden below the floor; uncalibrated scores are refused; the tradeoff curve is monotonic in both parameters |
| `test_budget_cap.py` | Token caps refuse before the call; session and daily ceilings read real committed spend; a call may reach the cap and may not exceed it; retried writes cannot double-count |
| `test_audit_trail.py` | All four event categories are recorded; the chain links and verifies; a payload altered with the trigger disabled is detected |
| `test_audit_immutability.py` | UPDATE, DELETE, TRUNCATE, and ORM mutation are all refused by a real Postgres; grants are revoked independently of the trigger; the sequence is GENERATED ALWAYS |

All of these run against real infrastructure. The database tests use a real
Postgres container rather than an in-memory substitute, because the guarantee
under test is the database's own behaviour. A fake session could be made to pass
while the deployed system silently allowed an UPDATE.
