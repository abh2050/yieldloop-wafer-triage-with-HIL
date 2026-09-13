# Guardrails

The agent is never called directly. Every path into the model goes through
`src/yieldloop/guardrails/`, on both sides, and there is no caller flag that
turns it off.

The layer **fails closed**. When a guardrail cannot establish that an output is
safe to show, the system returns an explicit abstention rather than a degraded
answer. A root cause hypothesis a process engineer cannot trust is worse than no
hypothesis, because it costs them the time to investigate it and the credibility
of everything the console says afterwards.

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

Validation comes first for a reason: an oversized note or an out-of-window lot
should cost nothing. Validating after assembling a context bundle would mean
paying for embeddings and prompt tokens to discover something a regex could have
told us.

## The gates

### `input_filter` — bound everything before spending

Wafer IDs, lot names, and reviewer IDs are matched against anchored, bounded
patterns narrower than the database columns, so anything reaching the database
has already been shown to be well formed. Free text is length-checked **after**
NFKC normalization, because a payload that passes the limit and then expands once
normalized would defeat a pre-normalization check, and the limit exists to bound
prompt cost, which is paid post-normalization. Lots outside the retention window
are refused rather than clamped.

The filter never modifies text. Neutralizing it is `injection`'s job.

### `injection` — isolate, do not sanitize

Two kinds of untrusted text reach a prompt: reviewer notes, and the resolution
text on retrieved historical lots.

The defence is **structural isolation, not removal**. Stripping matched
substrings is wrong three times over: it is trivially evaded by obfuscation, it
corrupts legitimate engineering prose (a reviewer writing *"ignore the previous
edge measurement"* means it literally), and it destroys the evidence that would
otherwise appear in the audit log. So the text passes through unmodified, wrapped
in a delimited block declared as data, with any occurrence of the fence inside
the payload neutralized so the block cannot be closed early.

Detection is a separate concern from isolation. A detection raises
`injection_suspected` and writes an audit record; it does **not** block the
request. Isolation is what actually protects the prompt, and it is applied to
every piece of untrusted text whether or not anything was detected.

Matching runs against an NFKC-normalized copy with invisible characters removed,
so zero-width joiners, fullwidth variants, and bidirectional overrides do not
defeat detection. The original string is never altered.

### `schema_validator` — strict, with no lenient path

Output that does not parse is rejected outright rather than partially recovered.
A half-parsed root cause claim reaches an engineer looking exactly as
authoritative as a complete one, with no indication that a field was dropped.

Markdown fences are not stripped either. The agent is called with a structured
output schema, so a fenced response means the model did not honour the response
format — a contract failure worth surfacing and worth counting toward the
breaker, not quietly papering over.

The JSON schema handed to the model is derived from the same Pydantic model the
parser enforces, so the two cannot drift apart.

### `grounding` — the hard gate

This is the guardrail the project exists to demonstrate.

A language model asked for root cause hypotheses will produce fluent, plausible,
well-structured hypotheses whether or not it has evidence, and a process engineer
cannot tell the two apart by reading them. The only defence that survives contact
with a real fab is refusing to render a claim that does not resolve to evidence
the system itself retrieved.

The rule is mechanical: every cited `evidence_id` must be present in the context
bundle assembled for *that* request. An ID that does not resolve is not a
formatting problem to repair — it is a fabrication, and the hypothesis carrying
it is dropped whole.

**Partial repair is explicitly not done.** Keeping a hypothesis after discarding
its bad citation would leave a statement partly built on something invented, with
the surviving citations lending it unearned credibility.

Dropping a middle hypothesis leaves a gap in the rank sequence, so survivors are
re-ranked densely while preserving relative order — a gap would violate the
response schema and corrupt precision-at-k.

If nothing survives, the result is an explicit abstention object. The console
renders that as "insufficient evidence", which is a useful answer. An empty card
is not.

### `thresholds` — routing, and the anchoring rule

Three bands, two boundaries, both configuration so the eval harness can sweep
them:

| Band | Condition | Human? | Prediction shown? |
| --- | --- | --- | --- |
| `auto_commit` | `confidence >= auto_commit_threshold` | no | yes |
| `uncertainty_band` | `confidence >= confidence_floor` | yes | yes |
| `below_floor` | otherwise | yes | **no** |

The third band is the one that is easy to get wrong. Showing a low-confidence
prediction to a reviewer does not help them, it anchors them. A reviewer told
"the model thinks this is a scratch, but it is not sure" agrees more often than
one shown the same wafer cold — and that agreement is not signal, it is the
model's error laundered into the training set as a human label.

So below the floor the console does not render the prediction at all, and
`decisions.prediction_was_shown` records which regime each decision was made
under, so the anchoring effect can be measured rather than assumed away.

### `budget` — three ceilings

Per request (prompt and completion tokens), per session, and per day.

Session and daily figures are read from the `cost_ledger` table, not from process
memory. An in-process counter resets on restart and is not shared across workers,
which makes it useless as a spend ceiling in the deployment this is built for.

Cost is checked **before** the call using an estimate built from the *maximum*
completion tokens the call could produce, and recorded **after** using the usage
the API actually reports. Under-estimating beforehand is exactly what lets the
call being checked breach the cap. The caps are ceilings that may be reached but
not exceeded, and `request_id` is unique in the ledger so a retried write cannot
double-count.

### `circuit_breaker` — degrade, do not fail

Three independent trip conditions: a run of consecutive schema failures, a
grounding rejection rate above the configured ratio, or a p95 latency breach.

Opening the circuit does not fail the page. It degrades the console to
classifier-only mode: wafer map, calibrated prediction, and review queue keep
working, and the hypothesis panel renders as unavailable. A triage console that
returns 503 because a language model is misbehaving is worse than one that drops
the feature the model powers, because the reviewer's actual job does not depend
on it.

Rate and latency conditions require a full window, so a cold process does not
look like an outage. After a cooldown the breaker goes half-open and admits one
probe; a good probe closes it, a bad one reopens it.

### `audit` — append only, and tamper evident

Four categories are recorded: every model output, every human decision, every
guardrail action, and every threshold change.

Immutability is a **database** property. `audit_records` has UPDATE, DELETE, and
TRUNCATE revoked, and a `BEFORE UPDATE OR DELETE OR TRUNCATE` trigger that raises
regardless of privilege. The trigger is necessary alongside the revoked grants
because the application connects as the table owner in the local stack, and an
owner retains implicit rights it can re-grant to itself.

On top of that sits a hash chain: each record's digest covers its own content and
its predecessor's digest. Altering a record is already impossible; altering one
and recomputing the chain requires rewriting every record after it, and
`verify_chain()` reports exactly where the break is. The digest covers every
field that carries meaning — a field left out could be altered without breaking
the chain.

## What the test suite asserts

| Suite | Asserts |
| --- | --- |
| `test_injection_resistance.py` | 19 attack shapes across 8 signal classes are detected; obfuscated variants survive normalization; legitimate engineering prose is not flagged; no payload can close its own isolation block |
| `test_grounding_enforcement.py` | A single fabricated citation drops the hypothesis; partly-fabricated claims drop whole; all-fabricated yields abstention, not an empty card; ranks re-densify; malformed IDs are rejected a layer earlier |
| `test_threshold_routing.py` | Band boundaries; the prediction is hidden below the floor; uncalibrated scores are refused; the tradeoff curve is monotonic in both parameters |
| `test_budget_cap.py` | Token caps refuse before the call; session and daily ceilings read real committed spend; the cap may be reached but not exceeded; retried writes cannot double-count |
| `test_audit_trail.py` | All four event categories are recorded; the chain links and verifies; a payload altered with the trigger disabled is detected |
| `test_audit_immutability.py` | UPDATE, DELETE, TRUNCATE, and ORM mutation are all refused by a real Postgres; grants are revoked independently of the trigger; the sequence is GENERATED ALWAYS |

All of these run against real infrastructure. The database tests use a real
Postgres container, not an in-memory substitute, because the guarantee under test
*is* the database's behaviour — a fake session could be made to pass while the
deployed system silently allowed an UPDATE.
