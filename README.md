<div align="center">

# Yieldloop

**Triage defects on wafer maps. Learn from the engineer who reviews them.**

[![accuracy](https://img.shields.io/badge/holdout%20accuracy-96.36%25-1d6fd0?style=flat-square)](#results-on-the-holdout-split)
[![automated](https://img.shields.io/badge/committed%20without%20a%20human-87.75%25-1d6fd0?style=flat-square)](#the-human-loop)
[![escaped errors](https://img.shields.io/badge/escaped%20errors-0.44%25-1d6fd0?style=flat-square)](#the-human-loop)
[![fabricated citations](https://img.shields.io/badge/fabricated%20citations-0-1d6fd0?style=flat-square)](#the-guardrails)

[![tests](https://img.shields.io/badge/tests-403%20passing-2f8c46?style=flat-square)](#testing)
[![mocks](https://img.shields.io/badge/mocks-none-2f8c46?style=flat-square)](#testing)
[![python](https://img.shields.io/badge/python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![react](https://img.shields.io/badge/react-18-61dafb?style=flat-square&logo=react&logoColor=black)](frontend/package.json)
[![license](https://img.shields.io/badge/license-Apache%202.0-6b7280?style=flat-square)](LICENSE)

<br>

[![architecture](docs/diagrams/yieldloop-architecture.png)](docs/diagrams/yieldloop-architecture.html)

<sub><b><a href="docs/index.html">Open the interactive diagrams</a></b> to explore the architecture, the routing bands, the human loop, and the guarded agent call.</sub>

</div>

<br>

A semiconductor fab produces wafer maps faster than its engineers can read them.
WM811K, the real dataset this runs on, holds 811,457 of them, and 638,507 carry
no human label at all.

Yieldloop sorts each map into one of nine defect patterns and attaches a
calibrated confidence. It commits the 87.75% it is confident about. It sends the
rest to a review console that an engineer clears from the keyboard.

Where the model is least sure, the console withholds its guess, so the engineer
answers cold. Every answer an engineer gives returns to the training set. A
separate agent proposes root causes for the lots that get flagged, and it cites
only the evidence retrieved for it.

| | Measured on 26,741 held-out wafers |
| --- | --- |
| Accuracy | **96.36%**, against an 86.38% majority-class baseline |
| Macro F1 | **0.8307**, against 0.1030 |
| Committed without a human | **87.75%**, with 0.44% escaped errors |
| Labels to reach macro F1 0.60 | **8,000** with active learning. Random sampling had not reached it by 16,000 |
| Fabricated citations from the agent | **0** |
| Tests | **384 backend and 19 Playwright**, all running against real infrastructure |

This repository contains no dataset. It reads the real
[WM811K wafer map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map),
about 2 GB. `scripts/fetch_dataset.py` fetches the archive into `data/`, and
every load verifies its sha256. No code path substitutes generated data.

## The problem

A semiconductor fab produces wafer maps by the hundred thousand. Each map is a
grid that records which die passed electrical test and which failed. Failures
sometimes form a pattern, such as a ring at the edge, a scratch across the
middle, or a cluster in one corner. That pattern points at a cause: a handling
fault, a chamber drift, or a recipe change upstream.

Three conditions make the work expensive.

**The volume exceeds what anyone can review.** WM811K holds 811,457 wafers. A
process engineer cannot look at them, so most wafers go unexamined and real
excursions surface late.

**Labels cost more than data.** Of those 811,457 wafers, 638,507 carry no human
label, which is 78.7% of the dataset. Wafer maps are cheap and expert attention
is scarce, so any method that demands many labeled examples spends the one
resource the fab cannot spare.

**The rare classes carry the signal.** 85% of labeled wafers are `none`, meaning
no pattern. `near_full` appears on 149 wafers out of 811,457. A model that always
predicts "no pattern" scores 85% accuracy and finds nothing, so this project
reports macro F1, balanced accuracy, and per-class recall alongside accuracy.

## What Yieldloop does about it

The classifier assigns each wafer map one of nine defect patterns and attaches a
calibrated confidence, so the number means what it says.

The router compares that confidence against two thresholds. It commits the
confident predictions automatically, sends the ambiguous ones to an engineer,
and sends the least confident ones to an engineer with the model's guess
withheld.

The sampler picks which unlabeled wafers to ask about. It ranks candidates by
prediction entropy and embedding diversity rather than drawing a random sample.

The review service records every decision as training signal, including what the
reviewer could see at the moment they decided.

The agent explains flagged lots. It ranks root cause hypotheses, cites the
evidence behind each claim, and abstains when the evidence does not support one.

At the configured thresholds, on 26,741 held-out wafers from lots the model has
never seen, the system committed 87.8% of wafers without a human and let 0.44%
of errors escape review. Active learning reached macro F1 0.60 at 8,000 labels,
and random sampling had not reached it by 16,000. The agent produced zero
hypotheses citing evidence that did not exist.

## How it fits together

[![yieldloop architecture](docs/diagrams/yieldloop-architecture.png)](docs/diagrams/yieldloop-architecture.html)

The interactive version at
[`docs/diagrams/yieldloop-architecture.html`](docs/diagrams/yieldloop-architecture.html)
supports themes, pan and zoom, search, and relationship tracing. Its source spec
is [`yieldloop-architecture.json`](docs/diagrams/yieldloop-architecture.json).

The diagram makes several structural facts visible. The classifier and the
language model run on separate paths, so no wafer map ever reaches OpenAI. The
guardrails sit between retrieval and the model on both sides, so every edge into
the model passes through them. The arrow running from the console back into
Postgres closes the loop, because the next training round reads those decisions.

## How to run it

You need Docker, Python 3.12, Node 22, a Kaggle account, and optionally an
OpenAI API key.

### 1. Configure

```bash
cp .env.example .env
```

`.env` holds every tunable value, including the routing thresholds. No module
hardcodes them, which lets the eval harness sweep them.

### 2. Start Postgres and migrate

```bash
docker compose up -d postgres
docker compose run --rm migrate
```

### 3. Get the real dataset

```bash
python scripts/fetch_dataset.py
```

The script needs `~/.kaggle/kaggle.json`, which you create under Kaggle
Settings, API, Create New API Token. You also accept the dataset terms once in a
browser. Without credentials the script prints setup instructions and exits
non-zero. A generated fallback would let this repository publish metrics
computed against a file that is not the dataset, so no such path exists. The
script prints a sha256. Put that value in `YIELDLOOP_WM811K_SHA256`, and every
later load verifies it.

### 4. Load it

```bash
python scripts/bootstrap_db.py          # about 3 min: 46,293 lots, 811,457 wafers
```

The loader normalizes every map to a 64x64 grid and assigns each lot to train,
validation, or holdout. Re-running truncates the tables and reloads.

### 5. Train

```bash
python -m scripts.train                 # about 80 min on Apple Silicon
```

Training fits the network, fits calibration, and registers the artifact. Pass
`--train-limit N` for a faster run.

### 6. Build the retrieval corpus

```bash
python scripts/seed_from_real_labels.py --max-lots 5000     # about 15 s
```

### 7. Fill the review queues

```bash
python scripts/run_active_round.py --strategy entropy_diversity --batch-size 64
python scripts/route_predictions.py
```

`run_active_round.py` feeds the label gate with the most informative unlabeled
wafers. `route_predictions.py` runs the production path. It commits the
confident predictions, queues the rest to the confirm gate, and escalates any
lot carrying several confident non-`none` predictions to the escalation gate.

### 8. Run the console

```bash
docker compose up api frontend
```

The console runs at http://localhost:5173 and the API docs at
http://localhost:8000/docs.

### Evaluate

```bash
python -m eval.harness --report eval_report.md     # all suites
python -m eval.harness --suite guardrail           # no DB, dataset, or key needed
python -m scripts.run_label_efficiency             # about 36 min, the second chart
python -m scripts.export_case_study --skip-agent
```

## The human loop

The name describes the mechanism. The console has a label gate, a confirm gate,
and an escalation gate, and decisions from all three return to the model.

**The label gate.** Active learning picks the most informative unlabeled wafers.
The console shows no prediction here at any confidence, because this gate exists
to collect an independent human label. `run_active_round.py` fills it.

**The confirm gate.** Predictions below the auto-commit threshold go to an
engineer. Inside the uncertainty band the console shows the prediction. Below
the confidence floor the API omits the prediction fields entirely, so the client
never receives them. `route_predictions.py` fills this gate and commits
everything above the threshold, which it expresses by producing no task at all.

**The escalation gate.** Lots carrying several confident non-`none` predictions
receive ranked root cause hypotheses with inline evidence. One odd wafer is
noise. A pattern repeating across a lot points at a lot-level cause worth
investigating.

[![routing bands](docs/diagrams/yieldloop-routing-bands.png)](docs/diagrams/yieldloop-routing-bands.html)

The interactive version at
[`docs/diagrams/yieldloop-routing-bands.html`](docs/diagrams/yieldloop-routing-bands.html)
supports themes, pan and zoom, search, and relationship tracing. Its source spec
is [`yieldloop-routing-bands.json`](docs/diagrams/yieldloop-routing-bands.json).

Every decision returns as training signal. A reviewer label overrides the
dataset's own annotation for that wafer. A reviewer label on a previously
unlabeled wafer becomes a new training example. A reviewed wafer leaves the
unlabeled pool, so the sampler stops offering it. Reviewer labels stay scoped to
their split, so a decision on a holdout wafer never reaches training. Each
artifact records how many of its labels came from the console, which
distinguishes a human-corrected model from one trained only on WM811K.

Each decision also records what the reviewer could see when they made it, which
turns the anchoring effect into a measured quantity.

[![the human loop closing](docs/diagrams/yieldloop-human-loop.png)](docs/diagrams/yieldloop-human-loop.html)

The interactive version at
[`docs/diagrams/yieldloop-human-loop.html`](docs/diagrams/yieldloop-human-loop.html)
supports themes, pan and zoom, search, and relationship tracing. Its source spec
is [`yieldloop-human-loop.json`](docs/diagrams/yieldloop-human-loop.json).

## The console

![label gate](docs/screenshots/01-label-gate-keyboard-first.png)

The console has six screens. Three of them carry the product.

**LabelGrid** renders the label gate. It was built first and it runs from the
keyboard: a digit key labels the focused wafer and advances to the next one, so
one keystroke completes a decision. Wafer maps draw on canvas from the real
arrays. This screen shows no model prediction at any confidence, because the
gate collects an independent human label.

**TriageQueue and WaferDetail** render the confirm gate. They display
uncertainty-band predictions with their calibrated confidence against the
configured thresholds. Below the confidence floor the API response omits the
prediction, and the row reads "withheld" so the reviewer knows the model holds
an opinion back.

**HypothesisCard** renders the escalation gate. It ranks root cause hypotheses
and renders each claim's evidence as a clickable citation. An abstention renders
as "insufficient evidence" together with the reason.

**ModelHealth** reports calibration, override rates, and per-class recall.
**AuditTrail** shows the append-only log and verifies the hash chain live.

| | |
| --- | --- |
| ![triage](docs/screenshots/02-triage-queue-prediction-withheld.png) | ![model health](docs/screenshots/04-model-health-calibration-and-per-class-recall.png) |
| The triage queue. Every row reads **withheld**, because these wafers fell below the confidence floor and the API omits the fields | Model health. The screen reports calibration before and after fitting, and reports override rate twice so the anchoring effect stays visible |

![audit trail](docs/screenshots/05-audit-trail-hash-chain-intact.png)

The audit trail verifies its hash chain live. Postgres revokes `UPDATE`,
`DELETE`, and `TRUNCATE` on that table, and a trigger raises regardless of
privilege. The hash chain catches anything that bypasses both.

## The ML model

The classifier is a small convolutional network with 305,129 parameters, trained
from scratch on this dataset. The defect patterns are spatial signatures on a
64x64 grid, and a few convolutional blocks capture them. A larger backbone would
raise the ceiling slightly and would slow down iteration on calibration,
routing, and the human loop.

The input layer one-hot encodes die values into three channels. WM811K encodes 0
as outside the wafer, 1 as a passing die, and 2 as a failing die. These values
are categorical, and feeding the raw integers would impose an ordering the data
does not carry.

The network returns a 128-dimensional L2-normalized embedding alongside its
logits. The same vector drives diversity selection in active learning and drives
retrieval, so "similar wafer" means one thing across the system.

Training used 119,944 labeled wafers and early stopping on validation loss
selected epoch 19 of 23.

### Results on the holdout split

The holdout split holds 26,741 wafers from lots the model has never seen.
Partitioning keys on the lot, so no lot straddles a split.

| Metric | Model | Majority-class baseline |
| --- | --- | --- |
| Accuracy | **96.36%** | 86.38% |
| Macro F1 | **0.8307** | 0.1030 |
| Balanced accuracy | **87.35%** | 11.11% |

The baseline column explains why this project never quotes accuracy alone.

| Class | Recall | Holdout support |
| --- | --- | --- |
| none | 0.976 | 23,098 |
| edge_ring | 0.954 | 1,315 |
| random | 0.948 | 153 |
| center | 0.931 | 662 |
| near_full | 0.900 | 30 |
| loc | 0.853 | 497 |
| scratch | 0.803 | 157 |
| edge_loc | 0.763 | 780 |
| donut | 0.735 | 49 |

### Calibration

Every routing decision compares a confidence against a threshold, so the
confidences have to mean something. A raw softmax runs systematically
overconfident. An auto-commit threshold set against uncalibrated scores commits
a known fraction of errors unreviewed while looking rigorous.

Temperature scaling fits one scalar on the validation split. One parameter
cannot change which class the model predicts, so accuracy stays fixed by
construction and any improvement in calibration error is real.

| | Value |
| --- | --- |
| Fitted temperature | 0.9261 |
| Expected calibration error | 0.0097, then **0.0066** |
| Gap in the auto-commit band | **0.0030** over 23,465 predictions |

That last row bears on safety. Among predictions confident enough to commit
without review, stated confidence and actual accuracy differ by 0.3 percentage
points.

### Active learning

The label gate selects wafers by prediction entropy plus embedding diversity.
Entropy alone selects redundantly, because the most ambiguous wafers resemble
each other and a batch can spend a whole review round on near-duplicates of one
pattern. Greedy max-min selection spreads the batch across the embedding space.

Both arms shared splits, seed, and hyperparameters, and the comparison runs at
matched label counts.

| Labels | Active | Random | Delta |
| --- | --- | --- | --- |
| 1,000 | 0.2516 | 0.2516 | +0.0000 |
| 2,000 | 0.3965 | 0.3000 | **+0.0965** |
| 4,000 | 0.4809 | 0.3856 | **+0.0953** |
| 8,000 | 0.6174 | 0.4966 | **+0.1208** |
| 16,000 | 0.7908 | 0.5891 | **+0.2017** |

The gap widens as the budget grows. The active arm reaches 95% of full-data
quality from 13% of the labels.

The report carries two qualifications. The arms match exactly at 1,000 labels by
construction, because active learning has no model to select with until it has
labels and therefore starts from a random seed set. The random arm never reached
the target within 16,000 labels, so the report states the saving as a lower
bound of 2x rather than an exact ratio.

## What the OpenAI API does here

The API does not classify wafers. The local CNN described above does that, and
no wafer map is ever sent to OpenAI.

The system calls the API for one task. When a lot is flagged for excursion
review, a language model reads a context bundle assembled entirely from
retrieval and produces ranked root cause hypotheses. Each hypothesis carries a
mechanism, a confidence, citations, and concrete queries that would confirm or
eliminate it.

The classifier names the pattern, and an engineer still has to work out why the
pattern appeared. Answering that means correlating this lot against similar
historical lots and process events, which is reading and cross-referencing work
that a language model does well.

The same task is where a language model becomes dangerous. Asked for root
causes, a model produces fluent, plausible, well-structured hypotheses whether
or not it holds evidence, and an engineer cannot separate the two by reading
them.

### The guardrails

`src/yieldloop/guardrails/` wraps every call on both sides. No caller flag
disables it. The transport client knows nothing about grounding, budgets, or the
breaker, so a future caller cannot reach the model without them.

[![guarded hypothesis request](docs/diagrams/yieldloop-guarded-agent.png)](docs/diagrams/yieldloop-guarded-agent.html)

The interactive version at
[`docs/diagrams/yieldloop-guarded-agent.html`](docs/diagrams/yieldloop-guarded-agent.html)
supports themes, pan and zoom, search, and relationship tracing. Its source spec
is [`yieldloop-guarded-agent.json`](docs/diagrams/yieldloop-guarded-agent.json).

The ordering carries the design. Validation and the budget check run before the
system spends a token. Isolation runs while the prompt is built, so untrusted
text never reaches the instruction layer. Parsing and grounding run before
anything is persisted, so an ungrounded claim never becomes a database row. An
empty bundle abstains without calling the model, because a call that cannot be
grounded wastes the money.

| Gate | What it does |
| --- | --- |
| `input_filter` | Bounds every input before the system spends a token |
| `injection` | Isolates untrusted text structurally and leaves it unmodified |
| `schema_validator` | Parses strictly and fails closed |
| `grounding` | Drops any hypothesis citing evidence absent from the bundle |
| `thresholds` | Applies the three routing bands and withholds the prediction below the floor |
| `budget` | Enforces per-request token caps and per-session and daily spend ceilings |
| `circuit_breaker` | Degrades the console to classifier-only mode instead of failing the page |
| `audit` | Appends a record of every output, decision, guardrail action, and threshold change |

**Grounding carries the most weight.** Every claim must cite an `evidence_id`
present in the bundle assembled for that request. An unresolvable citation is a
fabrication, and the gate drops the hypothesis carrying it whole. The gate does
not repair citations, because keeping a claim after discarding its bad citation
leaves a statement partly built on something invented, and the surviving
citations then lend it unearned credibility. When nothing survives, the system
returns an explicit abstention.

**Injection defence works by isolation.** Reviewer notes and retrieved report
text pass through unmodified inside a fenced block declared as data, and the
gate neutralizes the fence so a payload cannot close it early. Stripping matched
substrings fails three ways: obfuscation evades it, it corrupts legitimate prose
because a reviewer writing "ignore the previous edge measurement" means it
literally, and it destroys the evidence that should reach the audit log.

**Measured on live calls,** the agent grounded 100% of surviving claims and
fabricated zero citations. The abstention rate stays deliberately unminimized,
because abstaining on a thin bundle is the correct answer and a rate of zero
against sparse evidence would mean the model is inventing support.

**Cost stays bounded and recorded.** Every call writes a row to a `cost_ledger`
table that the spend ceilings read, so the caps survive a restart and apply
across workers. Building and verifying this system spent $0.56 across 54 calls.

## Other functionality

**An append-only audit trail.** Postgres revokes UPDATE, DELETE, and TRUNCATE on
`audit_records`, and a trigger raises regardless of privilege. The trigger is
necessary because the application connects as the table owner, and an owner
retains implicit rights it can re-grant itself. A hash chain sits on top, so the
system still detects tampering that bypasses the application entirely.
`GET /audit/verify` reports the chain state live.

**A content-addressed model registry.** Every artifact records the hash of the
ordered wafer and label pairs it was fit on, the git commit, whether the tree was
dirty, the seed, and the hyperparameters. Any number in the eval report traces
to a run that someone can repeat.

**A telemetry worker.** It computes override rate, calibration drift against the
training-time figure, and input distribution drift, measured as population
stability index over the embeddings, on a schedule. It never changes behaviour.
Retuning thresholds silently would leave the running configuration out of step
with `threshold_changes`, and the audit trail would stop explaining the system.

**Anchoring measurement.** Every decision records whether the console showed the
prediction, so the override rate splits two ways. The gap between the two rates
measures the anchoring effect. When reviewers disagree far more often with the
prediction hidden, the visible predictions are buying agreement rather than
earning it, and the floor should rise.

**An evaluation gate.** `eval/baselines.json` holds 18 committed metric floors.
CI fails a pull request when a metric regresses beyond tolerance.

## Data provenance

All wafer maps and defect labels come from real WM811K data. Derived rows, such
as process events and historical excursions, carry a `derivation_rule` naming
the deterministic function that produced them. The two tables that someone could
mistake for observed fab telemetry carry a non-nullable `is_derived` column with
a database check constraint pinning it true.

No derived event names a tool, chamber, recipe, operator, or production
timestamp, because the dataset contains none of those. The agent is therefore
structurally unable to cite one.
[docs/data_contract.md](docs/data_contract.md) specifies every derivation.

One check deserves mention. WM811K carries a `dieSize` field holding the die
count per wafer. The ingest layer counts die independently and reproduces that
field exactly on 60,000 of 60,000 sampled wafers. This check runs as a test, so
a regression in counting or in normalization ordering breaks the build instead
of quietly shifting every failure rate in the system.

## Testing

The suite holds 384 backend tests and 19 Playwright tests. All of them pass and
none are skipped, including the ones that need the real dataset, a real
Postgres, and the live API. The suite uses no mocks, stubs, monkeypatching, or
fake fixtures.

- Database tests start a real Postgres container.
- Dataset tests read the real 2 GB archive.
- OpenAI contract tests call the live API under a token cap.
- Playwright drives the real API and the real database rather than a stubbed
  fetch layer.
- Hypothesis drives property-based tests over sampler, calibration, and drift
  invariants.

The guardrail integration tests use one scripted transport, which is the single
deliberate exception. Every guardrail still runs for real against it. The
scripted transport acts as a controlled model, because a live API will not
produce adversarial behaviour on demand.

```bash
pytest                                  # everything
pytest -m "not openai and not dataset"  # fast loop
pytest -m openai                        # live API, bounded budget
cd frontend && npx playwright test      # needs the API running
```

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | Layer ownership and the decisions that shaped it |
| [Data contract](docs/data_contract.md) | Every field and every derivation rule |
| [Guardrails](docs/guardrails.md) | Each gate, why it exists, and what it refuses |
| [Evaluation](docs/evaluation.md) | Method, and why each metric is reported the way it is |
| [Runbook](docs/runbook.md) | Operating it, and what to do when something trips |
| [Diagrams](docs/index.html) | All four system diagrams on one scrollable page |

## Stack

The backend runs on Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2,
PyTorch, FAISS, and the OpenAI SDK. The frontend runs on React 18, TypeScript,
Vite, and Tailwind. Postgres stores state and Docker Compose runs everything.

This project is licensed Apache-2.0. WM811K carries its own terms, which you
accept on the Kaggle dataset page.
