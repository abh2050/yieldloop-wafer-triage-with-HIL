# Data contract

Everything yieldloop stores falls into three categories. A row holds a real field
from the WM811K wafer map dataset, a real human action captured by the console,
or a derived value produced by a deterministic rule documented on this page. No
fourth category exists. The system generates no wafer map, invents no defect
label, and fabricates no fab event.

Derived rows carry their rule name in a `derivation_rule` column. The two tables
that someone could otherwise mistake for observed fab telemetry, `process_events`
and `historical_excursions`, carry a non-nullable `is_derived` column with a
database check constraint pinning it to true.

## 1. Source

| Property | Value (measured from the real archive) |
| --- | --- |
| Dataset | WM811K (MIR-WM811K) |
| File | `LSWMD.pkl`, a pickled pandas DataFrame, 2,095,505,977 bytes |
| sha256 | `1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65` |
| Wafers | 811,457 |
| Lots | 46,293 |
| Human-labeled wafers | 172,950 (21.3%) |
| Unlabeled wafers | 638,507 (78.7%) |
| Wafer map shapes | not fixed; 136 distinct shapes in a 20,000-row sample |
| Acquisition | `scripts/fetch_dataset.py` via the Kaggle API |
| Integrity | sha256 verified on every load |

The label distribution explains why this project is shaped around active learning
rather than around a supervised baseline. 78.7% of the dataset carries no human
label at all, and `none` dominates the labeled remainder.

| `failureType` | Count | Share of labeled |
| --- | --- | --- |
| `none` | 147,431 | 85.2% |
| `Edge-Ring` | 9,680 | 5.6% |
| `Edge-Loc` | 5,189 | 3.0% |
| `Center` | 4,294 | 2.5% |
| `Loc` | 3,593 | 2.1% |
| `Scratch` | 1,193 | 0.7% |
| `Random` | 866 | 0.5% |
| `Donut` | 555 | 0.3% |
| `Near-full` | 149 | 0.1% |

`Near-full` accounts for 0.09% of labeled wafers. The eval harness therefore
reports per-class recall separately rather than folding it into an accuracy
figure that a majority-class predictor would score 85% on.

### Two properties of the real file

**A Python 2 pickle written with pandas 0.x in 2019.** The file references module
paths that no longer exist, namely `pandas.indexes.base` and
`pandas.indexes.range`, and its strings are latin1. `yieldloop.ingest.loader`
maps the old module paths onto their current homes at unpickling time. This
project deliberately refuses to convert the file once and commit the result.
Doing so would place a derived artifact between the published dataset and every
metric, and the pinned sha256 would then verify our copy rather than the real
thing.

**A misspelled train/test column.** The dataset spells it `trianTestLabel`. The
loader reads that exact name and exposes it under the corrected one. It accepts
only that spelling, so a future release that fixes the typo surfaces as a
contract failure someone looks at rather than as a column that quietly starts
arriving empty.

### Real fields consumed

| Source column | Type | Where it lands |
| --- | --- | --- |
| `waferMap` | 2-D `uint8` array, values `{0,1,2}` | `wafers.grid` after normalization; `wafers.raw_height/raw_width` record the original shape |
| `dieSize` | float (numpy scalar) | `wafers.die_size` |
| `lotName` | string (numpy scalar) | `lots.lot_name` |
| `waferIndex` | float (numpy scalar) | `wafers.wafer_index` |
| `trianTestLabel` | `(1,1)` array of string, or `(0,0)` when absent | `wafers.dataset_split_label`, preserved but **not** used for partitioning |
| `failureType` | `(1,1)` array of string, or `(0,0)` when absent | `wafers.dataset_label` via `DefectPattern.from_dataset_label` |

The dataset stores optional fields per row as small numpy arrays. A present value
has shape `(1, 1)` and an absent one has shape `(0, 0)`. The loader reads an
empty array as unlabeled and never as a label.

Normalization preserves the `waferMap` encoding exactly. 0 marks a position
outside the wafer, 1 marks a passing die, and 2 marks a failing die. No step
reassigns these values.

Most rows in WM811K carry no `failureType`. That absence is not a defect in the
data. It is the reason active learning fits this problem, and the unlabeled
majority is the pool the sampler draws from.

## 2. Derivation rules

Each rule is a pure function of real fields. Given the same `LSWMD.pkl` and the
same `YIELDLOOP_PARTITION_SEED`, each produces identical output on every machine.

### `GRID_NORMALIZE`

**Inputs:** `waferMap`, `YIELDLOOP_GRID_HEIGHT`, `YIELDLOOP_GRID_WIDTH`.

Wafer maps in WM811K vary in shape. The rule resamples each map to the configured
grid by nearest-neighbour index mapping, which preserves the `{0,1,2}` alphabet
exactly. No interpolation runs, so no die acquires a value the source did not
contain. The rule counts `die_total` and `die_fail` on the raw map before
resampling, so the reported die statistics are the real counts rather than an
artifact of normalization.

### `LOT_ORDINAL`

**Inputs:** `lotName` across the whole dataset.

The rule sorts lots by `lotName` under a stable, locale-independent byte ordering
and assigns a 0-based ordinal. This is the only ordering WM811K admits, because
the dataset carries no timestamps.

### `LOT_DATE`

**Inputs:** `LOT_ORDINAL`, a fixed epoch of `2021-01-01`.

The rule computes `derived_date = epoch + lot_ordinal days`. It exists because
the retention window and the process-event window need a total order, and WM811K
has no calendar. The value is a synthetic index expressed as a date rather than a
production date, and nothing in the system treats it as one. The input filter
uses it to enforce the retention window, and the retrieval layer uses it to order
process events relative to a lot.

### `PARTITION_ASSIGN`

**Inputs:** `lotName`, `YIELDLOOP_PARTITION_SEED`, the train and validation
fractions.

The rule assigns a lot to a split by computing `blake2b(f"{seed}:{lot_name}")`,
taking the first 8 bytes as an unsigned integer, and mapping it onto `[0, 1)`. A
value below the train fraction gives `train`, a value below train plus validation
gives `val`, and anything else gives `holdout`. Every wafer inherits its lot's
split, so no lot straddles a split and the classifier can never be evaluated on a
wafer from a lot it trained on. The rule ignores the dataset's own
`trainTestLabel`, which does not respect lot boundaries.

### `DIE_STATISTICS`

**Inputs:** the raw `waferMap`.

The rule counts entries `!= 0` as `die_total` and entries `== 2` as `die_fail`,
and computes the failure rate as `die_fail / die_total`.

The dataset verifies this rule independently. WM811K's `dieSize` field holds the
die count for the wafer, and `die_total` computed from the map reproduces it
exactly on 60,000 of 60,000 sampled wafers. The check runs as a test in
`test_ingest_contract.py`, so a regression in the counting or in the
normalization ordering breaks the build instead of silently shifting every
failure rate in the system.

The radial and edge concentration statistics presented to the agent come from the
raw map, computed by binning die by normalized radius from the wafer centroid.
These values are measured rather than modelled.

### `EXCURSION_RESOLUTION`

**Inputs:** the real `failureType` labels of a lot's wafers.

A lot enters `historical_excursions` only when it has labeled wafers and one
`DefectPattern` accounts for the plurality of them. `observed_pattern` holds that
label and `pattern_share` holds its fraction, and both come from real labels. The
`resolved_cause` comes from a fixed mapping from defect pattern to cause
category, not from an inference:

| Observed pattern | Resolved cause | Why this mapping |
| --- | --- | --- |
| `center` | `chamber_condition` | Centre-weighted signatures track chamber-centred process non-uniformity |
| `donut` | `chamber_condition` | Annular signatures track radial non-uniformity |
| `edge_ring` | `upstream_process` | Full-edge rings track edge-exclusion and upstream handling geometry |
| `edge_loc` | `handling_mechanical` | Localized edge damage tracks contact and handling |
| `loc` | `tool_drift` | Localized clusters track a drifting single tool |
| `scratch` | `handling_mechanical` | Linear signatures track physical contact |
| `random` | `material_lot` | Unstructured failure tracks incoming material |
| `near_full` | `recipe_change` | Near-total failure tracks a gross process or recipe fault |
| `none` | `unknown` | No pattern, no attributable cause |

This table holds the entire semantic content of the cause labels. It is a
documented convention of this repository rather than fab knowledge, and the
console displays it as such. A fixed template generates `resolution_text` from
the real measured values of that lot, covering pattern, share, die failure rate,
and wafer count. Every claim in that text restates a real measurement.

### `PROCESS_EVENT_DERIVE`

**Inputs:** `LOT_ORDINAL`, `LOT_DATE`, the `waferIndex` structure within the lot,
and the lot's real die-failure statistics.

The rule derives process events so that the retrieval and grounding machinery has
a non-trivial evidence corpus with a known ground truth. That known ground truth
is what makes the grounding test suite meaningful, because the harness knows
exactly which evidence IDs exist and any citation to anything else is provably a
fabrication.

For each lot, the rule emits events from real structural facts only:

| Event | Emitted when | Attributes (all real measurements) |
| --- | --- | --- |
| `wafer_index_gap` | The lot's `waferIndex` values are non-contiguous | The missing indices, the lot's wafer count |
| `partial_lot` | The lot has fewer than 25 wafers | Wafer count |
| `failure_rate_step` | The lot's die failure rate differs from the previous lot by ordinal by more than one standard deviation of the dataset-wide rate | Both rates, the delta, the dataset standard deviation |
| `die_size_shift` | The lot's `dieSize` differs from the previous lot by ordinal | Both die sizes |

Every attribute holds a real value read from or counted in the dataset. A fixed
mapping assigns the event `category`: `wafer_index_gap` and `partial_lot` map to
`handling_mechanical`, `failure_rate_step` maps to `tool_drift`, and
`die_size_shift` maps to `recipe_change`. A fixed template over the real
attributes produces the `summary`.

No event names a tool, a chamber, a recipe, an operator, or a timestamp, because
the dataset contains none of those. The agent is therefore structurally unable to
cite one, and the guardrail suite asserts that it does not invent one.

## 3. Evidence identifiers

Every element the agent may cite carries a stable `evidence_id`:

| Kind | Format | Example |
| --- | --- | --- |
| `classifier_prediction` | `cp:{wafer_id}` | `cp:lot12345-7` |
| `die_statistics` | `ds:{wafer_id}` | `ds:lot12345-7` |
| `similar_lot` | `hx:{lot_name}` | `hx:lot00891` |
| `process_event` | `pe:{lot_name}:{ordinal}` | `pe:lot00891:2` |

The context builder hashes the set of IDs assembled for a request into
`hypothesis_requests.context_hash` and stores them verbatim in
`hypothesis_requests.evidence_ids`. The grounding gate resolves every citation
against exactly that set. It drops an unresolvable citation and audits the drop,
and when nothing survives it records the request as an abstention.

## 4. What the system never stores

- Generated or augmented wafer maps.
- Labels absent from WM811K and not produced by a human using this console.
- Tool names, chamber IDs, recipe names, operator IDs, or production timestamps.
- Any metric in the eval report computed against anything other than real human
  labels or real reviewer decisions.
