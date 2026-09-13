# Data contract

Everything yieldloop stores is either a **real field** from the WM811K wafer map
dataset, a **real human action** captured by the console, or a **derived value**
produced by a deterministic rule documented on this page. There is no fourth
category. No wafer map is generated, no defect label is invented, and no fab
event is fabricated.

Rows that are derived carry their rule name in a `derivation_rule` column, and
the two tables that could otherwise be mistaken for observed fab telemetry
(`process_events`, `historical_excursions`) carry a non-nullable `is_derived`
column with a database check constraint pinning it to true.

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

The label distribution is the reason this project is shaped around active
learning rather than around a supervised baseline: **78.7% of the dataset carries
no human label at all**, and the labeled remainder is dominated by `none`.

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

`Near-full` is 0.09% of labeled wafers. Per-class recall is therefore reported
separately in the eval harness rather than being folded into an accuracy figure
that a majority-class predictor would score 85% on.

### Two properties of the real file

**It is a Python 2 pickle written with pandas 0.x (2019).** It references module
paths that no longer exist (`pandas.indexes.base`, `pandas.indexes.range`) and
its strings are latin1. `yieldloop.ingest.loader` maps the old module paths onto
their current homes at unpickling time. Converting the file once and committing
the result is rejected deliberately: it would place a derived artifact between
the published dataset and every metric, and the pinned sha256 would then verify
our copy rather than the real thing.

**The train/test column is misspelled in the dataset**, as `trianTestLabel`. The
loader reads that exact name and exposes it under the corrected one. It does not
accept either spelling, so a future release that fixes the typo surfaces as a
contract failure to be looked at rather than as a column that quietly starts
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

Optional fields are stored per row as small numpy arrays: shape `(1, 1)` when a
value is present and `(0, 0)` when it is absent. An empty array means *unlabeled*
and is read as `None`, never as a label.

The `waferMap` encoding is preserved exactly: `0` is outside the wafer, `1` is a
passing die, `2` is a failing die. Normalization never reassigns these values.

Most rows in WM811K carry no `failureType`. That is not a defect in the data; it
is the reason active learning is the right shape for this problem, and the
unlabeled majority is the pool the sampler draws from.

## 2. Derivation rules

Each rule is a pure function of real fields. Given the same `LSWMD.pkl` and the
same `YIELDLOOP_PARTITION_SEED`, each produces identical output on every machine.

### `GRID_NORMALIZE`

**Inputs:** `waferMap`, `YIELDLOOP_GRID_HEIGHT`, `YIELDLOOP_GRID_WIDTH`.

Wafer maps in WM811K vary in shape. Each map is resampled to the configured grid
by nearest-neighbour index mapping, which preserves the `{0,1,2}` alphabet
exactly — no interpolation, so no die ever acquires a value the source did not
contain. `die_total` and `die_fail` are counted on the **raw** map before
resampling, so reported die statistics are the real counts and not an artifact
of normalization.

### `LOT_ORDINAL`

**Inputs:** `lotName` across the whole dataset.

Lots are sorted by `lotName` under a stable, locale-independent byte ordering and
assigned a 0-based ordinal. This is the only ordering WM811K admits: the dataset
carries no timestamps.

### `LOT_DATE`

**Inputs:** `LOT_ORDINAL`, a fixed epoch of `2021-01-01`.

`derived_date = epoch + lot_ordinal days`. This exists because the retention
window and the process-event window need a total order, and WM811K has no
calendar. It is a synthetic index expressed as a date, **not** a production date,
and nothing in the system treats it as one. It is used only for (a) retention
window enforcement in the input filter and (b) ordering process events relative
to a lot.

### `PARTITION_ASSIGN`

**Inputs:** `lotName`, `YIELDLOOP_PARTITION_SEED`, the train/val fractions.

A lot is assigned to a split by `blake2b(f"{seed}:{lot_name}")`, taking the first
8 bytes as an unsigned integer and mapping it onto `[0, 1)`. Below the train
fraction is `train`; below train + val is `val`; otherwise `holdout`. Every wafer
inherits its lot's split, so **no lot straddles a split** and the classifier can
never be evaluated on a wafer from a lot it trained on. The dataset's own
`trainTestLabel` is deliberately not used, because it does not respect lot
boundaries.

### `DIE_STATISTICS`

**Inputs:** the raw `waferMap`.

`die_total` counts entries `!= 0`; `die_fail` counts entries `== 2`.

This rule is independently verifiable against the dataset itself: WM811K's
`dieSize` field is the die count for the wafer, and `die_total` computed from the
map reproduces it **exactly on 60,000 of 60,000 sampled wafers**. The check runs
as a test (`test_ingest_contract.py`), so a regression in the counting or in
normalization ordering breaks the build rather than silently shifting every
failure rate in the system. Failure rate
is `die_fail / die_total`. Radial and edge concentration statistics presented to
the agent are computed from the raw map by binning die by normalized radius from
the wafer centroid. These are measured, not modelled.

### `EXCURSION_RESOLUTION`

**Inputs:** the real `failureType` labels of a lot's wafers.

A lot enters `historical_excursions` only when it has labeled wafers and one
`DefectPattern` accounts for the plurality of them. `observed_pattern` is that
label and `pattern_share` is its fraction — both measured from real labels. The
`resolved_cause` is a **fixed mapping** from defect pattern to cause category,
not an inference:

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

This table is the **entire** semantic content of the cause labels. It is a
documented convention of this repository, not fab knowledge, and the console
displays it as such. `resolution_text` is generated from the real measured values
of that lot (pattern, share, die failure rate, wafer count) by a fixed template;
it contains no claim that is not a restatement of a real measurement.

### `PROCESS_EVENT_DERIVE`

**Inputs:** `LOT_ORDINAL`, `LOT_DATE`, `waferIndex` structure within the lot, and
the lot's real die-failure statistics.

Process events are derived so that the retrieval and grounding machinery has a
non-trivial evidence corpus with a **known ground truth**, which is what makes
the grounding test suite meaningful: the harness knows exactly which evidence IDs
exist, so a citation to anything else is provably a fabrication.

For each lot, events are emitted from real structural facts only:

| Event | Emitted when | Attributes (all real measurements) |
| --- | --- | --- |
| `wafer_index_gap` | The lot's `waferIndex` values are non-contiguous | The missing indices, the lot's wafer count |
| `partial_lot` | The lot has fewer than 25 wafers | Wafer count |
| `failure_rate_step` | The lot's die failure rate differs from the previous lot by ordinal by more than one standard deviation of the dataset-wide rate | Both rates, the delta, the dataset standard deviation |
| `die_size_shift` | The lot's `dieSize` differs from the previous lot by ordinal | Both die sizes |

Every attribute is a real value read from or counted in the dataset. The event
`category` is assigned by a fixed mapping (`wafer_index_gap` and `partial_lot` to
`handling_mechanical`, `failure_rate_step` to `tool_drift`, `die_size_shift` to
`recipe_change`). The `summary` is a fixed template over the real attributes.

**No event names a tool, a chamber, a recipe, an operator, or a timestamp**,
because the dataset contains none of those. The agent is therefore structurally
unable to cite one, and the guardrail suite asserts that it does not invent one.

## 3. Evidence identifiers

Every element the agent may cite carries a stable `evidence_id`:

| Kind | Format | Example |
| --- | --- | --- |
| `classifier_prediction` | `cp:{wafer_id}` | `cp:lot12345-7` |
| `die_statistics` | `ds:{wafer_id}` | `ds:lot12345-7` |
| `similar_lot` | `hx:{lot_name}` | `hx:lot00891` |
| `process_event` | `pe:{lot_name}:{ordinal}` | `pe:lot00891:2` |

The set of IDs assembled for a request is hashed into
`hypothesis_requests.context_hash` and stored verbatim in
`hypothesis_requests.evidence_ids`. The grounding gate resolves every citation
against exactly that set. An unresolvable citation is dropped and the drop is
audited; if nothing survives, the request is recorded as an abstention.

## 4. What is never stored

- Generated or augmented wafer maps.
- Labels not present in WM811K and not produced by a human using this console.
- Tool names, chamber IDs, recipe names, operator IDs, or production timestamps.
- Any metric in the eval report computed against anything other than real human
  labels or real reviewer decisions.
