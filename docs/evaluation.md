# Evaluation

The harness is a runnable program, not a notebook, and it exits non-zero on a
regression. A metric that is only ever read by a human is a metric that drifts.

```bash
python -m eval.harness                     # everything available
python -m eval.harness --suite guardrail   # no database, dataset, or key needed
python -m eval.harness --update-baselines  # after a deliberate improvement
```

## Ground truth

Two sources, no third. Defect labels come from WM811K's human annotations;
resolutions come from decisions real reviewers made in this console. **There is
no synthetic ground truth anywhere in the harness.**

Where data is absent, the harness says so instead of reporting zero. A fresh
console with no decisions cannot have hypothesis precision measured, and
reporting that as `0.0` would make an un-used install look like a broken agent.

## Why accuracy is never reported alone

85% of labeled WM811K wafers are `none`. A model that predicts `none`
unconditionally scores about 85% accuracy and has learned nothing. Every report
prints the majority-class baseline beside the model's own figure for exactly this
reason, and the numbers that actually separate a useful classifier from a
constant one are macro F1, balanced accuracy, and per-class recall.

The class distribution spans three orders of magnitude — 147,431 `none` against
149 `near_full` — so per-class recall is reported individually rather than folded
into any average.

## The two charts

### 1. Label efficiency

Does entropy-plus-diversity sampling reach a given quality with fewer human
labels than random?

Measured by running both arms through the *same* selection code path — `RANDOM`
is the control, not a fallback — and comparing at **matched label counts**. The
harness refuses to compare unmatched points: a strategy that reached macro F1 0.6
after 4,000 labels is not comparable to one that reached it after 12,000, and
averaging over unmatched budgets is how active learning results get overstated.

The headline is labels-to-target and the ratio between arms. Rare-class recall is
tracked separately, because that is where active learning should help most and
where a random sampler struggles to find examples at all.

Note that the greedy strategies are deterministic: `seed` governs only
tie-breaking and the random arm. Error bars therefore come from the random arm
and from retraining, not from resampling a deterministic selector.

### 2. Routing tradeoff

What does automation cost in accuracy?

Sweeps `auto_commit_threshold` with `confidence_floor` held fixed, and reports
**escaped errors** — wrong predictions committed without a human ever seeing them.
That is the column a fab actually asks about before allowing any automation.

A second sweep moves the floor with auto-commit fixed, which answers a different
question: how much of what a human sees should be shown to them.

Both parameters are configuration rather than constants, which is what makes
these sweeps possible without editing code.

## Suites

| Suite | Needs | Reports |
| --- | --- | --- |
| `classifier_suite` | Database, trained artifact | Accuracy, macro F1, per-class recall, ECE/MCE, reliability bins, both sweeps, escalation |
| `agent_suite` | Database, live API key | Grounding rate, abstention rate, fabricated citations, latency, precision@1 and @3, cost |
| `guardrail_suite` | Nothing | 55 adversarial checks |

## Metrics worth explaining

**Grounding rate.** The fraction of proposed hypotheses whose citations all
resolved. Below 100% does not mean a reviewer saw something wrong — the gate
dropped those claims — it means the model attempted a fabrication. The trend is
what matters.

**Abstention rate.** Deliberately *not* minimized. Abstaining on a thin bundle is
correct; a rate of zero against sparse evidence would mean the model is inventing
support.

**Anchoring delta.** Override rate when the prediction was hidden, minus override
rate when it was shown. A large positive value means visible predictions are
buying agreement rather than earning it, and the confidence floor should rise.

**Precision@k.** Measured against causes a human actually accepted. Calls with no
human resolution are excluded rather than counted as failures — an unanswered
case is not a wrong answer.

**Escaped error rate.** Errors auto-committed and never reviewed. The single
number that says whether the current configuration is safe.

## The gate

`eval/baselines.json` holds the committed metric floors and a tolerance.
`eval_gate.yaml` runs the harness on every pull request and fails the build when
a metric regresses beyond that tolerance.

Metrics that could not be measured this run are not treated as regressions —
skipping a suite because the API key is absent must not fail a PR — but a metric
that *is* measured and has dropped will.

`guardrail.passed` is in the floors as a boolean. It is a property, not a rate:
one pass-through is a failure regardless of how many other cases were handled.

Some metrics are lower-is-better (calibration error, escaped errors, fabrication
rate) and the comparison flips for them; the list is explicit in `harness.py`
rather than inferred from the name.

Updating a baseline is a deliberate act with the reason in the commit message.

## Reproducibility

Every artifact records the hash of the ordered wafer/label pairs it was fit on,
the git commit, whether the tree was dirty, the seed, and the hyperparameters. A
number in the report traces to a training run that can be repeated. An artifact
trained from a dirty tree is stored but flagged, so a result can never be
presented as reproducible when it is not.
