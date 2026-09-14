# Evaluation

The harness runs as a program rather than a notebook, and it exits non-zero on a
regression. A metric that only a human ever reads is a metric that drifts.

```bash
python -m eval.harness                     # everything available
python -m eval.harness --suite guardrail   # no database, dataset, or key needed
python -m eval.harness --update-baselines  # after a deliberate improvement
```

## Ground truth

The harness draws ground truth from two sources. Defect labels come from
WM811K's human annotations, and resolutions come from decisions real reviewers
made in this console. The harness contains no synthetic ground truth.

Where data is absent, the harness says so rather than reporting zero. A fresh
console with no decisions cannot have hypothesis precision measured, and
reporting that as `0.0` would make an unused install look like a broken agent.

## Why no report quotes accuracy alone

85% of labeled WM811K wafers are `none`. A model that predicts `none`
unconditionally scores about 85% accuracy and has learned nothing. Every report
prints the majority-class baseline beside the model's own figure for that
reason. Macro F1, balanced accuracy, and per-class recall separate a useful
classifier from a constant one.

The class distribution spans three orders of magnitude, from 147,431 `none` down
to 149 `near_full`, so the harness reports per-class recall individually rather
than folding it into any average.

## The two charts

### 1. Label efficiency

This chart answers whether entropy-plus-diversity sampling reaches a given
quality with fewer human labels than random sampling.

Both arms run through the same selection code path, where `RANDOM` acts as the
control, and the harness compares them at matched label counts. The harness
refuses to compare unmatched points. A strategy that reached macro F1 0.6 after
4,000 labels does not compare to one that reached it after 12,000, and averaging
over unmatched budgets is how active learning results get overstated.

The headline numbers are labels-to-target and the ratio between the arms. The
harness tracks rare-class recall separately, because active learning should help
most there and a random sampler struggles to find those examples at all.

The greedy strategies run deterministically. `seed` governs only tie-breaking and
the random arm, so error bars come from the random arm and from retraining
rather than from resampling a deterministic selector.

### 2. Routing tradeoff

This chart answers what automation costs in accuracy.

The harness sweeps `auto_commit_threshold` with `confidence_floor` held fixed and
reports escaped errors, meaning wrong predictions committed without a human ever
seeing them. A fab asks about that column before allowing any automation.

A second sweep moves the floor with auto-commit fixed, which answers how much of
what a human sees the console should show them.

Both parameters live in configuration rather than in constants, which is what
makes these sweeps possible without editing code.

## Suites

| Suite | Needs | Reports |
| --- | --- | --- |
| `classifier_suite` | Database, trained artifact | Accuracy, macro F1, per-class recall, ECE and MCE, reliability bins, both sweeps, escalation |
| `agent_suite` | Database, live API key | Grounding rate, abstention rate, fabricated citations, latency, precision@1 and @3, cost |
| `guardrail_suite` | Nothing | 55 adversarial checks |

## Metrics worth explaining

**Grounding rate** is the fraction of proposed hypotheses whose citations all
resolved. A value below 100% means the model attempted a fabrication and the
gate dropped those claims before any reviewer saw them. The trend over time is
what matters.

**Abstention rate** stays deliberately unminimized. Abstaining on a thin bundle
is the correct answer, and a rate of zero against sparse evidence would mean the
model is inventing support.

**Anchoring delta** is the override rate when the console hid the prediction,
minus the override rate when it showed the prediction. A large positive value
means visible predictions are buying agreement rather than earning it, and the
confidence floor should rise.

**Precision@k** measures hypotheses against causes a human actually accepted.
The harness excludes calls with no human resolution rather than counting them as
failures, because an unanswered case is not a wrong answer.

**Escaped error rate** counts errors the system auto-committed and never
reviewed. This single number says whether the current configuration is safe.

## The gate

`eval/baselines.json` holds the committed metric floors and a tolerance.
`eval_gate.yaml` runs the harness on every pull request and fails the build when
a metric regresses beyond that tolerance.

The gate does not treat unmeasured metrics as regressions, because skipping a
suite when the API key is absent must not fail a PR. A metric the harness did
measure, and that has dropped, fails the build.

`guardrail.passed` sits in the floors as a boolean. It is a property rather than
a rate, so one pass-through fails the gate regardless of how many other cases
the layer handled correctly.

Some metrics are lower-is-better, including calibration error, escaped errors,
and fabrication rate, and the comparison flips for them. `harness.py` lists them
explicitly rather than inferring the direction from the name.

Updating a baseline is a deliberate act, and the commit message carries the
reason.

## Reproducibility

Every artifact records the hash of the ordered wafer and label pairs it was fit
on, the git commit, whether the tree was dirty, the seed, and the
hyperparameters. A number in the report traces to a training run that someone
can repeat. The registry stores an artifact trained from a dirty tree and flags
it, so nobody can present such a result as reproducible.
