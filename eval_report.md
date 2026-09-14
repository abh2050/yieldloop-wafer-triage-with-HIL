# yieldloop evaluation report

Every metric here comes from the real WM811K dataset and from decisions real 
reviewers made in this console. This report contains no synthetic ground truth.

**Gate: PASS** · 16.7s

## Classifier

Evaluated on 26,741 labeled wafers from the **holdout** split. 
Holdout is read only here; training fits on train and the temperature fits on 
validation, and because partitioning is lot-keyed these are lots the model has 
never seen.

| Metric | Model | Majority-class baseline | Reading |
| --- | --- | --- | --- |
| Accuracy | 96.36% | 86.38% | Misleads on its own, because 85% of labels are `none` |
| Macro F1 | 0.8307 | 0.1030 | Weights every class equally, so rare classes count |
| Balanced accuracy | 87.35% | 11.11% | Averages recall across classes with equal weight |


### Per-class recall

Reported separately because the class distribution spans three orders of 
magnitude. A model can score well on accuracy while never predicting 
`near_full` at all.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| none | 0.994 | 0.976 | 0.985 | 23,098 |
| edge_ring | 0.977 | 0.954 | 0.965 | 1,315 |
| edge_loc | 0.736 | 0.763 | 0.749 | 780 |
| center | 0.830 | 0.931 | 0.877 | 662 |
| loc | 0.560 | 0.853 | 0.676 | 497 |
| scratch | 0.586 | 0.803 | 0.677 | 157 |
| random | 0.843 | 0.948 | 0.892 | 153 |
| donut | 0.679 | 0.735 | 0.706 | 49 |
| near_full | 1.000 | 0.900 | 0.947 | 30 |


### Calibration

Every routing decision compares a confidence against a threshold, so those 
confidences have to mean something. Temperature scaling cannot change which 
class is predicted, so accuracy above is unaffected and the improvement below 
is real.

The last row bears on whether automation is safe. It measures only the band 
that auto-commit governs rather than averaging across a range most predictions 
never reach. Max calibration error ignores bins holding fewer than 30 
predictions, because a two-sample bin admits accuracies of only 0, 0.5 or 1 
and its apparent gap carries noise.

| Metric | Value |
| --- | --- |
| Fitted temperature | 0.9261 |
| ECE before calibration | 0.0097 |
| ECE after calibration | 0.0066 |
| Max calibration error | 0.0853 (worst bin holding at least 30 predictions) |
| Gap in the auto-commit band | 0.0030 over 23,465 predictions |


### Routing tradeoff

**Chart 1 of 2.** Sweeping the auto-commit threshold with the confidence floor 
held at 0.55. 
`escaped errors` are wrong predictions committed without a human ever seeing 
them, which is the column a fab actually cares about.

| Auto-commit | Automated | Accuracy when automated | Escaped errors | Human load |
| --- | --- | --- | --- | --- |
| 0.56 | 98.18% | 97.28% | 2.67% | 486 |
| 0.60 | 97.59% | 97.53% | 2.41% | 645 |
| 0.64 | 97.01% | 97.76% | 2.18% | 799 |
| 0.68 | 96.44% | 97.94% | 1.99% | 951 |
| 0.72 | 95.88% | 98.17% | 1.75% | 1,103 |
| 0.76 | 95.19% | 98.40% | 1.52% | 1,287 |
| 0.80 | 94.50% | 98.61% | 1.31% | 1,471 |
| 0.84 | 93.53% | 98.81% | 1.11% | 1,730 |
| 0.88 | 92.09% | 99.11% | 0.82% | 2,115 |
| 0.92 | 90.10% | 99.33% | 0.60% | 2,647 |
| 0.96 | 86.45% | 99.56% | 0.38% | 3,624 |
| 1.00 | 0.00% | 0.00% | 0.00% | 26,741 |


### Current configuration

| Metric | Value |
| --- | --- |
| Automation rate | 87.75% |
| Escalation rate | 12.25% |
| Blind review share | 13.61% of human work has the prediction withheld |
| Escaped error rate | 0.44% |


### Label efficiency

**Chart 2 of 2.** Entropy-plus-diversity selection against random, compared 
only at matched label counts. Both arms use the same splits, hyperparameters 
and seed and differ only in which wafers were chosen.

The simulation runs the active arm iteratively. The selector at each step 
trains only on what it has acquired so far. Selecting in one shot with a model 
that had seen the whole dataset would leak the dataset into the selection, and 
that leak is how this experiment usually gets reported wrongly.

| Labels | Macro F1 (active) | Macro F1 (random) | Delta |
| --- | --- | --- | --- |
| 1,000 | 0.2516 | 0.2516 | +0.0000 |
| 2,000 | 0.3965 | 0.3000 | +0.0965 |
| 4,000 | 0.4809 | 0.3856 | +0.0953 |
| 8,000 | 0.6174 | 0.4966 | +0.1208 |
| 16,000 | 0.7908 | 0.5891 | +0.2017 |


| Summary | Value |
| --- | --- |
| Mean macro F1 delta | +0.1029 |
| Labels to reach macro F1 0.6 | active 8,000 |
|  | random never |
| Label saving | not reached by both arms |


> This report **omits** rare-class recall and says so here. The smallest rare class has 8 wafers in this evaluation set, below the 30 a recall figure needs to carry meaning. At that size recall takes only a few discrete values, so movement between the arms reflects quantization rather than signal, and an average is only as trustworthy as its weakest term. The table above reports per-class recall on the full holdout.

> The two arms are **identical at 1,000 labels** by construction, not by coincidence: active learning has no model to select with until it has labels, so it starts from a random seed set of that size. The matching scores confirm the harness is comparing what it claims. That point contributes a zero to the mean delta, so the mean understates the effect; the comparison begins at the second budget.

> Random never reached macro F1 0.6 within 16,000 labels, so an exact saving ratio cannot be computed. The measurable statement is a lower bound: active reached it at 8,000, so the saving is **at least 2.0x**, and the true figure requires extending the random arm.

## Root cause agent

Grounding rate reports the fraction of proposed hypotheses whose citations all 
resolved to evidence in the context bundle. A rate below 100% means the model 
attempted a fabrication and the gate dropped those claims before any reviewer 
saw them.

Abstention rate stays deliberately **unminimized**. Abstaining on a thin bundle 
is the correct answer, and a rate of zero against sparse evidence would mean the 
model is inventing support.

| Metric | Value |
| --- | --- |
| Calls | 4 |
| Hypotheses proposed | 2 |
| Hypotheses grounded | 2 |
| Grounding rate | 100.00% |
| Abstention rate | 75.00% |
| Fabricated citations | 0 |
| Median latency | 2744 ms |
| p95 latency | 3337 ms |
| Cost | $0.0333 |


> No reviewer resolutions exist for the evaluated lots, so hypothesis precision cannot be measured. This is a data gap, not a score of zero.

## Reviewer agreement

Override rate alone reads two ways. A low rate can mean the model is good, or 
it can mean reviewers are deferring to it. Splitting the rate by whether the 
console showed the prediction separates the two, so every decision records what 
the reviewer could see.

A large positive anchoring delta means reviewers disagreed far more often when 
they could not see the prediction. Visible predictions are then buying agreement 
rather than earning it, and the confidence floor should rise.

| Metric | Value |
| --- | --- |
| Decisions captured | 13 |
| With a model prediction to compare | 13 |
| Override rate (all) | 84.62% |
| Override rate (prediction shown) | 0.00% |
| Override rate (prediction withheld) | 84.62% |
| Anchoring delta | not measurable, because one regime has no decisions |
| Median decision time | 0.01 s |
| p95 decision time | 0.02 s |


> Fewer than 30 decisions. These rates are reported for completeness but are not yet a measurement, and the eval gate does not hold them to a floor.

## Guardrails

55 adversarial checks: **PASS**

The suite asserts a property rather than a rate. The layer must fail closed on 
every hostile input, so a single pass-through fails the suite regardless of how 
many other cases it handled. The suite runs without a database or an API key, so 
it gates every pull request.

