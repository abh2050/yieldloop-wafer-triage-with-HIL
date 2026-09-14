# Runbook

## First run

```bash
cp .env.example .env
docker compose up -d postgres
docker compose run --rm migrate

# Requires ~/.kaggle/kaggle.json. The script fails loudly without it, and no
# generated fallback dataset exists.
python scripts/fetch_dataset.py

python scripts/bootstrap_db.py               # 46,293 lots, 811,457 wafers
python -m scripts.train                      # see "Training" below
python scripts/seed_from_real_labels.py      # retrieval corpus and FAISS index
python scripts/run_active_round.py           # fill the review queue

docker compose up api frontend
```

The console runs at http://localhost:5173, the API at http://localhost:8000, and
the OpenAPI page at http://localhost:8000/docs.

## Health

| Endpoint | Answers |
| --- | --- |
| `GET /health/live` | Is the process up (it touches nothing else) |
| `GET /health/ready` | Is the database reachable and does it hold wafers |
| `GET /health/model` | Calibration, routing configuration, override rates |
| `GET /audit/verify` | Is the audit hash chain intact |

## Common situations

### The console shows "classifier-only mode"

The circuit breaker has opened. The system is degrading rather than failing.
Wafer maps, predictions, and the review queue keep working, and only hypothesis
generation is unavailable.

Check `GET /audit/guardrails?stage=circuit_breaker` for the trip reason. It
reports a schema-failure streak, a grounding rejection rate above the configured
ratio, or a p95 latency breach. The breaker half-opens automatically after
`YIELDLOOP_BREAKER_COOLDOWN_SECONDS` and closes on a good probe.

Do not raise the thresholds to make it close. A breaker that trips is reporting
something real.

### Every hypothesis request abstains

Check `GET /hypothesis/lot/{lot_name}` first. It reports what evidence exists
and spends nothing. An empty bundle abstains without calling the model.

If evidence exists and the gate still drops claims, look at
`GET /audit/guardrails?stage=grounding`. The `unresolvable_ids` field lists
exactly which citations failed to resolve, which shows directly what the model
tried to invent.

### Spend has hit a ceiling

The API returns a typed error with `reason: budget_exceeded`. The ledger is
authoritative:

```sql
SELECT spend_date, sum(cost_usd) FROM cost_ledger GROUP BY 1 ORDER BY 1 DESC;
```

The caps are `YIELDLOOP_SESSION_COST_CAP_USD` and
`YIELDLOOP_DAILY_COST_CAP_USD`.

### Temperature is pinned at a bound

The model health screen flags this. Calibration failed to fit and the confidences
are not trustworthy. The usual cause is an undertrained network, because class
weighting flattens logits and undertraining flattens them further.

Do not tune the routing thresholds against such an artifact. Retrain first.

### Override rate is climbing

Compare the override rate against the override-rate-when-shown on the model
health screen. If the overall rate rises while the shown rate stays flat, the
model is degrading on the hard cases. If both rise together, look for input drift
in `drift_snapshots.input_psi`.

### The audit chain reports broken

Treat this as serious. A record's content no longer matches what was hashed when
it was written, and the application cannot produce that state. Postgres revokes
UPDATE, DELETE, and TRUNCATE, and a trigger raises regardless of privilege. A
break means someone disabled the trigger or edited the table out of band.

`GET /audit/verify` names the sequence numbers where the chain breaks.

## Changing the routing thresholds

The thresholds live in configuration rather than in code. Sweep them first:

```bash
python -m eval.harness --suite classifier --report routing.md
```

Read the routing table, pick a point, set `YIELDLOOP_CONFIDENCE_FLOOR` and
`YIELDLOOP_AUTO_COMMIT_THRESHOLD`, and record the change with its rationale in
`threshold_changes` so the audit trail carries the reasoning.

The floor must stay strictly below auto-commit. `Settings` refuses to start
otherwise.

## Training

```bash
python -m eval.harness --suite classifier   # score the current artifact first
```

Training reads train and validation only. Expect roughly six minutes per epoch on
Apple Silicon over 119,944 labeled wafers, with early stopping on validation
loss.

The registry stores artifacts content-addressed under `YIELDLOOP_REGISTRY_DIR`,
and exactly one per kind stays active. The database determines which model is
current, not whichever file a script happened to load.

## Exporting the case study

```bash
python -m eval.harness --report eval_report.md
python -m scripts.export_case_study --skip-agent --output case_study.md
```

`--skip-agent` omits the suite that spends tokens. Both commands read only what
the system recorded. Neither estimates anything, and both say so where a number
is unavailable.

## Backups

Replay cannot restore `audit_records`. The table is append-only by design and its
hash chain will not survive a partial restore. Back up the whole database, and
verify the chain after any restore.
