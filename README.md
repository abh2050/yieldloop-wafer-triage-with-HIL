# yieldloop

A wafer map defect triage console with a human decision loop, active learning,
and a grounded root cause agent.

yieldloop takes real wafer maps from the WM811K dataset, classifies the defect
pattern with a calibrated model, routes the cases a human should actually look
at to a keyboard-driven review console, and — for lots flagged for excursion
review — produces ranked root cause hypotheses that are forced to cite retrieved
evidence or abstain. Every human decision is captured as training signal for the
next round.

## Why it is built this way

Three claims carry the project, and each is measured rather than asserted:

1. **Active learning reduces labeling cost.** The label efficiency curve compares
   entropy-plus-diversity sampling against random sampling at matched label counts.
2. **Calibrated routing beats a fixed threshold.** The routing tradeoff curve sweeps
   the confidence floor and the uncertainty band, showing what automation rate costs
   in accuracy.
3. **A root cause agent is only useful if it cannot invent evidence.** Every
   hypothesis claim must resolve to an evidence ID present in the context bundle
   assembled for that request. Claims that do not resolve are dropped; if nothing
   survives, the system abstains explicitly rather than rendering an empty card.

## Guardrails

The agent is never called directly. `src/yieldloop/guardrails/` wraps it on both
sides and is not bypassable by a caller flag:

| Module | Responsibility |
| --- | --- |
| `input_filter` | Bounds and validates wafer IDs, grids, retention window, and free text before a token is spent |
| `injection` | Detects instruction-shaped content in reviewer notes and retrieved report text, neutralizes it by structural isolation |
| `schema_validator` | Parses agent output against a strict schema and fails closed |
| `grounding` | Drops any claim whose evidence IDs are not in the bundle; returns an abstention object if nothing survives |
| `thresholds` | Owns the routing bands: auto-commit, uncertainty band, and the floor below which the prediction is hidden from the reviewer |
| `budget` | Per-request token caps, per-session and daily spend ceilings, typed errors |
| `circuit_breaker` | Trips on schema failure streaks, grounding rejection rate, or latency breach; degrades to classifier-only mode |
| `audit` | Append-only record of every model output, human decision, guardrail action, and threshold change — enforced by revoked UPDATE/DELETE grants in the migration, not by convention |

## Data

All data is real. Wafer maps, defect labels, lot structure, and wafer index come
from the WM811K dataset, fetched by `scripts/fetch_dataset.py`. No wafer maps are
generated, no defect labels are invented, and no tool history is fabricated. Any
derived metadata is a deterministic function of real dataset fields, and every
derivation is documented in [docs/data_contract.md](docs/data_contract.md).

## Quickstart

```bash
cp .env.example .env          # then set OPENAI_API_KEY if you want the agent
docker compose up -d postgres
docker compose run --rm migrate

# Kaggle credentials required: ~/.kaggle/kaggle.json
python scripts/fetch_dataset.py       # needs ~/.kaggle/kaggle.json
python scripts/bootstrap_db.py        # 46,293 lots / 811,457 wafers
python -m scripts.train               # trains, calibrates, registers
python scripts/seed_from_real_labels.py
python scripts/run_active_round.py    # fills the review queue

docker compose up api frontend
```

The console is at http://localhost:5173, the API at http://localhost:8000.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest
```

Tests run against real infrastructure. Database tests start a real Postgres
container; OpenAI contract tests call the live API with a bounded token budget and
skip only when `OPENAI_API_KEY` is absent; frontend tests are Playwright runs
against the real API and database. There are no mocks, stubs, or fixtures
standing in for external services.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data_contract.md)
- [Guardrails](docs/guardrails.md)
- [Evaluation](docs/evaluation.md)
- [Runbook](docs/runbook.md)
