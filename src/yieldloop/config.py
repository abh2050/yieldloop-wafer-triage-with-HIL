"""Process configuration.

Every tunable in yieldloop lives here and is read from the environment. In
particular the routing bands (:attr:`Settings.confidence_floor` and
:attr:`Settings.auto_commit_threshold`) are configuration rather than module
constants, so the evaluation harness can sweep them and emit the routing
tradeoff curve without editing code.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Immutable, validated process settings."""

    model_config = SettingsConfigDict(
        env_prefix="YIELDLOOP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON

    # --- database ----------------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+psycopg://yieldloop:yieldloop@localhost:5432/yieldloop")
    )
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_pool_max_overflow: int = Field(default=5, ge=0, le=100)
    db_statement_timeout_ms: int = Field(default=15_000, ge=100, le=600_000)

    # --- dataset -----------------------------------------------------------
    data_dir: Path = Path("./data")
    kaggle_dataset: str = "qingyi/wm811k-wafer-map"
    wm811k_filename: str = "LSWMD.pkl"
    wm811k_sha256: str = ""

    # --- ingest ------------------------------------------------------------
    grid_height: int = Field(default=64, ge=8, le=512)
    grid_width: int = Field(default=64, ge=8, le=512)
    partition_seed: int = 20260913
    partition_train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    partition_val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    lot_retention_days: int = Field(default=365, ge=1)

    # --- model registry ----------------------------------------------------
    registry_dir: Path = Path("./data/registry")
    train_seed: int = 20260913
    embedding_dim: int = Field(default=128, ge=8, le=4096)

    # --- routing bands -----------------------------------------------------
    auto_commit_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    confidence_floor: float = Field(default=0.55, gt=0.0, le=1.0)

    # --- active learning ---------------------------------------------------
    round_batch_size: int = Field(default=64, ge=1, le=4096)
    diversity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    diversity_min_distance: float = Field(default=0.10, ge=0.0, le=2.0)

    # --- retrieval ---------------------------------------------------------
    faiss_index_path: Path = Path("./data/registry/wafer.index")
    retrieval_top_k: int = Field(default=8, ge=1, le=128)

    # --- agent -------------------------------------------------------------
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-2024-08-06"
    agent_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    agent_max_hypotheses: int = Field(default=3, ge=1, le=10)

    # --- guardrails: budget ------------------------------------------------
    max_prompt_tokens: int = Field(default=12_000, ge=256)
    max_completion_tokens: int = Field(default=2_000, ge=64)
    session_cost_cap_usd: float = Field(default=1.00, gt=0.0)
    daily_cost_cap_usd: float = Field(default=25.00, gt=0.0)
    input_cost_per_mtok_usd: float = Field(default=2.50, ge=0.0)
    output_cost_per_mtok_usd: float = Field(default=10.00, ge=0.0)

    # --- guardrails: input filter -----------------------------------------
    max_free_text_chars: int = Field(default=2_000, ge=1, le=100_000)

    # --- guardrails: circuit breaker --------------------------------------
    breaker_schema_failure_streak: int = Field(default=3, ge=1)
    breaker_grounding_reject_ratio: float = Field(default=0.30, gt=0.0, le=1.0)
    breaker_latency_p95_seconds: float = Field(default=20.0, gt=0.0)
    breaker_window_size: int = Field(default=20, ge=2)
    breaker_cooldown_seconds: float = Field(default=300.0, gt=0.0)

    # --- api ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:5173"

    @property
    def partition_holdout_fraction(self) -> float:
        return 1.0 - self.partition_train_fraction - self.partition_val_fraction

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def wm811k_path(self) -> Path:
        return self.data_dir / self.wm811k_filename

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_height, self.grid_width)

    @model_validator(mode="after")
    def _check_partition_fractions(self) -> Self:
        holdout = self.partition_holdout_fraction
        if holdout <= 0.0:
            raise ValueError(
                "partition_train_fraction + partition_val_fraction must be < 1.0 so that a "
                f"holdout split exists; got holdout fraction {holdout:.4f}"
            )
        return self

    @model_validator(mode="after")
    def _check_routing_bands(self) -> Self:
        if self.confidence_floor >= self.auto_commit_threshold:
            raise ValueError(
                "confidence_floor must be strictly below auto_commit_threshold so that a "
                f"non-empty uncertainty band exists; got floor={self.confidence_floor} "
                f"auto_commit={self.auto_commit_threshold}"
            )
        return self

    @model_validator(mode="after")
    def _check_cost_caps(self) -> Self:
        if self.session_cost_cap_usd > self.daily_cost_cap_usd:
            raise ValueError(
                "session_cost_cap_usd must not exceed daily_cost_cap_usd; got "
                f"session={self.session_cost_cap_usd} daily={self.daily_cost_cap_usd}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
