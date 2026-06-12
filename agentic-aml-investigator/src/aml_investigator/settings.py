"""Project settings.

All knobs live here (pydantic-settings). Override via environment variables with
the ``AML_`` prefix, e.g. ``AML_AGENT_MODEL=qwen3.5:9b`` to switch the backbone.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AML_", env_file=".env", extra="ignore")

    # --- LLM backbone ---
    agent_model: str = "granite4.1:8b"  # primary: fastest reliable tool-caller on 8 GB VRAM
    judge_model: str = "qwen3.5:9b"  # cross-family judge to reduce self-preference
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 8192
    # Per-request ceiling. A legit slow qwen json_schema call is ~200s; an Ollama
    # hang is unbounded. This catches the hang (call raises -> ladder falls back)
    # without killing real slow generations.
    request_timeout: float = 300.0

    # --- paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    warehouse_path: Path = PROJECT_ROOT / "data" / "aml.duckdb"
    sdn_csv_path: Path = PROJECT_ROOT / "data" / "raw" / "sdn.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    checkpoint_db_path: Path = PROJECT_ROOT / "artifacts" / "checkpoints" / "checkpoints.db"

    # --- ledger generation ---
    seed: int = 42
    n_accounts: int = 200
    n_days: int = 90
    ledger_end_date: str = "2026-05-31"

    # --- forensic thresholds ---
    ctr_threshold: float = 10_000.0  # US Currency Transaction Report threshold
    structuring_band: float = 0.85  # deposits in (band*threshold, threshold) count as sub-threshold
    sanctions_fuzzy_cutoff: int = 87  # rapidfuzz token_sort_ratio 0-100
    sql_row_limit: int = 50

    # --- graph guards ---
    investigator_recursion_limit: int = 16
    max_reflection_rounds: int = 1
    max_report_retries: int = 1


settings = Settings()
