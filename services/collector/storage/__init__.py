"""Storage for collector: Postgres (current state) + ClickHouse (snapshots)."""

from pathlib import Path

from dotenv import load_dotenv

# корень репо: services/collector/storage/__init__.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")
