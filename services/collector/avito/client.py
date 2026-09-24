"""
Тонкий адаптер к fork ilyaievlev/parser_avito (ветка estate).

Снаружи: list[Apartment].
Внутри: collect_ads + to_ml_dicts из vendor.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .mapper import ml_dict_to_apartment
from .models import Apartment

_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "parser_avito"


def _ensure_vendor_on_path() -> None:
    root = str(_VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_vendor_on_path()

from apartment_ml import to_ml_dicts  # noqa: E402
from fetch_ml_data import collect_ads  # noqa: E402
from load_config import load_avito_config  # noqa: E402
from parser_cls import AvitoParse  # noqa: E402


class AvitoClient:
    def __init__(self, config_path: Path | None = None):
        path = config_path or (_VENDOR_ROOT / "config.toml")
        self.config = load_avito_config(str(path))
        self.parser = AvitoParse(self.config)

    def fetch_apartments(self) -> list[Apartment]:
        """Собрать объявления и привести к Apartment. Страницы = config.count."""
        ads = collect_ads(self.parser)
        rows = to_ml_dicts(ads)

        apartments: list[Apartment] = []
        for row in rows:
            try:
                apartments.append(ml_dict_to_apartment(row))
            except (ValueError, TypeError, KeyError) as err:
                logger.warning(f"skip id={row.get('id')}: {err}")

        logger.info(f"Item={len(rows)} → Apartment={len(apartments)}")
        return apartments


if __name__ == "__main__":
    client = AvitoClient()
    apartments = client.fetch_apartments()
    print(f"apartments: {len(apartments)}")
    if apartments:
        print(apartments[0].model_dump_json(indent=2))
