"""
Тонкий адаптер к fork ilyaievlev/parser_avito (ветка estate).

Снаружи: list[Apartment].
Внутри: collect_ads + to_ml_dicts из vendor.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from loguru import logger

from domain import Apartment

from .mapper import ml_dict_to_apartment

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "parser_avito"

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

# vendor пишет logs/, storage/ (cookies) и database.db относительно cwd,
# поэтому импорт и все вызовы vendor идут из его собственной папки.
with contextlib.chdir(_VENDOR_ROOT):
    from apartment_ml import to_ml_dicts
    from fetch_ml_data import collect_ads
    from load_config import load_avito_config
    from parser_cls import AvitoParse


class AvitoClient:
    name = "avito"

    def __init__(self, config_path: Path | None = None):
        path = config_path.resolve() if config_path else _VENDOR_ROOT / "config.toml"
        with contextlib.chdir(_VENDOR_ROOT):
            self.parser = AvitoParse(load_avito_config(str(path)))

    def fetch(self) -> list[Apartment]:
        """Собрать объявления и привести к Apartment. Страницы = config.count."""
        with contextlib.chdir(_VENDOR_ROOT):
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
