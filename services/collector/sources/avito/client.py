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

# Сторонний парсер сохраняет cookies и logs относительно своего каталога,
# поэтому импорт и вызовы выполняются из его рабочей папки.
with contextlib.chdir(_VENDOR_ROOT):
    from apartment_ml import to_ml_dicts
    from fetch_ml_data import collect_ads
    from load_config import load_avito_config
    from parser_cls import AvitoParse


class AvitoClient:
    """Клиент сбора объявлений с Avito через встроенный парсер."""
    name = "avito"

    def __init__(self, config_path: Path | None = None):
        path = config_path.resolve() if config_path else _VENDOR_ROOT / "config.toml"
        with contextlib.chdir(_VENDOR_ROOT):
            self.parser = AvitoParse(load_avito_config(str(path)))

    def fetch(self) -> list[Apartment]:
        """Собрать объявления и привести их к доменной модели Apartment."""
        with contextlib.chdir(_VENDOR_ROOT):
            ads = collect_ads(self.parser)
        rows = to_ml_dicts(ads)

        apartments: list[Apartment] = []
        for row in rows:
            try:
                apartments.append(ml_dict_to_apartment(row))
            except (ValueError, TypeError, KeyError) as err:
                logger.warning(f"Пропуск объявления id={row.get('id')}: {err}")

        logger.info(f"Получено объявлений: {len(rows)} → Сформировано объектов: {len(apartments)}")
        return apartments
