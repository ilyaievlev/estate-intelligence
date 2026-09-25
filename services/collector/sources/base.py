from typing import Protocol

from domain import Apartment


class Source(Protocol):
    """Источник объявлений: Avito, Cian, ... Каждый отдаёт общую модель Apartment."""

    name: str

    def fetch(self) -> list[Apartment]: ...
