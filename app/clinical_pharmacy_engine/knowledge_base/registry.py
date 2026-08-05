"""
PHK Studio
Clinical Pharmacy Engine

Registro central das fontes clínicas.
"""

from __future__ import annotations

from app.clinical_pharmacy_engine.knowledge_base.source import (
    ClinicalKnowledgeSource,
)


class ClinicalKnowledgeRegistry:
    """Mantém e resolve os conectores clínicos disponíveis."""

    def __init__(self) -> None:
        self._sources: dict[
            str,
            ClinicalKnowledgeSource,
        ] = {}

    def register(
        self,
        source: ClinicalKnowledgeSource,
        replace: bool = False,
    ) -> None:
        code = self._normalize_code(
            source.code,
        )

        if code in self._sources and not replace:
            raise ValueError(
                f"A fonte '{code}' já está registrada.",
            )

        self._sources[code] = source

    def unregister(
        self,
        code: str,
    ) -> ClinicalKnowledgeSource | None:
        normalized = self._normalize_code(code)

        return self._sources.pop(
            normalized,
            None,
        )

    def get(
        self,
        code: str,
    ) -> ClinicalKnowledgeSource:
        normalized = self._normalize_code(code)

        try:
            return self._sources[normalized]

        except KeyError as error:
            available = ", ".join(
                self.list_codes(),
            )

            raise KeyError(
                f"Fonte clínica não registrada: '{normalized}'. "
                f"Fontes disponíveis: {available or 'nenhuma'}."
            ) from error

    def list_codes(self) -> list[str]:
        return sorted(self._sources)

    def list_sources(
        self,
    ) -> list[ClinicalKnowledgeSource]:
        return [
            self._sources[code]
            for code in self.list_codes()
        ]

    def contains(
        self,
        code: str,
    ) -> bool:
        return (
            self._normalize_code(code)
            in self._sources
        )

    def __len__(self) -> int:
        return len(self._sources)

    @staticmethod
    def _normalize_code(
        value: str,
    ) -> str:
        normalized = value.strip().casefold()

        if not normalized:
            raise ValueError(
                "O código da fonte não pode ser vazio.",
            )

        return normalized