"""
PHK Studio
Clinical Pharmacy Engine

Contrato base das fontes de conhecimento clínico.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.clinical_pharmacy_engine.knowledge_base.models import (
    KnowledgeEvidence,
    KnowledgeQuery,
    KnowledgeSourceDescriptor,
    SourceStatus,
)


class ClinicalKnowledgeSource(ABC):
    """
    Interface comum para conectores clínicos.

    Nenhuma fonte deve retornar seus objetos nativos diretamente.
    Todos os dados devem ser convertidos para KnowledgeEvidence.
    """

    descriptor: KnowledgeSourceDescriptor

    @property
    def code(self) -> str:
        return self.descriptor.code

    @property
    def status(self) -> SourceStatus:
        return self.descriptor.status

    def is_available(self) -> bool:
        return self.status == SourceStatus.AVAILABLE

    @abstractmethod
    async def search(
        self,
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence]:
        """Executa uma consulta na fonte."""

    async def health_check(self) -> bool:
        """
        Verificação operacional simples.

        Conectores reais podem sobrescrever este método.
        """

        return self.is_available()