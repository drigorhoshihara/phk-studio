"""
PHK Studio
Clinical Pharmacy Engine

Cache em memória da Clinical Knowledge Base.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.clinical_pharmacy_engine.knowledge_base.models import (
    KnowledgeEvidence,
    KnowledgeQuery,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CacheEntry:
    evidences: list[KnowledgeEvidence]
    expires_at: datetime


class ClinicalKnowledgeCache:
    """Cache simples com expiração."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError(
                "ttl_seconds não pode ser negativo.",
            )

        self.ttl_seconds = ttl_seconds

        self._entries: dict[
            str,
            CacheEntry,
        ] = {}

    def get(
        self,
        source_code: str,
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence] | None:
        key = self.build_key(
            source_code,
            query,
        )

        entry = self._entries.get(key)

        if entry is None:
            return None

        if entry.expires_at <= utc_now():
            self._entries.pop(
                key,
                None,
            )
            return None

        return deepcopy(
            entry.evidences,
        )

    def set(
        self,
        source_code: str,
        query: KnowledgeQuery,
        evidences: list[KnowledgeEvidence],
    ) -> None:
        key = self.build_key(
            source_code,
            query,
        )

        expires_at = utc_now() + timedelta(
            seconds=self.ttl_seconds,
        )

        self._entries[key] = CacheEntry(
            evidences=deepcopy(evidences),
            expires_at=expires_at,
        )

    def clear(self) -> None:
        self._entries.clear()

    def remove_expired(self) -> int:
        now = utc_now()

        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]

        for key in expired_keys:
            self._entries.pop(
                key,
                None,
            )

        return len(expired_keys)

    @staticmethod
    def build_key(
        source_code: str,
        query: KnowledgeQuery,
    ) -> str:
        domains = ",".join(
            sorted(
                domain.value
                for domain in query.domains
            )
        )

        agents = ",".join(
            sorted(
                " ".join(
                    agent.casefold().split()
                )
                for agent in query.related_agents
            )
        )

        return "|".join(
            [
                source_code.strip().casefold(),
                query.normalized_term(),
                domains,
                agents,
                query.language or "",
                query.country or "",
                str(query.limit_per_source),
            ]
        )